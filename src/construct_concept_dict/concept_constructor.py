import base64
import json
import os
from typing import Dict, Tuple
from PIL import Image
import h5py
import numpy as np
import openai
import torch
from tqdm import tqdm

from credential import OPENAIKEY
from src.conceptscope.sae_module import SAEModule
from src.utils.image_dataset_loader import ImageDatasetLoader
from src.utils.processor import BatchIterator, H5ActivationtWriter, ImageProcessor
from src.utils.utils import (
    apply_sae_mask_to_input,
    get_sae_mask,
    load_class_names,
    plot_images,
)

openai.api_key = OPENAIKEY


class ConceptDictConstructor(SAEModule):
    def __init__(
        self,
        save_root,
        sae_path,
        device,
        backbone,
        dir_name="dataset_analysis",
        checkpoint_dir_name="checkpoints",
        data_root="./data",
        dataset_name="imagenet",
        split="train",
        num_top_images=100,
        seed=1,
    ):
        super().__init__(
            save_root=save_root,
            sae_path=sae_path,
            device=device,
            backbone=backbone,
            dir_name=dir_name,
            checkpoint_dir_name=checkpoint_dir_name,
            data_root=data_root,
        )
        self.dataset_name = dataset_name
        self.split = split
        self.dataset = ImageDatasetLoader.load_dataset(dataset_name, seed=seed, split=split)
        self.class_labels = np.array(self.dataset[self.label_key])
        self.num_classes = len(np.unique(self.class_labels))
        self.num_top_images = num_top_images
        self.class_names = load_class_names(data_root, dataset_name, self.dataset)

        save_dir = f"{self.save_root}/{self.dir_name}/{dataset_name}/{self.checkpoint_name}"
        os.makedirs(save_dir, exist_ok=True)
        self.h5_path = f"{save_dir}/{split}_sae_latents.h5"
        self.concept_dict_path = f"{self.checkpoint_path}/concept_dict.json"

        if os.path.exists(self.concept_dict_path):
            with open(self.concept_dict_path, "r") as f:
                self.concept_dict = json.load(f)
        else:
            self.concept_dict = {}

    def initialize_storage_tensors(self):
        return {
            "max_activating_image_values": torch.zeros([self.cfg.d_sae, self.num_top_images]).to(self.device),
            "max_activating_image_indices": torch.zeros([self.cfg.d_sae, self.num_top_images]).to(self.device),
            "sae_sparsity": torch.zeros([self.cfg.d_sae]).to(self.device),
            "sae_mean_acts": torch.zeros([self.cfg.d_sae]).to(self.device),
        }

    def _get_top_activations(self, sae_activations: torch.Tensor, batch_indices):
        """Get top activating images and their indices."""
        top_k = min(self.num_top_images, sae_activations.size(1))
        values, indices = torch.topk(sae_activations, k=top_k, dim=1)
        sampled_indices = batch_indices[indices.cpu()]
        return values, sampled_indices

    def _get_new_top_k(
        self,
        first_values: torch.Tensor,
        first_indices: torch.Tensor,
        second_values: torch.Tensor,
        second_indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get top k values and indices from two sets of values/indices."""
        total_values = torch.cat([first_values, second_values], dim=1)
        total_indices = torch.cat([first_indices.cpu(), torch.tensor(second_indices)], dim=1)
        new_values, indices_of_indices = torch.topk(total_values, k=self.num_top_images, dim=1)
        new_indices = torch.gather(total_indices, 1, indices_of_indices.cpu())
        return new_values, new_indices

    def _update_storage(self, storage, sae_activations: torch.Tensor, batch_indices: np.ndarray):
        sae_activations = sae_activations.transpose(0, 1)
        mean_acts = sae_activations.sum(dim=1)
        sparsity = (sae_activations > 0).sum(dim=1)
        storage["sae_mean_acts"] += mean_acts
        storage["sae_sparsity"] += sparsity

        top_values, top_indices = self._get_top_activations(sae_activations, batch_indices)
        storage["max_activating_image_values"], storage["max_activating_image_indices"] = self._get_new_top_k(
            storage["max_activating_image_values"],
            storage["max_activating_image_indices"],
            top_values,
            top_indices,
        )

    def process_class(
        self,
        storage,
        dataset,
        class_indices,
        activation_writer,
        cls_idx,
        batch_size=64,
        save_features=True,
        activation_threshold=0.5,
    ):

        batch_iterator = BatchIterator.get_batches(dataset, batch_size)
        num_samples = len(class_indices)
        activation_writer.create_cls_dataset(cls_idx, num_samples)

        class_activations = np.zeros((num_samples, self.cfg.d_sae), dtype=np.float32)
        active_counts = np.zeros(self.cfg.d_sae, dtype=np.int32)

        for start_idx, end_idx, batch in batch_iterator:
            batch_inputs = ImageProcessor.process_model_inputs(batch, self.vit, self.device)
            batch_indices = class_indices[start_idx:end_idx]

            sae_latents = ImageProcessor.get_sae_activations(
                self.sae,
                self.vit,
                batch_inputs,
                self.cfg.block_layer,
                self.cfg.module_name,
                self.cfg.class_token,
                get_mean=False,
            )

            active_counts += (sae_latents.mean(dim=1) > activation_threshold).sum(dim=0).cpu().numpy()

            avg_sae_latents = sae_latents.mean(dim=1)
            class_activations[start_idx:end_idx] = avg_sae_latents.cpu().numpy()
            self._update_storage(storage, avg_sae_latents, batch_indices)

        if save_features:
            activation_writer.write_cls_activation(
                class_activations,
                cls_idx=cls_idx,
            )
        return class_activations, active_counts

    def save_sae_stat_storage(
        self,
        save_path,
        storage: Dict[str, torch.Tensor],
    ) -> None:
        """Save results to disk using HDF5 (h5py)."""

        with h5py.File(save_path, "a") as h5f:
            h5f.create_dataset(
                "max_activating_image_indices",
                data=storage["max_activating_image_indices"].numpy(),
                compression="gzip",
            )
            h5f.create_dataset(
                "max_activating_image_values",
                data=storage["max_activating_image_values"].cpu().numpy(),
                compression="gzip",
            )
            h5f.create_dataset("sae_sparsity", data=storage["sae_sparsity"].cpu().numpy(), compression="gzip")
            h5f.create_dataset("sae_mean_acts", data=storage["sae_mean_acts"].cpu().numpy(), compression="gzip")

            max_activating_image_label_indices = self.class_labels[
                storage["max_activating_image_indices"].numpy().astype(int)
            ]

            h5f.create_dataset(
                "max_activating_image_label_indices", data=max_activating_image_label_indices, compression="gzip"
            )

    def process_reference_dataset(self, h5_path, batch_size=64, save_features=True):

        activation_writer = H5ActivationtWriter(save_path=h5_path, feature_dim=self.cfg.d_sae)
        class_mean_var_activations = np.zeros((2, self.num_classes, self.cfg.d_sae), dtype=np.float32)
        storage = self.initialize_storage_tensors()
        activation_counts = np.zeros(self.cfg.d_sae, dtype=np.int32)
        for i in tqdm(range(self.num_classes), desc=f"Processing classes in {self.dataset_name}"):

            class_indices = np.where(self.class_labels == i)[0]
            class_subset = self.dataset.select(class_indices)
            class_activations, cls_active_counts = self.process_class(
                storage,
                class_subset,
                class_indices,
                activation_writer,
                cls_idx=i,
                batch_size=batch_size,
                save_features=save_features,
            )
            activation_counts += cls_active_counts
            class_mean_var_activations[0][i] = class_activations.mean(axis=0)
            class_mean_var_activations[1][i] = class_activations.var(axis=0)

        storage["sae_mean_acts"] /= storage["sae_sparsity"]
        storage["sae_sparsity"] /= len(self.class_labels)
        self.save_sae_stat_storage(
            save_path=h5_path,
            storage=storage,
        )

        if save_features:
            with h5py.File(h5_path, "a") as hf:
                hf.create_dataset(
                    "sae_mean_var_activations",
                    data=class_mean_var_activations,
                    compression="gzip",
                )

        return activation_counts

    def check_validity(self, h5_path):
        """Check if the H5 file exists and contains the necessary datasets."""
        if not os.path.exists(h5_path):
            return False

        with h5py.File(h5_path, "r") as hf:
            if "max_activating_image_indices" in hf:
                return True
            else:
                return False
        return True

    def run(
        self,
        batch_size: int = 64,
        top_k: int = 5,
        resize_size: int = 256,
        use_gpt: bool = False,
        save_features=True,
    ):

        is_valid = self.check_validity(self.h5_path)
        if not is_valid:
            activation_counts = self.process_reference_dataset(
                self.h5_path, batch_size=batch_size, save_features=save_features
            )
        else:
            activation_counts = None

        max_act_imgs, mean_acts = self.get_sae_stats(self.h5_path)
        valid_latents = self.get_valid_latents(
            activation_counts,
            mean_acts=mean_acts,
            activation_threshold=0.5,
            num_activation_threshold=1,
            save_features=save_features,
        )

        for latent_idx in tqdm(valid_latents, desc="Processing valid latents"):
            self.get_reference_images_and_save(
                max_act_imgs=max_act_imgs,
                latent_idx=latent_idx,
                top_k=top_k,
                resize_size=resize_size,
            )
            self.naming_latent(latent_idx, use_gpt=use_gpt)

    def get_valid_latents(
        self,
        activation_counts,
        mean_acts,
        activation_threshold=0.5,
        num_activation_threshold=1,
        save_features=True,
    ):

        valid_latent_dir = f"{self.checkpoint_path}/valid_latents.json"
        if os.path.exists(valid_latent_dir):
            with open(valid_latent_dir, "r") as f:
                valid_indices = np.array(json.load(f))
            return valid_indices

        if save_features:
            active_data_count = np.zeros(mean_acts.shape[0])
            for cls_idx in tqdm(range(self.num_classes), desc="Processing classes"):
                with h5py.File(self.h5_path, "r") as hf:
                    cls_activations = hf[f"activations_{cls_idx}"][:]
                cls_activations = self.filter_out_nosiy_activation(cls_activations, mean_acts)
                cls_activations = np.where(cls_activations > activation_threshold, 1, 0).sum(0)
                active_data_count += cls_activations
            valid_indices = np.where(active_data_count > num_activation_threshold)[0]
        else:
            valid_indices = np.where(activation_counts > num_activation_threshold)[0]

        with open(valid_latent_dir, "w") as f:
            json.dump(valid_indices.tolist(), f)
        return valid_indices

    def filter_out_nosiy_activation(self, sae_activations, mean_acts):

        noisy_features_indices = (mean_acts > 0.1).nonzero()[0].tolist()
        sae_activations[:, noisy_features_indices] = 0
        return sae_activations

    def get_sae_stats(self, save_dir):

        with h5py.File(save_dir, "r") as hf:
            max_act_imgs = hf["max_activating_image_indices"][:].astype(int)
            mean_acts = hf["sae_mean_acts"][:]

        return max_act_imgs, mean_acts

    def get_reference_images_and_save(self, max_act_imgs, latent_idx, top_k=5, resize_size=256):
        save_dir = f"{self.checkpoint_path}/reference_images"
        os.makedirs(save_dir, exist_ok=True)
        save_path = f"{save_dir}/{latent_idx}.png"
        if os.path.exists(save_path):
            return Image.open(save_path)

        img_indices = max_act_imgs[latent_idx]
        images = []
        for i in img_indices[:top_k]:
            images.append(self.dataset[i.item()][self.image_key])

        sae_masks = get_sae_mask(
            images, self.sae, self.vit, self.cfg, latent_idx, resize_size=resize_size, device=self.device
        )
        masked_images, _ = apply_sae_mask_to_input(
            images,
            sae_masks,
            resize_size=resize_size,
            blend_rate=0.0,
            gamma=0.001,
            reverse=False,
        )

        fig = plot_images(
            masked_images,
            num_cols=5,
            show_plot=False,
            resize_size=resize_size,
        )

        fig.savefig(
            save_path,
            bbox_inches="tight",
        )

    def naming_latent(self, latent_idx, use_gpt=True):
        if str(latent_idx) in self.concept_dict:
            return

        if use_gpt:
            image_path = f"{self.save_root}/reference_images/{latent_idx}.png"
            try:
                caption = self.generate_image_caption_gpt(image_path)
            except Exception as e:
                print(f"Error generating caption for latent {latent_idx}: {e}")
                caption = "Unknown concept"
        else:
            caption = "not inferred"

        self.concept_dict[str(latent_idx)] = caption
        with open(self.concept_dict_path, "w") as f:
            json.dump(self.concept_dict, f, indent=4)

    def generate_image_caption_gpt(self, image_path, model_name="gpt-4o"):
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode("utf-8")

        response = openai.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Identify the shared concept among images, focusing solely on non-blocked areas and excluding dark silhouette areas, using 2-3 clear and specific words. Answer with concepts only.",
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                        },
                    ],
                }
            ],
            max_tokens=300,
        )

        return response.choices[0].message.content
