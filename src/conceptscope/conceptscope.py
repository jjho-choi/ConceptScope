import json
import os
import random

import h5py
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import silhouette_score
from torchvision import transforms
from torchvision.transforms import ToPILImage
from tqdm import tqdm

from src.conceptscope.clip_processor import CLIPPreprocessor
from src.conceptscope.sae_module import SAEModule
from src.utils.image_dataset_loader import ImageDatasetLoader
from src.utils.processor import BatchIterator, H5ActivationtWriter, ImageProcessor
from src.utils.utils import get_len_dataset, load_class_names, make_json_serializable


class ConceptScope(SAEModule):
    def __init__(
        self,
        save_root,
        sae_path,
        device,
        backbone,
        clip_model_name="openai/clip-vit-large-patch14",
        dir_name="dataset_analysis",
        checkpoint_dir_name="checkpoints",
        data_root="./data",
        num_samples_for_alignment=128,
        top_k_for_alignment=20,
    ):
        super().__init__(save_root, sae_path, device, backbone, dir_name, checkpoint_dir_name, data_root)
        self.clip_processor = CLIPPreprocessor(clip_model_name, device)
        self.num_samples_for_alignment = num_samples_for_alignment
        self.top_k_for_alignment = top_k_for_alignment
        self.concept_dict, self.valid_latents = self.load_concept_dict()
        self.frequency_thresholds = [0, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5]

    def load_concept_dict(self):
        save_path = f"{self.checkpoint_path}/concept_dict.json"
        with open(save_path, "r") as f:
            concept_dict = json.load(f)
            valid_latents = np.array(list(concept_dict.keys())).astype(int)
        return concept_dict, valid_latents

    def _get_class_label_embedding(self, label_name):
        if isinstance(label_name, str):
            label_name = label_name.replace("_", " ")

        with torch.no_grad():
            label_embedding = self.clip_processor.get_text_embedding(label_name)
        if isinstance(label_name, str):
            label_embedding = label_embedding.reshape(1, -1)
        return label_embedding

    def get_alignment_score(self, dataset_name, split, dataset, class_names, target_attribute, batch_size=64):
        save_dir = f"{self.save_root}/{self.dir_name}/{dataset_name}/{self.checkpoint_name}"
        if target_attribute is not None:
            save_dir = f"{save_dir}/{split}_alignment_scores_{target_attribute}.json"
        else:
            save_dir = f"{save_dir}/{split}_alignment_scores.json"
        os.makedirs(os.path.dirname(save_dir), exist_ok=True)
        if os.path.exists(save_dir):
            with open(save_dir, "r") as f:
                out_dict = json.load(f)
            if len(out_dict) == len(class_names):
                return out_dict
        else:
            out_dict = {}

        latent_save_dir = (
            f"{self.save_root}/{self.dir_name}/{dataset_name}/{self.checkpoint_name}/{split}_sae_latents.h5"
        )
        os.makedirs(os.path.dirname(latent_save_dir), exist_ok=True)

        activation_writer = H5ActivationtWriter(save_path=latent_save_dir, feature_dim=self.cfg.d_sae)

        if dataset_name == "ms_coco":
            class_labels = []
        else:
            class_labels = np.array(dataset[self.label_key])
        num_classes = len(class_names)
        class_mean_var_activations = np.zeros((2, num_classes, self.cfg.d_sae), dtype=np.float32)
        frequency_dict = {threshold: np.zeros((num_classes, self.cfg.d_sae)) for threshold in self.frequency_thresholds}

        for i in tqdm(range(num_classes), desc=f"Processing classes in {dataset_name}"):
            if str(i) in out_dict:
                continue
            out_dict[str(i)] = {}

            if dataset_name == "ms_coco":
                class_indices = np.where(np.array(dataset[class_names[i]]))[0]
            else:
                class_indices = np.where(class_labels == i)[0]
            class_subset = dataset.select(class_indices)

            if activation_writer.check_dataset_exists(i):
                class_avg_activations, frequency_dict_cls, sampled_latent_activations, sampled_images = (
                    self.get_sampled_image_with_activations(
                        latent_save_dir, i, dataset, class_indices, batch_size=batch_size
                    )
                )
            else:
                class_avg_activations, frequency_dict_cls, sampled_latent_activations, sampled_images = (
                    self.process_class(
                        class_subset,
                        class_indices,
                        activation_writer,
                        class_mean_var_activations,
                        cls_idx=i,
                        batch_size=batch_size,
                    )
                )

            for threshold in self.frequency_thresholds:
                frequency_dict[threshold][i] = frequency_dict_cls[threshold]

            score = self.compute_alignment_score_per_class(
                sampled_latent_activations,
                sampled_images,
                class_avg_activations,
                class_names[i],
                frequency_dict_cls,
                batch_size=batch_size,
            )
            out_dict[str(i)] = score

            out_dict = make_json_serializable(out_dict)
            with open(save_dir, "w") as f:
                json.dump(out_dict, f, indent=4)

        with h5py.File(latent_save_dir, "a") as hf:
            if "sae_mean_var_activations" not in hf:
                hf.create_dataset(
                    "sae_mean_var_activations",
                    data=class_mean_var_activations,
                    compression="gzip",
                )

            if "frequency_0" not in hf:
                for threshold in self.frequency_thresholds:
                    hf.create_dataset(
                        f"frequency_{threshold}",
                        data=frequency_dict[threshold],
                        compression="gzip",
                    )
        return out_dict

    def get_sampled_image_with_activations(self, save_path, cls_idx, dataset, class_indices, batch_size=64):
        with h5py.File(save_path, "r") as hf:
            mean_activations = hf["sae_mean_var_activations"][:][0][cls_idx]
            if "frequency_0" not in hf:
                frequency_dict = {}
                class_activations = hf[f"activations_{cls_idx}"][:]
                for threshold in self.frequency_thresholds:
                    frequency_dict[threshold] = np.where(class_activations > threshold, 1, 0).mean(axis=0)
            else:
                frequency_dict = {
                    threshold: hf[f"frequency_{threshold}"][:][cls_idx] for threshold in self.frequency_thresholds
                }

        top_k = min(self.num_samples_for_alignment, len(class_indices))
        sampled_indices = set(random.sample(class_indices.tolist(), k=top_k))
        subset = dataset.select(sampled_indices)

        batch_iterator = BatchIterator.get_batches(subset, batch_size)
        sampled_activations = []
        sampled_images = []
        for _, _, batch in batch_iterator:
            batch_inputs = ImageProcessor.process_model_inputs(batch, self.vit, self.device)
            sae_latents = ImageProcessor.get_sae_activations(
                self.sae,
                self.vit,
                batch_inputs,
                self.cfg.block_layer,
                self.cfg.module_name,
                self.cfg.class_token,
                get_mean=False,
            )

            for sae_latent, image in zip(sae_latents, batch[self.image_key]):
                sampled_activations.append(sae_latent.cpu())
                if isinstance(image, str):
                    image = Image.open(image).convert("RGB")
                sampled_images.append(image)

        return mean_activations, frequency_dict, sampled_activations, sampled_images

    def run(
        self,
        dataset_name,
        split,
        target_attribute=None,
        batch_size=64,
        target_threshold=0.0,
        verbose=True,
    ):
        if verbose:
            print(f"Running ConceptScope for dataset: {dataset_name}, split: {split}")
            print(f"Using SAE model checkpoint: {self.checkpoint_name}")
            print("Load dataset and extract SAE activations")

        dataset = ImageDatasetLoader.load_dataset(dataset_name=dataset_name, split=split, root=self.data_root)
        class_names = load_class_names(self.data_root, dataset_name, dataset)
        alignment_scores_dict = self.get_alignment_score(
            dataset_name, split, dataset, class_names, target_attribute, batch_size=batch_size
        )

        if verbose:
            print("Categorizing concepts based on alignment scores")
        concept_categorized_dict = self.concept_cateogorization(
            dataset_name,
            split,
            class_names,
            alignment_scores_dict,
            target_attribute=target_attribute,
            target_threshold=target_threshold,
        )

        return concept_categorized_dict

    def get_concept_dict(self, dataset_name, split, target_attribute=None):
        save_path = f"{self.save_root}/{self.dir_name}/{dataset_name}/{self.checkpoint_name}"
        if target_attribute is not None:
            save_path = f"{save_path}/{split}_concept_categorization_{target_attribute}.json"
        else:
            save_path = f"{save_path}/{split}_concept_categorization.json"

        if os.path.exists(save_path):
            with open(save_path, "r") as f:
                concept_dict = json.load(f)
            return concept_dict

        return concept_dict

    def concept_cateogorization(
        self,
        dataset_name,
        split,
        class_names,
        alignment_scores_dict,
        bias_threshold_sigma=1.0,
        target_threshold=0.0,
        target_attribute=None,
    ):
        out_dict = {}
        for cls_idx in tqdm(range(len(class_names)), desc="Categorizing concepts"):
            out_dict[str(cls_idx)] = {"target": [], "context": []}
            alignment_scores = np.array(
                [score_dict["alignment_score"] for score_dict in alignment_scores_dict[str(cls_idx)].values()]
            )

            normalized_scores = (alignment_scores - alignment_scores.mean()) / alignment_scores.std()

            silhouette_scores = np.zeros(len(alignment_scores))
            for i, alignment_score in enumerate(alignment_scores):
                label = (alignment_scores >= alignment_score).astype(int)
                if np.unique(label).shape[0] == 1:
                    silhouette_scores[i] = 0.0
                    continue
                silhouette_scores[i] = silhouette_score(alignment_scores.reshape(-1, 1), label)
            # target_threshold = alignment_scores[silhouette_scores.argmax()]
            max_silhouette_score = silhouette_scores.max()

            context_act_means = []
            for i, (latent_idx, score_dict) in enumerate(alignment_scores_dict[str(cls_idx)].items()):
                common_dict = {
                    "latent_idx": int(latent_idx),
                    "latent_name": self.concept_dict[str(latent_idx)],
                    "alignment_score": score_dict["alignment_score"],
                    "mean_activation": score_dict["mean_activation"],
                    "normalized_alignment_score": float(normalized_scores[i]),
                    "target_threshold": float(target_threshold),
                    "max_silhouette_score": float(max_silhouette_score),
                }
                for threshold in self.frequency_thresholds:
                    common_dict[f"frequency_{threshold}"] = score_dict[f"frequency_{threshold}"]

                if normalized_scores[i] >= target_threshold:
                    out_dict[str(cls_idx)]["target"].append(common_dict)
                else:
                    out_dict[str(cls_idx)]["context"].append(common_dict)
                    context_act_means.append(score_dict["mean_activation"])
            context_act_means = np.array(context_act_means)
            bias_threshold = context_act_means.mean() + bias_threshold_sigma * context_act_means.std()

            for i, latent_info in enumerate(out_dict[str(cls_idx)]["context"]):
                if latent_info["mean_activation"] >= bias_threshold:
                    out_dict[str(cls_idx)]["context"][i]["bias"] = True
                else:
                    out_dict[str(cls_idx)]["context"][i]["bias"] = False

        save_path = f"{self.save_root}/{self.dir_name}/{dataset_name}/{self.checkpoint_name}"
        if target_attribute is not None:
            save_path = f"{save_path}/{split}_concept_categorization_{target_attribute}.json"
        else:
            save_path = f"{save_path}/{split}_concept_categorization.json"

        with open(save_path, "w") as f:
            json.dump(out_dict, f, indent=4)

        return out_dict

    def apply_sae_mask_to_input_tensor(self, images, masks, blend_rate=0.0, gamma=0.001, reverse=False):

        N = images.size(0)

        min_val = masks.view(N, -1).min(dim=1, keepdim=True)[0].view(N, 1, 1, 1)
        max_val = masks.view(N, -1).max(dim=1, keepdim=True)[0].view(N, 1, 1, 1)
        norm_masks = (masks - min_val) / (max_val - min_val + 1e-10)

        norm_masks = norm_masks**gamma

        if reverse:
            norm_masks = 1.0 - norm_masks

        blended = images * (blend_rate + (1 - blend_rate) * norm_masks)
        blended = torch.clamp(blended, 0.0, 1.0)

        return blended

    def to_pil_image(self, images):
        to_pil = ToPILImage()
        pil_images = [to_pil(img.cpu()) for img in images]
        return pil_images

    def norm_image(self, images):
        means = (
            torch.Tensor(self.clip_processor.clip_processor.image_processor.image_mean).view(1, 3, 1, 1).to(self.device)
        )
        stds = (
            torch.Tensor(self.clip_processor.clip_processor.image_processor.image_std).view(1, 3, 1, 1).to(self.device)
        )
        return (images - means) / stds

    def compute_alignment_score_per_latent(
        self, rand_images, selected_act, label_embedding, resize_size=224, batch_size=64
    ):
        feature_size = int(np.sqrt(selected_act.shape[1] - 1))
        masks = selected_act[:, 1:].reshape(selected_act.shape[0], 1, feature_size, feature_size)
        masks = torch.nn.functional.interpolate(masks, (resize_size, resize_size), mode="bilinear")

        concept_only_image = self.apply_sae_mask_to_input_tensor(rand_images, masks, blend_rate=0, gamma=0.5)
        concept_exclude_image = self.apply_sae_mask_to_input_tensor(
            rand_images, masks, blend_rate=0, gamma=0.001, reverse=True
        )

        rand_images = self.to_pil_image(rand_images)
        concept_only_image = self.to_pil_image(concept_only_image)
        concept_exclude_image = self.to_pil_image(concept_exclude_image)

        # rand_images = self.norm_image(rand_images)
        # concept_only_image = self.norm_image(concept_only_image)
        # concept_exclude_image = self.norm_image(concept_exclude_image)

        concept_only_label_sim = self.clip_processor.get_image_text_cos(
            concept_only_image, label_embedding, batch_size=batch_size
        )
        concept_exclude_image_sim = self.clip_processor.get_image_text_cos(
            concept_exclude_image, label_embedding, batch_size=batch_size
        )
        image_label_sim = self.clip_processor.get_image_text_cos(rand_images, label_embedding, batch_size=batch_size)

        return {
            "orginal_image_label_sim": image_label_sim,
            "concept_only_label_sim": concept_only_label_sim,
            "concept_exclude_image_label_sim": concept_exclude_image_sim,
        }

    def compute_alignment_score_per_class(
        self,
        sampled_activatons,
        sampled_images,
        class_mean_activation,
        class_name,
        frequency_dict,
        batch_size=64,
        resize_size=224,
    ):
        class_label_embedding = self._get_class_label_embedding(class_name)

        sorted_latent_indices = np.argsort(class_mean_activation[self.valid_latents])[::-1]
        top_indices = self.valid_latents[sorted_latent_indices[: self.top_k_for_alignment]]
        num_tokens = sampled_activatons[0].shape[0]
        all_activations = torch.zeros((len(sampled_activatons), num_tokens, len(top_indices)), dtype=torch.float32)
        for i, activation in enumerate(sampled_activatons):
            all_activations[i] = activation[:, top_indices]

        to_tensor = transforms.ToTensor()

        scores = {
            "orginal_image_label_sim": np.zeros((len(sampled_images), self.top_k_for_alignment), dtype=np.float32),
            "concept_only_label_sim": np.zeros((len(sampled_images), self.top_k_for_alignment), dtype=np.float32),
            "concept_exclude_image_label_sim": np.zeros(
                (len(sampled_images), self.top_k_for_alignment), dtype=np.float32
            ),
        }
        for i in range(0, len(sampled_images), batch_size):
            end_idx = min(i + batch_size, len(sampled_images))
            batch_activations = all_activations[i:end_idx,].to(self.device)
            batch_sampled_images = torch.cat(
                [
                    to_tensor(image.resize((resize_size, resize_size))).unsqueeze(0)
                    for image in sampled_images[i:end_idx]
                ],
                dim=0,
            ).to(self.device)

            for j, token_idx in enumerate(tqdm(top_indices)):
                class_scores = self.compute_alignment_score_per_latent(
                    batch_sampled_images,
                    batch_activations[:, :, j],
                    class_label_embedding,
                )

                scores["orginal_image_label_sim"][i:end_idx, j] = class_scores["orginal_image_label_sim"]
                scores["concept_only_label_sim"][i:end_idx, j] = class_scores["concept_only_label_sim"]
                scores["concept_exclude_image_label_sim"][i:end_idx, j] = class_scores[
                    "concept_exclude_image_label_sim"
                ]

        # succiency = 1 - (
        #     abs(scores["orginal_image_label_sim"] - scores["concept_only_label_sim"])
        #     / scores["orginal_image_label_sim"]
        # )
        # succiency = succiency.clip(0, 1)
        # necessary = (
        #     abs(scores["orginal_image_label_sim"] - scores["concept_exclude_image_label_sim"])
        #     / scores["orginal_image_label_sim"]
        # )

        # necessary = necessary.clip(0, 1)

        # sufficient_scores = succiency.mean(0)
        # necessary_scores = necessary.mean(0)
        # alignment_score = (sufficient_scores + necessary_scores) / 2
        # sufficient_std = succiency.std(0)
        # necessary_std = necessary.std(0)
        # alignment_std = (sufficient_std + necessary_std) / 2

        sufficient_scores = (scores["concept_only_label_sim"] / scores["orginal_image_label_sim"]).mean(0)
        necessary_scores = (scores["orginal_image_label_sim"] / scores["concept_exclude_image_label_sim"]).mean(0)
        alignment_score = (sufficient_scores + necessary_scores) / 2

        sufficient_std = (scores["concept_only_label_sim"] / scores["orginal_image_label_sim"]).std(0)
        necessary_std = (scores["orginal_image_label_sim"] / scores["concept_exclude_image_label_sim"]).std(0)
        alignment_std = (sufficient_std + necessary_std) / 2

        class_scores = {}
        for j, token_idx in enumerate(top_indices):
            mean_activation = class_mean_activation[token_idx]
            class_scores[str(token_idx)] = {
                "latent_name": self.concept_dict[str(token_idx)],
                "alignment_score": round(alignment_score[j], 4),
                "mean_activation": round(mean_activation, 4),
                "sufficient_score": round(sufficient_scores[j], 4),
                "necessary_score": round(necessary_scores[j], 4),
                "concept_only_label_sim": round(scores["concept_only_label_sim"][:, j].mean(), 4),
                "concept_exclude_image_label_sim": round(scores["concept_exclude_image_label_sim"][:, j].mean(), 4),
                "orginal_image_label_sim": round(scores["orginal_image_label_sim"][:, j].mean(), 4),
                "sufficient_std": round(sufficient_std[j], 4),
                "necessary_std": round(necessary_std[j], 4),
                "alignment_std": round(alignment_std[j], 4),
                "concept_only_label_sim_std": round(scores["concept_only_label_sim"][:, j].std(), 4),
                "concept_exclude_image_label_sim_std": round(scores["concept_exclude_image_label_sim"][:, j].std(), 4),
                "orginal_image_label_sim_std": round(scores["orginal_image_label_sim"][:, j].std(), 4),
            }

            for threshold in self.frequency_thresholds:
                class_scores[str(token_idx)][f"frequency_{threshold}"] = round(frequency_dict[threshold][token_idx], 4)

        return class_scores

    def process_class(
        self, dataset, class_indices, activation_writer, class_mean_var_activations, cls_idx, batch_size=64
    ):

        batch_iterator = BatchIterator.get_batches(dataset, batch_size)
        num_samples = get_len_dataset(dataset)

        top_k = min(self.top_k_for_alignment, len(class_indices))
        sampled_indices = set(random.sample(class_indices.tolist(), k=top_k))
        sampled_latent_activations = []
        sampled_images = []
        class_activations = np.zeros((num_samples, self.cfg.d_sae), dtype=np.float32)
        frequency_dict = {}

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

            for i, idx in enumerate((batch_indices)):
                if idx in sampled_indices:
                    sampled_latent_activations.append(sae_latents[i].cpu())
                    image = batch[self.image_key][i]
                    if isinstance(image, str):
                        image = Image.open(image).convert("RGB")
                    if image.mode == "L":
                        image = image.convert("RGB")
                    sampled_images.append(image)

            avg_sae_latents = sae_latents.mean(dim=1)
            class_activations[start_idx:end_idx] = avg_sae_latents.cpu().numpy()

        avg_sae_latents = class_activations.mean(axis=0)
        std_sae_latents = class_activations.std(axis=0)

        class_mean_var_activations[0][cls_idx] = avg_sae_latents
        class_mean_var_activations[1][cls_idx] = std_sae_latents
        for threshold in self.frequency_thresholds:
            frequency_dict[threshold] = np.where(class_activations > threshold, 1, 0).mean(axis=0)

        if not activation_writer.check_dataset_exists(cls_idx):
            activation_writer.create_cls_dataset(cls_idx, num_samples)
            activation_writer.write_cls_activation(
                class_activations,
                cls_idx=cls_idx,
            )

        return avg_sae_latents, frequency_dict, sampled_latent_activations, sampled_images

    def extract_sae_latent_activations_and_save(self, file_name, dataset, dataset_name, batch_size=64):
        activation_writer = H5ActivationtWriter(save_path=file_name, feature_dim=self.cfg.d_sae)

        class_labels = np.array(dataset[self.label_key])
        num_classes = len(set(class_labels))
        class_mean_var_activations = np.zeros((2, num_classes, self.cfg.d_sae), dtype=np.float32)

        for i in tqdm(range(num_classes), desc=f"Processing classes in {dataset_name}"):
            class_indices = np.where(class_labels == i)[0]
            class_subset = dataset.select(class_indices)
            self.process_class(
                class_subset,
                class_indices,
                activation_writer,
                class_mean_var_activations,
                cls_idx=i,
                batch_size=batch_size,
            )
        with h5py.File(file_name, "a") as hf:
            hf.create_dataset(
                "sae_mean_var_activations",
                data=class_mean_var_activations,
                compression="gzip",
            )

    def get_sae_latent_activations(self, dataset_name, split, dataset, batch_size=64):
        file_dir = f"{self.save_root}/{self.dir_name}/{dataset_name}/{self.checkpoint_name}"
        os.makedirs(file_dir, exist_ok=True)
        file_name = f"{file_dir}/{split}_sae_latents.h5"

        if dataset_name == "waterbird" or dataset_name == "celeba":
            dataset = ImageDatasetLoader.load_dataset(dataset_name, split=split)

        if not os.path.exists(file_name):
            self.extract_sae_latent_activations_and_save(file_name, dataset, dataset_name, batch_size=batch_size)

        class_labels = np.array(dataset[self.label_key])
        num_classes = len(set(class_labels))
        activation_all = np.zeros((len(class_labels), self.cfg.d_sae), dtype=np.float32)

        for i in tqdm(range(num_classes), desc="Loading SAE activations"):
            class_indices = np.where(class_labels == i)[0]
            with h5py.File(file_name, "r") as hf:
                cls_activation = hf[f"activations_{i}"][:]
            activation_all[class_indices] = cls_activation
        return activation_all

    def get_train_mean_var_activations(self, dataset_name, split):
        file_dir = f"{self.save_root}/{self.dir_name}/{dataset_name}/{self.checkpoint_name}/{split}_sae_latents.h5"

        with h5py.File(file_dir, "r") as hf:
            mean_var_activations = hf["sae_mean_var_activations"][:]
        return mean_var_activations
