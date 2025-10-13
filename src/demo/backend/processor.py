import base64
import json
import os
from io import BytesIO

import h5py
import hdbscan
import numpy as np
import pandas as pd
import torch
import umap
from PIL import Image
from sklearn.decomposition import PCA
from statsmodels.stats.proportion import proportion_confint
from tqdm import tqdm

from src.conceptscope.conceptscope import ConceptScope
from src.utils.image_dataset_loader import ImageDatasetLoader
from src.utils.processor import ImageProcessor
from src.utils.utils import load_class_names, make_json_serializable


def crop_from_top_left(image: Image.Image) -> Image.Image:
    width, height = image.size

    # Crop to H x H starting from top-left
    crop_width = min(width, height)
    return image.crop((0, 0, crop_width, crop_width))  # (left, top, right, bottom)


class Processor:
    def __init__(
        self,
        root,
        dir_name="dataset_analysis",
        clip_model_name="openai/clip-vit-large-patch14",
        backbone="openai/clip-vit-large-patch14",
        checkpoint_name="openai_l14_32K_base",
        num_samples_for_alignment=128,
        device="cuda",
    ):

        self.save_root = f"{root}/out"
        self.data_root = f"{root}/data"
        self.dir_name = dir_name
        self.checkpoint_name = checkpoint_name
        sae_path = f"{self.save_root}/checkpoints/{self.checkpoint_name}/clip-vit-large-patch14_-2_resid_32768.pt"
        self.device = device
        print("initializing ConceptScope with device:", self.device)
        self.conceptscope = ConceptScope(
            save_root=self.save_root,
            dir_name=self.dir_name,
            backbone=backbone,
            clip_model_name=clip_model_name,
            sae_path=sae_path,
            num_samples_for_alignment=num_samples_for_alignment,
            device=self.device,
            data_root=self.data_root,
        )
        print(f"ConceptScope initialized with device: {self.device}")
        self._train_dataset = None
        self._test_dataset = None
        self._train_labels = None
        self._test_labels = None
        self._dataset_name = None
        self._train_split = "train"
        self._test_split = "val"
        self._compare_dataset_name = None
        self._pred_results = None
        self._model_name = None
        self._concept_categorization_dict = None
        self._class_names = None
        self.image_root = f"{self.save_root}/checkpoints/{self.checkpoint_name}/reference_images"
        self._refernce_dataset = ImageDatasetLoader.load_dataset(
            dataset_name="imagenet", split="train", root=self.data_root
        )

    @property
    def concept_categorization_dict(self):
        return self._concept_categorization_dict

    @property
    def train_split(self):
        return self._train_split

    @property
    def test_split(self):
        return self._test_split

    @property
    def dataset_name(self):
        return self._dataset_name

    @property
    def class_names(self):
        return self._class_names

    @property
    def reference_dataset(self):
        return self._refernce_dataset

    @property
    def train_labels(self):
        return self._train_labels

    @property
    def test_labels(self):
        return self._test_labels

    @property
    def pred_results(self):
        return self._pred_results

    @property
    def model_name(self):
        return self._model_name

    def get_splits(self, dataset_name):
        if dataset_name == "imagenet":
            return ["train", "val"]
        else:
            return ["train", "test"]

    @dataset_name.setter
    def dataset_name(self, dataset_name, train_split="train"):
        dataset_name = dataset_name.lower()
        self._train_split, self._test_split = self.get_splits(dataset_name)

        self._dataset_name = dataset_name
        if dataset_name == "imagenet":
            self._train_dataset = self._refernce_dataset
        else:
            self._train_dataset = ImageDatasetLoader.load_dataset(
                dataset_name=dataset_name, split=train_split, root=self.data_root
            )
        self._test_dataset = ImageDatasetLoader.load_dataset(
            dataset_name=dataset_name, split=self._test_split, root=self.data_root
        )
        self._concept_categorization_dict = self.get_concept_categorization_dict(
            dataset_name=dataset_name,
            train_split=train_split,
        )
        self._class_names = load_class_names(self.data_root, dataset_name, self._train_dataset)
        self._concept_matrix = self._get_concept_matrix(
            dataset_name=dataset_name,
            split=train_split,
        )

        self._train_labels = np.array(self._train_dataset["label"])
        self._test_labels = np.array(self._test_dataset["label"])

    @model_name.setter
    def model_name(self, model_name):
        self._model_name = model_name
        self._pred_results = self._load_pred_results(dataset_name=self.dataset_name, model_name=model_name)

    def get_concept_info(self, latent_idx, resize_size=256, top_k_images=20):

        save_path = f"{self.save_root}/{self.dir_name}/imagenet/{self.checkpoint_name}/train_sae_latents.h5"
        with h5py.File(save_path, "r") as hf:
            max_image_indices = hf["max_activating_image_indices"][:].astype(int)
        max_image_indices = max_image_indices[latent_idx][:top_k_images]

        high_activating_images = self.get_image_from_index(
            indices=max_image_indices,
            dataset=self.reference_dataset,
            resize_size=resize_size,
        )

        high_images_mask = self.get_sae_mask(
            images=high_activating_images,
            idx=latent_idx,
            resize_size=resize_size,
        )
        masked_high_images = self.apply_sae_mask_to_input(
            images=high_activating_images,
            masks=high_images_mask,
            resize_size=resize_size,
        )

        latent_avg_activations = self.concept_matrix[0][:, latent_idx]

        return {
            "high_activating_images": self.encode_images(high_activating_images),
            "masked_high_images": self.encode_images(masked_high_images),
            "latent_avg_activations": latent_avg_activations.tolist(),
        }

    def get_concept_categorization_dict(
        self,
        dataset_name,
        train_split="train",
    ):
        save_path = f"{self.save_root}/{self.dir_name}/{dataset_name}/{self.checkpoint_name}/{train_split}_concept_categorization.json"
        with open(save_path, "r") as f:
            concept_dict = json.load(f)
        return concept_dict

    @property
    def train_dataset(self):
        return self._train_dataset

    @train_dataset.setter
    def train_dataset(self, dataset_name, split):
        self.train_dataset = ImageDatasetLoader.load_dataset(
            dataset_name=dataset_name, split=split, root=self.data_root
        )

    @property
    def test_dataset(self):
        return self._test_dataset

    @test_dataset.setter
    def test_dataset(self, dataset_name, split):
        self.test_dataset = ImageDatasetLoader.load_dataset(dataset_name=dataset_name, split=split, root=self.data_root)

    def _get_concept_matrix(self, dataset_name, split):
        concept_matrix = self.conceptscope.get_train_mean_var_activations(dataset_name=dataset_name, split=split)
        return concept_matrix

    @property
    def concept_matrix(self):
        if self._concept_matrix is None:
            self._concept_matrix = self._get_concept_matrix(self.dataset_name, self.train_split)
        return self._concept_matrix

    def _filter_concept_matrix(self, concept_type):
        concept_matrix = self.concept_matrix[0].copy()
        if concept_type != "all":
            for selected_class in range(len(self.class_names)):

                latent_indices = []
                if concept_type == "context":
                    for slice_info in self.concept_categorization_dict[str(selected_class)]["context"]:
                        latent_indices.append(slice_info["latent_idx"])
                elif concept_type == "target":
                    for slice_info in self.concept_categorization_dict[str(selected_class)]["target"]:
                        latent_indices.append(slice_info["latent_idx"])

                latent_indices = np.array(latent_indices).astype(int)

                masks = np.zeros(concept_matrix.shape[1], dtype=bool)
                masks[latent_indices] = True

                concept_matrix[selected_class, ~masks] = 0.0
        return concept_matrix[:, self.conceptscope.valid_latents]

    def get_class_indices_and_activations(self, class_idx, split, latent_idx=None):
        if split == "train":
            all_label = self.train_labels
        else:
            all_label = self.test_labels

        class_indices = np.where(all_label == class_idx)[0]
        activations = self.get_class_sae_activation(split=split, class_idx=class_idx)

        if latent_idx is not None:
            activations = activations[:, latent_idx]

        return class_indices, activations

    def get_latent_activations(self, selected_class, latent_idx):
        _, train_activations = self.get_class_indices_and_activations(
            class_idx=selected_class, split=self.train_split, latent_idx=latent_idx
        )
        _, test_activations = self.get_class_indices_and_activations(
            class_idx=selected_class, split=self.test_split, latent_idx=latent_idx
        )
        all_activations = np.concatenate([train_activations, test_activations], axis=0)
        return all_activations

    def get_class_images(self, selected_class):
        train_indices = np.where(self.train_labels == selected_class)[0]
        test_indices = np.where(self.test_labels == selected_class)[0]

        train_images = self.get_image_from_index(
            indices=train_indices,
            dataset=self.train_dataset,
        )
        test_images = self.get_image_from_index(
            indices=test_indices,
            dataset=self.test_dataset,
        )
        return self.encode_images(train_images + test_images)

    def get_samples_by_indices(self, indices, split, with_mask=False, latent_idx=None, resize_size=256, top_k=10):
        if split == "train":
            dataset = self.train_dataset
        else:
            dataset = self.test_dataset

        if len(indices) == 0:
            if with_mask:
                return {"images": [], "masked_images": []}
            else:
                return {"images": []}

        images = self.get_image_from_index(
            indices=indices,
            dataset=dataset,
        )[:top_k]
        out = {"images": self.encode_images(images)}
        if with_mask:
            highest_image_mask = self.get_sae_mask(images, latent_idx, resize_size)
            masked_highest_images = self.apply_sae_mask_to_input(images, highest_image_mask, reverse=False)
            out["masked_images"] = self.encode_images(masked_highest_images)

        return out

    def get_info_for_selected_samples(self, selected_class, train_selected_indices, test_selected_indices, top_k=3):
        train_indices, train_activations = self.get_class_indices_and_activations(
            class_idx=selected_class, split=self.train_split
        )
        test_indices, test_activations = self.get_class_indices_and_activations(
            class_idx=selected_class, split=self.test_split
        )

        def get_local_indices(global_indices, reference_indices):
            return np.array([np.where(reference_indices == idx)[0][0] for idx in global_indices])

        train_local = get_local_indices(train_selected_indices, train_indices)
        test_local = get_local_indices(test_selected_indices, test_indices)

        if len(train_local) == 0:
            selected_activations = test_activations[test_local]
        elif len(test_local) == 0:
            selected_activations = train_activations[train_local]
        else:
            selected_activations = np.concatenate(
                [train_activations[train_local], test_activations[test_local]], axis=0
            )

        avg_activations = np.mean(selected_activations[:, self.conceptscope.valid_latents], axis=0)
        top_indices = np.argsort(-avg_activations)[:top_k]
        top_activations = avg_activations[top_indices]

        latent_names = []
        image_dict = {}
        for latent_idx in top_indices:
            latent_idx = self.conceptscope.valid_latents[latent_idx]
            latent_name = self.conceptscope.concept_dict[str(latent_idx)]
            latent_names.append(latent_name)
            image_dict[str(latent_idx)] = self.get_concept_info(latent_idx=latent_idx)

        concept_dict = {
            "activations": top_activations.tolist(),
            "latent_names": latent_names,
            "latent_idx": self.conceptscope.valid_latents[top_indices].tolist(),
            "image_dict": image_dict,
        }

        if len(train_local) == 0:
            train_image_dict = {"images": []}
        else:
            train_image_dict = self.get_samples_by_indices(
                indices=train_indices[train_local], split=self.train_split, with_mask=False
            )
        if len(test_local) == 0:
            test_image_dict = {"images": []}
        else:
            test_image_dict = self.get_samples_by_indices(
                indices=test_indices[test_local], split=self.test_split, with_mask=False
            )

        return {
            "concept_dict": concept_dict,
            "train_images": train_image_dict,
            "test_images": test_image_dict,
        }

    def get_activation_from_indices(self, indices, class_idx, split="test", latent_idx=None):
        if split == "train":
            split = self.train_split
        else:
            split = self.test_split
        class_indices_all, class_activations = self.get_class_indices_and_activations(
            class_idx=class_idx, split=split, latent_idx=latent_idx
        )
        local_class_indices = np.where(np.isin(class_indices_all, indices))[0]
        local_activations = class_activations[local_class_indices,]
        return local_activations

    def get_concept_distribution(self, selected_class, bias_sigma=1, max_concepts=20):
        data = {
            "Class aligned": [],
            "Mean": [],
            "Concept Type": [],
            "latent_idx": [],
            "latent_name": [],
            "slice_idx": [],
            "thumbnail_image": [],
            "full_image": [],
            "frequency": [],
        }

        context_means = []
        len_target = len(self.concept_categorization_dict[str(selected_class)]["target"])
        len_context = max_concepts - len_target
        all_slices = (
            self.concept_categorization_dict[str(selected_class)]["target"]
            + self.concept_categorization_dict[str(selected_class)]["context"][:len_context]
        )

        class_activations = self.get_class_sae_activation(split="train", class_idx=selected_class)
        for i, slice_info in enumerate(all_slices):
            data["latent_idx"].append(slice_info["latent_idx"])
            data["latent_name"].append(slice_info["latent_name"])
            data["Mean"].append(slice_info["mean_activation"])
            data["Class aligned"].append(slice_info["alignment_score"])
            data["slice_idx"].append(i)
            if "bias" not in slice_info:
                data["Concept Type"].append("target")
            else:
                context_means.append(slice_info["mean_activation"])
                if slice_info["bias"] == True:
                    data["Concept Type"].append("bias")
                else:
                    data["Concept Type"].append("context")

            image = Image.open(f"{self.image_root}/{slice_info['latent_idx']}.png")
            thumbnail_image = self.encode_image(crop_from_top_left(image))
            full_image = self.encode_image(image)
            data["thumbnail_image"].append(thumbnail_image)
            data["full_image"].append(full_image)

            latent_activations = class_activations[:, int(slice_info["latent_idx"])]
            data["frequency"].append(np.where(latent_activations > 0.5, 1, 0).mean())

        context_means = np.array(context_means)
        bias_threshold = np.mean(context_means) + bias_sigma * np.std(context_means)

        alignment_scores = np.array(data["Class aligned"])
        target_threshold = alignment_scores.mean() + 1.0 * alignment_scores.std()

        df = pd.DataFrame(data)
        color_map = {"target": "green", "context": "orange", "bias": "red"}
        df["Color"] = df["Concept Type"].map(color_map)
        df["bias_threshold"] = bias_threshold
        df["target_threshold"] = target_threshold

        df = df.sort_values("Class aligned", ascending=True).reset_index(drop=True)
        return df

    def get_class_sae_activation(self, split, class_idx, dataset_name=None):
        if dataset_name is None:
            dataset_name = self.dataset_name
        save_path = f"{self.save_root}/{self.dir_name}/{dataset_name}/{self.checkpoint_name}/{split}_sae_latents.h5"
        with h5py.File(save_path, "r") as hf:
            if f"activations_{class_idx}" in hf:
                activations = hf[f"activations_{class_idx}"][:]
            else:
                raise ValueError(f"No activations found for class index {class_idx} in {save_path}")
        return activations

    def get_sae_mask(self, images, idx, resize_size=256):
        inputs = ImageProcessor.process_model_inputs({"image": images}, self.conceptscope.vit, self.device)
        sae_act = ImageProcessor.get_sae_activations(
            self.conceptscope.sae,
            self.conceptscope.vit,
            inputs,
            self.conceptscope.cfg.block_layer,
            self.conceptscope.cfg.module_name,
            self.conceptscope.cfg.class_token,
            get_mean=False,
        )
        selected_act = sae_act[:, :, idx]
        feature_size = int(np.sqrt(selected_act.shape[1] - 1))

        masks = torch.Tensor(selected_act[:, 1:].reshape(sae_act.shape[0], 1, feature_size, feature_size))
        masks = (
            torch.nn.functional.interpolate(masks, (resize_size, resize_size), mode="bilinear").squeeze(1).cpu().numpy()
        )
        return masks

    def apply_sae_mask_to_input(self, images, masks, resize_size=256, blend_rate=0.0, gamma=0.001, reverse=False):
        masked_images = []
        for i, image in enumerate(images):
            if isinstance(image, str):
                image = Image.open(image).convert("RGB")
            image_array = np.array(image.resize((resize_size, resize_size)))[..., :3].astype(np.float32)

            mask = (masks[i] - masks[i].min()) / (masks[i].max() - masks[i].min() + 1e-10)
            mask = np.expand_dims(mask, axis=-1)  # Make shape (H, W, 1) to broadcast over RGB

            mask = mask**gamma

            if reverse:
                mask = 1 - mask

            blended = image_array * (blend_rate + (1 - blend_rate) * mask)
            blended = np.clip(blended, 0, 255).astype(np.uint8)

            masked_images.append(Image.fromarray(blended))
        return masked_images

    @staticmethod
    def get_image_from_index(indices, dataset, resize_size=256):
        images = []
        subset = dataset[indices]["image"]
        for image in subset:
            if isinstance(image, str):
                image = Image.open(image)
            images.append(image.resize((resize_size, resize_size)))
        return images

    def get_images_from_class(
        self,
        class_idx,
        latent_idx,
        split="train",
        resize_size=256,
        top_k=10,
        threshold=None,
        dataset_name=None,
    ):
        class_activations = self.get_class_sae_activation(split=split, class_idx=class_idx)
        latent_activations = class_activations[:, latent_idx]

        if split == "train":
            all_label = self.train_labels
            dataset = self.train_dataset
        else:
            all_label = self.test_labels
            dataset = self.test_dataset
        class_indices = np.where(all_label == class_idx)[0]

        if threshold is None:
            sorted_indices = np.argsort(latent_activations)[::-1]
            highest_indices = sorted_indices[:top_k]
            lowest_indices = sorted_indices[-top_k:]
        else:
            all_high_indices = np.where(latent_activations >= threshold)[0]
            all_low_indices = np.where(latent_activations < threshold)[0]
            all_high_indices = all_high_indices[np.argsort(latent_activations[all_high_indices])[::-1]]
            all_low_indices = all_low_indices[np.argsort(latent_activations[all_low_indices])[::-1]]

            sorted_indices = np.argsort(latent_activations)[::-1]
            highest_indices = sorted_indices[:top_k]
            lowest_indices = sorted_indices[-top_k:]

        # if threshold is None:
        #     sorted_indices = np.argsort(latent_activations)[::-1]
        #     highest_indices = sorted_indices[:top_k]
        #     lowest_indices = sorted_indices[-top_k:]
        # else:

        #     highest_indices = all_high_indices[:top_k]
        #     lowest_indices = all_low_indices[:top_k]

        def _get_image_from_index(class_indices, indices, dataset, resize_size):
            images = []
            subset = dataset[class_indices]["image"]
            for idx in indices:
                image = subset[idx]
                if isinstance(image, str):
                    image = Image.open(image)
                images.append(image.resize((resize_size, resize_size)))
            return images

        highest_images = _get_image_from_index(class_indices, highest_indices, dataset, resize_size)
        lowest_images = _get_image_from_index(class_indices, lowest_indices, dataset, resize_size)

        highest_image_mask = self.get_sae_mask(highest_images, latent_idx, resize_size)
        masked_highest_images = self.apply_sae_mask_to_input(highest_images, highest_image_mask, reverse=False)
        out = {
            "highest_images": self.encode_images(highest_images),
            "lowest_images": self.encode_images(lowest_images),
            "masked_highest_images": self.encode_images(masked_highest_images),
            "high_activations": [round(x, 3) for x in latent_activations[highest_indices].tolist()],
            "low_activations": [round(x, 3) for x in latent_activations[lowest_indices].tolist()],
            "high_indices": class_indices[highest_indices].tolist(),
            "low_indices": class_indices[lowest_indices].tolist(),
        }
        if threshold is not None:
            out["all_high_indices"] = class_indices[all_high_indices]
            out["all_low_indices"] = class_indices[all_low_indices]

        return out

    def get_images_with_prediction(
        self, class_idx, latent_idx, resize_size=256, top_k=10, threshold=0.5, dataset_name=None
    ):
        class_dict = self.get_images_from_class(
            class_idx, latent_idx, split=self.test_split, resize_size=resize_size, top_k=top_k, threshold=threshold
        )

        save_path = f"{self.save_root}/classification/results/{self.dataset_name}/resnet50.csv"
        all_prediction = pd.read_csv(save_path)
        preds = all_prediction["pred_label"].to_numpy()
        gts = all_prediction["gt_label"].to_numpy()
        is_correct = preds == gts
        if dataset_name == "imagenet":
            class_dict["high_acc"] = float(is_correct[class_dict["high_indices"]].mean())
            class_dict["low_acc"] = float(is_correct[class_dict["low_indices"]].mean())
        else:
            class_dict["high_acc"] = float(is_correct[class_dict["all_high_indices"]].mean())
            class_dict["low_acc"] = float(is_correct[class_dict["all_low_indices"]].mean())
        class_mean_acc = is_correct[
            np.concatenate([class_dict["all_high_indices"], class_dict["all_low_indices"]])
        ].mean()
        class_dict["mean_acc"] = float(class_mean_acc)
        class_dict["num_high"] = len(class_dict["all_high_indices"])
        class_dict["num_low"] = len(class_dict["all_low_indices"])
        class_dict["high_correct"] = is_correct[class_dict["high_indices"]].tolist()
        class_dict["low_correct"] = is_correct[class_dict["low_indices"]].tolist()
        del class_dict["all_high_indices"]
        del class_dict["all_low_indices"]

        train_activations = self.get_class_sae_activation(split="train", class_idx=class_idx)[:, latent_idx]
        num_high_train = np.where(train_activations >= threshold)[0].shape[0]
        num_high_train_ratio = num_high_train / train_activations.shape[0]
        class_dict["num_high_train"] = num_high_train
        class_dict["num_high_train_ratio"] = num_high_train_ratio

        return class_dict

    @staticmethod
    def encode_images(image_list):
        encoded_images = []
        for img in tqdm(image_list, desc="Encoding images"):
            encoded_image = Processor.encode_image(img)
            encoded_images.append(encoded_image)
        return encoded_images

    @staticmethod
    def encode_image(image):
        buf = BytesIO()
        image.save(buf, format="PNG")
        base64_img = base64.b64encode(buf.getvalue()).decode("utf-8")
        return base64_img
