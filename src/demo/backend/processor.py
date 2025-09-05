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

from src.conceptscope import ConceptScope
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
        self.load_slice_info()

    def load_slice_info(self, dataset_name="imagenet"):
        slice_info_path = f"{self.save_root}/slice_info/{dataset_name}/resnet50/slice_info.json"
        if os.path.exists(slice_info_path):
            with open(slice_info_path, "r") as f:
                self.slice_info = json.load(f)

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
        self.load_slice_info(dataset_name=dataset_name)

    @model_name.setter
    def model_name(self, model_name):
        self._model_name = model_name
        self._pred_results = self._load_pred_results(dataset_name=self.dataset_name, model_name=model_name)

    def compute_importance_scores(self, i, alpha=1.0):

        high_acc = np.array(self.slice_info[str(i)]["slice_info"]["high_group_accuracy_1.0"])
        low_acc = np.array(self.slice_info[str(i)]["slice_info"]["low_group_accuracy_1.0"])
        acc_diff = np.abs(high_acc - low_acc)
        alignment_scores = np.array(self.slice_info[str(i)]["slice_info"]["alignment_score"])
        tot_samples = len(self.slice_info[str(i)]["slice_info"]["num_high_samples_1.0"]) + len(
            self.slice_info[str(i)]["slice_info"]["num_low_samples_1.0"]
        )
        population = np.array(self.slice_info[str(i)]["slice_info"]["num_high_samples_1.0"]) / tot_samples

        score = acc_diff * population * np.exp(-alpha * alignment_scores)
        # score = (low_acc - high_acc) * population * alignment_scores
        return score

    def _load_pred_results(self, dataset_name, model_name):
        save_path = f"{self.save_root}/classification/results/{dataset_name}/{model_name}.csv"
        all_prediction = pd.read_csv(save_path)

        num_classes = len(self.class_names)
        conf_matrix = np.zeros((num_classes, num_classes), dtype=int)
        preds = all_prediction["pred_label"].to_numpy()
        gts = all_prediction["gt_label"].to_numpy()
        for true_label, pred_label in zip(gts, preds):
            conf_matrix[true_label, pred_label] += 1

        class_acc_dict = {
            "class_names": self.class_names,
            "class_indices": np.arange(len(self.class_names)).tolist(),
            "accuracy": [],
            "fp": [],
            "fn": [],
            "tp": [],
            "tn": [],
            "num_samples": [],
            "fn_classes": [],
            "fp_classes": [],
            "precision": [],
            "recall": [],
            "f1_score": [],
        }
        for i in range(len(self.class_names)):
            class_indices = np.where(self.test_labels == i)[0]
            preds = all_prediction["pred_label"].to_numpy()
            gts = all_prediction["gt_label"].to_numpy()
            class_acc_dict["num_samples"].append(len(class_indices))

            tp = np.sum((gts == i) & (preds == i))
            fn = np.sum((gts == i) & (preds != i))
            fp = np.sum((gts != i) & (preds == i))
            tn = np.sum((gts != i) & (preds != i))

            class_acc_dict["tp"].append(tp)
            class_acc_dict["fp"].append(fp)
            class_acc_dict["fn"].append(fn)
            class_acc_dict["tn"].append(tn)

            acc = (tp + tn) / (tp + tn + fp + fn + 1e-10)
            precision = tp / (tp + fp + 1e-10)
            recall = tp / (tp + fn + 1e-10)
            f1_score = 2 * (precision * recall) / (precision + recall + 1e-10)

            class_acc_dict["precision"].append(precision)
            class_acc_dict["recall"].append(recall)
            class_acc_dict["f1_score"].append(f1_score)
            class_acc_dict["accuracy"].append(acc)

            fp_indics = np.where((gts != i) & (preds == i))[0]
            class_acc_dict["fp_classes"].append(self.test_labels[fp_indics].tolist())
            fn_indics = np.where((gts == i) & (preds != i))[0]
            class_acc_dict["fn_classes"].append(preds[fn_indics].tolist())

        if hasattr(self, "slice_info"):
            importance_scores = np.zeros(len(self.class_names))
            for i in range(len(self.class_names)):
                score = self.compute_importance_scores(i)
                importance_scores[i] = np.max(score)
            self.max_importance_score = np.max(importance_scores)
            class_acc_dict["importance_score"] = importance_scores / self.max_importance_score
        return {
            "all_prediction": all_prediction,
            "class_accuracy": pd.DataFrame(class_acc_dict),
            "confusion_matrix": conf_matrix,
        }

    def get_concept_info(self, latent_idx, resize_size=256, top_k_images=5):

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

    def precompute_slices(self, class_idx, top_k=50, confidence=0.95, min_samples=1, max_samples=50):
        out_dict = {}
        class_indices = np.where(self.test_labels == class_idx)[0]
        all_prediction = self.pred_results["all_prediction"]
        class_preds = all_prediction["pred_label"][class_indices].to_numpy()
        class_gt = all_prediction["gt_label"][class_indices].to_numpy()
        lower, upper = proportion_confint(
            np.sum(class_preds == class_gt),
            len(class_indices),
            alpha=1 - confidence,
            method="normal",
        )
        out_dict["metric"] = {
            "accuracy": np.mean(class_preds == class_gt),
            "lower_bound": lower,
            "upper_bound": upper,
        }
        class_activations = self.get_class_sae_activation(split=self.test_split, class_idx=class_idx)[
            :, self.conceptscope.valid_latents
        ]
        train_activations = self.get_class_sae_activation(split=self.train_split, class_idx=class_idx)[
            :, self.conceptscope.valid_latents
        ]
        train_indices = np.where(self.train_labels == class_idx)[0]

        concept_matrix = self.concept_matrix[:, class_idx][:, self.conceptscope.valid_latents]
        sigma_list = [0.0, 1.0, 2.0]
        top_indices_local = np.argsort(concept_matrix[0])[::-1][:top_k]
        out_dict["slice_info"] = {
            "latent_idx": [],
            "latent_name": [],
            "alignment_score": [],
            "avg_activation": concept_matrix[0][top_indices_local].tolist(),
            "highest_train_idices": [],
            "high_threshold": [],
            "low_threshold": [],
            "high_group_accuracy": [],
            "low_group_accuracy": [],
            "high_indices": [],
            "low_indices": [],
            "num_high_samples": [],
            "num_low_samples": [],
        }
        for sigma in sigma_list:
            out_dict["slice_info"][f"high_threshold_{sigma}"] = []
            out_dict["slice_info"][f"num_high_samples_{sigma}"] = []
            out_dict["slice_info"][f"num_low_samples_{sigma}"] = []
            out_dict["slice_info"][f"high_group_accuracy_{sigma}"] = []
            out_dict["slice_info"][f"low_group_accuracy_{sigma}"] = []
            out_dict["slice_info"][f"high_indices_{sigma}"] = []
            out_dict["slice_info"][f"low_indices_{sigma}"] = []

        alignment_score_dict = self.conceptscope.get_alignment_score(
            self.dataset_name,
            self.train_split,
            dataset=self.train_dataset,
            class_names=self.class_names,
            target_attribute=None,
        )
        aligment_scores = np.array(
            [score_dict["alignment_score"] for score_dict in alignment_score_dict[str(class_idx)].values()]
        )
        max_aligment_score = np.max(aligment_scores)
        min_aligment_score = np.min(aligment_scores)

        for idx in top_indices_local:
            global_idx = self.conceptscope.valid_latents[idx]
            out_dict["slice_info"]["latent_idx"].append(global_idx)
            out_dict["slice_info"]["latent_name"].append(self.conceptscope.concept_dict[str(global_idx)])

            try:
                aligment_score = alignment_score_dict[str(class_idx)][str(global_idx)]["alignment_score"]
                normalized_aligment_score = (aligment_score - min_aligment_score) / (
                    max_aligment_score - min_aligment_score + 1e-10
                )
            except KeyError:
                print(f"Alignment score for class {class_idx} and latent {global_idx} not found.")
                normalized_aligment_score = 0.5
            out_dict["slice_info"]["alignment_score"].append(normalized_aligment_score)

            highest_train_idices = train_indices[np.argsort(-train_activations[:, idx])[:10]]
            out_dict["slice_info"]["highest_train_idices"].append(highest_train_idices.tolist())

            high_threshold = np.percentile(class_activations[:, idx], 80)
            low_threshold = np.percentile(class_activations[:, idx], 20)
            out_dict["slice_info"]["high_threshold"].append(high_threshold)
            out_dict["slice_info"]["low_threshold"].append(low_threshold)
            high_indices = np.where(class_activations[:, idx] >= high_threshold)[0]
            low_indices = np.where(class_activations[:, idx] <= low_threshold)[0]
            sorted_high_indices = np.argsort(class_activations[high_indices, idx])[::-1]
            sorted_high_indices = class_indices[sorted_high_indices][:max_samples]
            sorted_low_indices = np.argsort(class_activations[low_indices, idx])
            sorted_low_indices = class_indices[sorted_low_indices][:max_samples]

            high_acc = np.mean(class_preds[high_indices] == class_gt[high_indices])
            low_acc = np.mean(class_preds[low_indices] == class_gt[low_indices])
            out_dict["slice_info"]["high_group_accuracy"].append(high_acc)
            out_dict["slice_info"]["low_group_accuracy"].append(low_acc)
            out_dict["slice_info"]["high_indices"].append(sorted_high_indices.tolist())
            out_dict["slice_info"]["low_indices"].append(sorted_low_indices.tolist())
            out_dict["slice_info"]["num_high_samples"].append(len(high_indices))
            out_dict["slice_info"]["num_low_samples"].append(len(low_indices))

            for sigma in sigma_list:
                high_threshold = concept_matrix[0][idx] + sigma * np.sqrt(concept_matrix[1][idx])

                high_indices = np.where(class_activations[:, idx] > high_threshold)[0]
                low_indices = np.array(list(set(np.arange(len(class_activations))) - set(high_indices)))

                sorted_high_indices = np.argsort(class_activations[high_indices, idx])[::-1]
                sorted_low_indices = np.argsort(class_activations[low_indices, idx])
                sorted_high_indices = class_indices[sorted_high_indices][:max_samples]
                sorted_low_indices = class_indices[sorted_low_indices][:max_samples]

                if len(high_indices) < min_samples:
                    high_acc = 0
                else:
                    high_acc = np.mean(class_preds[high_indices] == class_gt[high_indices])
                if len(low_indices) < min_samples:
                    low_acc = 0
                else:
                    low_acc = np.mean(class_preds[low_indices] == class_gt[low_indices])

                out_dict["slice_info"][f"high_threshold_{sigma}"].append(high_threshold)
                out_dict["slice_info"][f"num_high_samples_{sigma}"].append(len(high_indices))
                out_dict["slice_info"][f"num_low_samples_{sigma}"].append(len(low_indices))
                out_dict["slice_info"][f"high_group_accuracy_{sigma}"].append(high_acc)
                out_dict["slice_info"][f"low_group_accuracy_{sigma}"].append(low_acc)
                out_dict["slice_info"][f"high_indices_{sigma}"].append(sorted_high_indices.tolist())
                out_dict["slice_info"][f"low_indices_{sigma}"].append(sorted_low_indices.tolist())
        out_dict = make_json_serializable(out_dict)
        return out_dict

    def get_slices(self, class_idx, top_k=10, sigma=1.0):
        class_dict = self.slice_info[str(class_idx)].copy()

        concept_matrix = self.concept_matrix[:, class_idx][:, self.conceptscope.valid_latents]
        top_indices_local = np.argsort(concept_matrix[0])[::-1][:top_k]

        importance_scores = self.compute_importance_scores(class_idx) / self.max_importance_score

        latent_names = [
            self.conceptscope.concept_dict[str(idx)] for idx in class_dict["slice_info"]["latent_idx"][:top_k]
        ]

        out_dict = {}
        out_dict["metric"] = class_dict["metric"]
        out_dict["slice_info"] = {
            "latent_idx": class_dict["slice_info"]["latent_idx"][:top_k],
            "latent_name": latent_names,
            # "latent_name": class_dict["slice_info"]["latent_name"][:top_k],
            "image": [],
            "full_image": [],
            "avg_activation": class_dict["slice_info"]["avg_activation"][:top_k],
            "importance_score": importance_scores[:top_k].tolist(),
            "alignment_score": class_dict["slice_info"]["alignment_score"][:top_k],
            "highest_train_idices": class_dict["slice_info"]["highest_train_idices"][:top_k],
        }

        if sigma < 0:
            slice_dict = {
                "high_indices": class_dict["slice_info"]["high_indices"][:top_k],
                "low_indices": class_dict["slice_info"][f"low_indices"][:top_k],
                "num_high_samples": class_dict["slice_info"][f"num_high_samples"][:top_k],
                "num_low_samples": class_dict["slice_info"][f"num_low_samples"][:top_k],
                "high_group_accuracy": class_dict["slice_info"][f"high_group_accuracy"][:top_k],
                "low_group_accuracy": class_dict["slice_info"][f"low_group_accuracy"][:top_k],
                "high_threshold": class_dict["slice_info"][f"high_threshold"][:top_k],
                "low_threshold": class_dict["slice_info"][f"low_threshold"][:top_k],
            }
        else:
            slice_dict = {
                "high_indices": class_dict["slice_info"][f"high_indices_{sigma}"][:top_k],
                "low_indices": class_dict["slice_info"][f"low_indices_{sigma}"][:top_k],
                "num_high_samples": class_dict["slice_info"][f"num_high_samples_{sigma}"][:top_k],
                "num_low_samples": class_dict["slice_info"][f"num_low_samples_{sigma}"][:top_k],
                "high_group_accuracy": class_dict["slice_info"][f"high_group_accuracy_{sigma}"][:top_k],
                "low_group_accuracy": class_dict["slice_info"][f"low_group_accuracy_{sigma}"][:top_k],
                "high_threshold": class_dict["slice_info"][f"high_threshold_{sigma}"][:top_k],
            }
        out_dict["slice_info"].update(slice_dict)

        for idx in top_indices_local:
            global_idx = self.conceptscope.valid_latents[idx]

            image_path = f"{self.image_root}/{global_idx}.png"
            image = Image.open(image_path)
            cropped_image = crop_from_top_left(image)
            encoded_image = self.encode_image(cropped_image)
            full_image = self.encode_image(image)

            out_dict["slice_info"]["image"].append(encoded_image)
            out_dict["slice_info"]["full_image"].append(full_image)

        out_dict = make_json_serializable(out_dict)
        return out_dict

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

    def get_activation_from_indices(self, indices, classes, split="test", latent_idx=None):
        if split == "train":
            split = self.train_split
        else:
            split = self.test_split

        unique_classes = np.unique(classes)
        out = []
        for selected_class in unique_classes:
            class_indices = indices[np.where(classes == selected_class)[0]]
            class_indices_all, class_activations = self.get_class_indices_and_activations(
                class_idx=selected_class, split=split, latent_idx=latent_idx
            )
            local_class_indices = np.where(np.isin(class_indices_all, class_indices))[0]
            local_activations = class_activations[local_class_indices,]
            out.append(local_activations)
        out = np.concatenate(out, axis=0)
        return out

    def get_class_samples_umap(
        self,
        selected_class,
        top_k_latents=100,
        n_neighbors=15,
        min_dist=0.1,
        metric="cosine",
        seed=42,
    ):
        train_indices, train_activations = self.get_class_indices_and_activations(
            class_idx=selected_class, split=self.train_split
        )
        test_indices, test_activations = self.get_class_indices_and_activations(
            class_idx=selected_class, split=self.test_split
        )

        all_pred_labels = self.pred_results["all_prediction"]["pred_label"].to_numpy()
        all_gt_labels = self.pred_results["all_prediction"]["gt_label"].to_numpy()
        test_preds = all_pred_labels[test_indices]

        false_positive_indices = np.where((all_gt_labels != selected_class) & (all_pred_labels == selected_class))[0]
        fp_classes = all_gt_labels[false_positive_indices]
        test_fp_acivations = self.get_activation_from_indices(
            indices=false_positive_indices, classes=fp_classes, split=self.test_split, latent_idx=None
        )

        all_activations = np.concatenate([train_activations, test_activations, test_fp_acivations], axis=0)
        all_indices = np.concatenate([train_indices, test_indices, false_positive_indices], axis=0)

        latent_means = np.mean(all_activations, axis=0)
        top_latents = np.argsort(latent_means)[-top_k_latents:]
        selected_activations = all_activations[:, top_latents]

        reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, metric=metric, random_state=seed)
        embedding = reducer.fit_transform(selected_activations)

        pred_combined = np.concatenate(
            [[-1] * len(train_indices), test_preds, selected_class * np.ones(len(false_positive_indices))], axis=0
        )
        split_labels = (
            ["train"] * len(train_indices) + ["test"] * len(test_indices) + ["fp"] * len(false_positive_indices)
        )
        gt_combined = np.concatenate([[selected_class] * (len(train_indices) + len(test_indices)), fp_classes], axis=0)

        out_dict = {
            "UMAP-1": embedding[:, 0],
            "UMAP-2": embedding[:, 1],
            "split": split_labels,
            "indices": all_indices,
            "preds": pred_combined,
            "gts": gt_combined,
        }
        df = pd.DataFrame(out_dict)

        return df

    def get_class_samples_umap_clip(
        self,
        selected_class,
        top_k_latents=100,
        n_neighbors=15,
        min_dist=0.1,
        metric="cosine",
        seed=42,
    ):
        train_indices = np.where(self.train_labels == selected_class)[0]
        test_indices = np.where(self.test_labels == selected_class)[0]

        clip_save_root = (
            "/home/nas4_user/jinhochoi/repo/dataset_fingerprinting/out/dataset_bias/clip-vit-large-patch14/imagenet"
        )
        with h5py.File(f"{clip_save_root}/{self.train_split}_clip_embedding.h5", "r") as hf:
            train_activations = hf[f"class_{selected_class}"][:]
        with h5py.File(f"{clip_save_root}/{self.test_split}_clip_embedding.h5", "r") as hf:
            test_activations = hf[f"class_{selected_class}"][:]

        all_activations = np.concatenate([train_activations, test_activations], axis=0)
        all_indices = np.concatenate([train_indices, test_indices], axis=0)

        latent_means = np.mean(all_activations, axis=0)
        top_latents = np.argsort(latent_means)[-top_k_latents:]
        selected_activations = all_activations[:, top_latents]

        dim_reduction = PCA(n_components=top_k_latents, random_state=seed)
        selected_activations = dim_reduction.fit_transform(selected_activations)

        reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, metric=metric, random_state=seed)
        embedding = reducer.fit_transform(selected_activations)

        all_prediction = self.pred_results["all_prediction"]
        test_preds = all_prediction["pred_label"].to_numpy()[test_indices]
        pred_combined = np.concatenate([[-1] * len(train_indices), test_preds], axis=0)

        out_dict = {
            "UMAP-1": embedding[:, 0],
            "UMAP-2": embedding[:, 1],
            "split": ["train"] * len(train_indices) + ["test"] * len(test_indices),
            "indices": all_indices,
            "preds": pred_combined,
        }
        df = pd.DataFrame(out_dict)

        return df

    def get_class_umap_data(
        self,
        concept_type="all",
        n_neighbors=15,
        min_dist=0.1,
        metric="cosine",
        run_dim_reduction=True,
        run_clustering=True,
        reduced_dim=100,
        min_cluster_size=10,
        seed=42,
    ):

        concept_matrix = self._filter_concept_matrix(concept_type)
        df = self.run_umap(
            input_matrix=concept_matrix,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            run_dim_reduction=run_dim_reduction,
            run_clustering=run_clustering,
            reduced_dim=reduced_dim,
            min_cluster_size=min_cluster_size,
            seed=seed,
        )
        df["label"] = self.class_names
        df["class_idx"] = np.arange(len(self.class_names))

        return df

    def run_umap(
        self,
        input_matrix,
        n_neighbors=15,
        min_dist=0.1,
        metric="cosine",
        run_dim_reduction=True,
        run_clustering=True,
        reduced_dim=100,
        min_cluster_size=10,
        seed=42,
    ):

        if run_dim_reduction:
            pca = PCA(n_components=reduced_dim, random_state=seed)
            input_matrix = pca.fit_transform(input_matrix)

        reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, metric=metric, random_state=seed)
        embedding = reducer.fit_transform(input_matrix)

        df = pd.DataFrame({"UMAP-1": embedding[:, 0], "UMAP-2": embedding[:, 1]})

        if run_clustering:
            clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
            cluster_labels = clusterer.fit_predict(embedding)
            df["Cluster"] = cluster_labels

        return df

    def get_cluster_concepts(self, class_indices, top_k=10):
        class_indices = np.array(class_indices).astype(int)
        concept_matrix = self.concept_matrix[class_indices, :][:, self.conceptscope.valid_latents]
        avg_concept_matrix = np.mean(concept_matrix, axis=0)
        highest_indices = np.argsort(avg_concept_matrix)[::-1][:top_k]
        summary_concept_dist = avg_concept_matrix[highest_indices]

        highest_indices = np.array(self.conceptscope.valid_latents)[highest_indices]
        concept_names = [self.conceptscope.concept_dict[str(idx)] for idx in highest_indices]

        images = []
        image_root = "/home/nas4_user/jinhochoi/repo/ConceptScope/out/checkpoints/openai_l14_32K_base/reference_images"
        for idx in highest_indices:
            image_path = f"{image_root}/{idx}.png"
            images.append(Image.open(image_path))
        encoded_images = self.encode_images(images)

        df = pd.DataFrame(
            {
                "summary_concept_dist": summary_concept_dist,
                "highest_indices": highest_indices,
                "concept_names": concept_names,
            }
        )
        return df, encoded_images

    def get_concept_distribution(self, selected_class, bias_sigma=1, max_concepts=20):
        data = {
            "Class aligned": [],
            "Mean": [],
            "Concept Type": [],
            "latent_idx": [],
            "latent_name": [],
            "slice_idx": [],
        }

        context_means = []
        len_target = len(self.concept_categorization_dict[str(selected_class)]["target"])
        len_context = max_concepts - len_target
        all_slices = (
            self.concept_categorization_dict[str(selected_class)]["target"]
            + self.concept_categorization_dict[str(selected_class)]["context"][:len_context]
        )
        for i, slice_info in enumerate(all_slices):
            data["latent_idx"].append(slice_info["latent_idx"])
            data["latent_name"].append(slice_info["latent_name"])
            data["Mean"].append(slice_info["mean_activation"])
            data["Class aligned"].append(slice_info["normalized_alignment_score"])
            data["slice_idx"].append(i)
            if "bias" not in slice_info:
                data["Concept Type"].append("target")
            else:
                context_means.append(slice_info["mean_activation"])
                if slice_info["bias"] == True:
                    data["Concept Type"].append("bias")
                else:
                    data["Concept Type"].append("context")

        context_means = np.array(context_means)
        bias_threshold = np.mean(context_means) + bias_sigma * np.std(context_means)

        df = pd.DataFrame(data)
        color_map = {"target": "green", "context": "orange", "bias": "red"}
        df["Color"] = df["Concept Type"].map(color_map)
        df["bias_threshold"] = bias_threshold

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

    def get_images(
        self,
        split,
        class_idx,
        latent_idx,
        resize_size=256,
        top_k=10,
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

        sosrted_indices = np.argsort(latent_activations)[::-1]
        highest_indices = sosrted_indices[:top_k]
        lowest_indices = sosrted_indices[-top_k:]

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
        return {
            "highest_images": self.encode_images(highest_images),
            "lowest_images": self.encode_images(lowest_images),
            "masked_highest_images": self.encode_images(masked_highest_images),
        }

    def get_all_images(self, class_idx, latent_idx, resize_size=256, top_k=10):
        out_dict = {}

        for i, split in enumerate([self.train_split, self.test_split]):
            images_data = self.get_images(
                split=split,
                class_idx=class_idx,
                latent_idx=latent_idx,
                resize_size=resize_size,
                top_k=top_k,
            )
            if i == 0:
                split_name = "train"
            else:
                split_name = "test"
            out_dict[split_name] = images_data
        return out_dict

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
