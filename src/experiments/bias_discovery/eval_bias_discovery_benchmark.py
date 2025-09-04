import argparse
import json
import os

import h5py
import numpy as np
import torch
from tqdm import tqdm

from src.conceptscope.conceptscope import ConceptScope
from src.utils.image_dataset_loader import ImageDatasetLoader
from src.utils.processor import H5ActivationtWriter
from src.utils.utils import load_class_names, make_json_serializable


class ConceptScopeBiasDiscovery(ConceptScope):

    def load_all_data(self, dataset_name, split):
        if dataset_name == "waterbird":
            target_attribute = "species"
            train_split = "train"
            context_attribute = "place"
        elif dataset_name == "celeba":
            target_attribute = "Blond_Hair"
            context_attribute = "Male"
            train_split = "valid"
        else:
            target_attribute = None
            context_attribute = "context"
            train_split = "train"

        train_dataset = ImageDatasetLoader.load_dataset(
            dataset_name, split=train_split, target_attribute=target_attribute
        )
        train_sae_activations = self.get_sae_latent_activations(dataset_name, train_split, train_dataset)

        eval_dataset = ImageDatasetLoader.load_dataset(dataset_name, split=split, target_attribute=target_attribute)
        eval_labels = np.array(eval_dataset["label"])
        context_labels = np.array(eval_dataset[context_attribute])

        num_classes = len(np.unique(eval_labels))
        class_names = load_class_names(self.data_root, dataset_name, eval_dataset, target_attribute)

        eval_sae_activations = self.get_sae_latent_activations(dataset_name, split, eval_dataset)

        return {
            "dataset": eval_dataset,
            "labels": eval_labels,
            "context_labels": context_labels,
            "num_classes": num_classes,
            "class_names": class_names,
            "sae_activations": eval_sae_activations,
            "target_attribute": target_attribute,
            "context_attribute": context_attribute,
            "train_sae_activations": train_sae_activations,
            "train_dataset": train_dataset,
            "train_split": train_split,
        }

    def get_concept_categorization(
        self,
        dataset_name,
        split,
        dataset,
        class_names,
        target_attribute,
        bias_threshold_sigma=1.0,
        target_threshold=0.0,
    ):
        alignment_scores_dict = self.get_alignment_score(dataset_name, split, dataset, class_names, target_attribute)

        concept_catetorized_dict = self.concept_cateogorization(
            dataset_name,
            split,
            class_names,
            alignment_scores_dict,
            bias_threshold_sigma=bias_threshold_sigma,
            target_threshold=target_threshold,
        )
        return concept_catetorized_dict

    def evalute_bias_discovery_benchmark(
        self, dataset_name, split, num_precision=10, max_num=100, bias_threshold_sigma=1.0, target_threshold=0.0
    ):
        all_data = self.load_all_data(dataset_name, split)
        concept_catetorized_dict = self.get_concept_categorization(
            dataset_name,
            all_data["train_split"],
            all_data["train_dataset"],
            all_data["class_names"],
            all_data["target_attribute"],
            bias_threshold_sigma=bias_threshold_sigma,
            target_threshold=target_threshold,
        )

        score_dict = {}
        for selected_class in range(all_data["num_classes"]):
            if dataset_name == "celeba" and selected_class == 0:
                continue
            score_dict[selected_class] = {}

            class_indices = np.where(all_data["labels"] == selected_class)[0]
            for class_idx in range(all_data["num_classes"]):
                if class_idx == selected_class:
                    continue
                score_dict[selected_class][class_idx] = []
                latent_indices = []
                for slice_info in concept_catetorized_dict[str(class_idx)]["context"]:
                    if slice_info["bias"] is True:
                        latent_indices.append(slice_info["latent_idx"])
                for latent_idx in latent_indices:

                    latent_cls_act = all_data["sae_activations"][class_indices][:, latent_idx]
                    top_indices = np.argsort(-latent_cls_act)[:max_num]
                    selected_context_label = all_data["context_labels"][class_indices][top_indices]

                    if dataset_name == "celeba":
                        scores = selected_context_label[:num_precision].sum() / num_precision
                    else:
                        scores = (selected_context_label[:num_precision] == class_idx).sum() / num_precision
                    latent_dict = {
                        "score": scores,
                        "latent_idx": latent_idx,
                        "concept": self.concept_dict[str(latent_idx)],
                    }

                    for k in [20, 30, 40, 50]:
                        precsion_score = (
                            selected_context_label[:k].sum() / k
                            if dataset_name == "celeba"
                            else (selected_context_label[:k] == class_idx).sum() / k
                        )
                        latent_dict[f"precision_{k}"] = precsion_score
                    score_dict[selected_class][class_idx].append(latent_dict)

        all_scores = []
        for class_dict in score_dict.values():
            class_scores = []
            for score_info in class_dict.values():
                class_scores = np.array([latent_info["score"] for latent_info in score_info])
                all_scores.append(np.max(class_scores))

        all_scores = np.array(all_scores)
        avg_scores = np.mean(all_scores)
        std_scores = np.std(all_scores)

        score_dict["average"] = {"score": avg_scores, "std": std_scores}
        score_dict = make_json_serializable(score_dict)
        save_dir = f"{self.save_root}/{self.dir_name}/{dataset_name}/{self.checkpoint_name}"
        save_path = f"{save_dir}/{split}_bias_discovery_scores.json"
        with open(save_path, "w") as f:
            json.dump(score_dict, f, indent=4)

    # def evalute_bias_discovery_benchmark(self, dataset_name, split, num_precision=10, bias_threshold_sigma=1.0):
    #     all_data = self.load_all_data(dataset_name, split)
    #     concept_catetorized_dict = self.get_concept_categorization(
    #         dataset_name,
    #         all_data["train_split"],
    #         all_data["train_dataset"],
    #         all_data["class_names"],
    #         all_data["target_attribute"],
    #         bias_threshold_sigma=bias_threshold_sigma,
    #     )

    #     score_dict = {}
    #     for selected_class in range(all_data["num_classes"]):
    #         if dataset_name == "celeba" and selected_class == 0:
    #             continue
    #         score_dict[selected_class] = {}

    #         class_indices = np.where(all_data["labels"] == selected_class)[0]
    #         for class_idx in range(all_data["num_classes"]):
    #             if class_idx == selected_class:
    #                 continue
    #             score_dict[selected_class][class_idx] = {}
    #             latent_indices = []
    #             for slice_info in concept_catetorized_dict[str(class_idx)]["context"]:
    #                 if slice_info["bias"] is True:
    #                     latent_indices.append(slice_info["latent_idx"])
    #             latent_indices = np.array(latent_indices)
    #             latent_cls_act = all_data["sae_activations"][class_indices][:, latent_indices].mean(-1)
    #             top_indices = np.argsort(-latent_cls_act)[:num_precision]
    #             selected_context_label = all_data["context_labels"][class_indices][top_indices]

    #             if dataset_name == "celeba":
    #                 scores = selected_context_label.sum() / num_precision
    #             else:
    #                 scores = (selected_context_label == class_idx).sum() / num_precision
    #             score_dict[selected_class][class_idx]["score"] = scores
    #             score_dict[selected_class][class_idx]["latent"] = latent_indices.tolist()
    #             score_dict[selected_class][class_idx]["concept"] = [self.concept_dict[str(i)] for i in latent_indices]
    #     all_scores = []
    #     for class_dict in score_dict.values():
    #         for score_info in class_dict.values():
    #             all_scores.append(score_info["score"])
    #     all_scores = np.array(all_scores)

    #     avg_scores = np.mean(all_scores)
    #     std_scores = np.std(all_scores)

    #     score_dict["average"] = {"score": avg_scores, "std": std_scores}
    #     score_dict = make_json_serializable(score_dict)
    #     save_dir = f"{self.save_root}/{self.dir_name}/{dataset_name}/{self.checkpoint_name}"
    #     save_path = f"{save_dir}/{split}_bias_discovery_scores.json"
    #     with open(save_path, "w") as f:
    #         json.dump(score_dict, f, indent=4)

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

    def get_sae_latent_activations(self, dataset_name, split, dataset):
        file_dir = f"{self.save_root}/{self.dir_name}/{dataset_name}/{self.checkpoint_name}"
        os.makedirs(file_dir, exist_ok=True)
        file_name = f"{file_dir}/{split}_sae_latents.h5"

        if dataset_name == "waterbird" or dataset_name == "celeba":
            dataset = ImageDatasetLoader.load_dataset(dataset_name, split=split)

        if not os.path.exists(file_name):
            self.extract_sae_latent_activations_and_save(file_name, dataset, dataset_name)

        class_labels = np.array(dataset[self.label_key])
        num_classes = len(set(class_labels))
        activation_all = np.zeros((len(class_labels), self.cfg.d_sae), dtype=np.float32)

        for i in tqdm(range(num_classes), desc="Loading SAE activations"):
            class_indices = np.where(class_labels == i)[0]
            with h5py.File(file_name, "r") as hf:
                cls_activation = hf[f"activations_{i}"][:]
            activation_all[class_indices] = cls_activation
        return activation_all


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_root", type=str, default="./out", help="Root directory to save results")
    parser.add_argument(
        "--dir_name", type=str, default="dataset_analysis", help="Name of the directory to save results"
    )
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints/submitted", help="Name of the checkpoint")
    parser.add_argument("--sae_path", type=str, required=True, help="Path to the SAE model checkpoint")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model on")
    parser.add_argument("--backbone", type=str, default="openai/clip-vit-large-patch14", help="Backbone model name")
    parser.add_argument("--clip_model_name", type=str, default="openai/clip-vit-large-patch14", help="CLIP model name")
    parser.add_argument("--bias_threshold_sigma", type=float, default=1.0, help="Bias threshold in sigma")
    parser.add_argument(
        "--target_threshold",
        type=float,
        default=0.0,
        help="Threshold for the target attribute to consider a sample as positive",
    )
    args = parser.parse_args()

    evaluator = ConceptScopeBiasDiscovery(
        save_root=args.save_root,
        sae_path=args.sae_path,
        device=args.device,
        backbone=args.backbone,
        clip_model_name=args.clip_model_name,
        dir_name=args.dir_name,
    )

    split_dict = {
        "waterbird": "test",
        "celeba": "test",
        "nico_95": "test",
        "nico_90": "test",
        "nico_75": "test",
    }
    for dataset_name, split in split_dict.items():
        print(f"Running evaluation for {dataset_name} on {split} split")
        with torch.no_grad():
            evaluator.evalute_bias_discovery_benchmark(
                dataset_name,
                split,
                bias_threshold_sigma=args.bias_threshold_sigma,
                target_threshold=args.target_threshold,
            )
