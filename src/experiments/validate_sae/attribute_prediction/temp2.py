import argparse
import json
import os

import h5py
import numpy as np
import torch
from PIL import Image
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import auc, precision_recall_curve, precision_score, recall_score
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from src.conceptscope.conceptscope import ConceptScope
from src.experiments.validate_sae.attribute_prediction.attr_prediction_evaluator import (
    AttrPredictionEvaluator,
)
from src.utils.image_dataset_loader import ImageDatasetLoader
from src.utils.processor import BatchIterator, H5ActivationtWriter, ImageProcessor
from src.utils.utils import (
    apply_sae_mask_to_input,
    get_len_dataset,
    get_sae_mask,
    load_class_names,
    make_json_serializable,
)


class SAEAttributePredictionEvaluator(ConceptScope):
    def __init__(
        self,
        save_root,
        sae_path,
        device,
        backbone,
        dir_name="attribute_prediction",
        checkpoint_dir_name="checkpoints",
        clip_model_name="openai/clip-vit-large-patch14",
        data_root="./data",
    ):
        super().__init__(
            save_root=save_root,
            sae_path=sae_path,
            device=device,
            backbone=backbone,
            dir_name=dir_name,
            clip_model_name=clip_model_name,
            checkpoint_dir_name=checkpoint_dir_name,
            data_root=data_root,
        )
        self.eval_helper = AttrPredictionEvaluator()
        self.num_samples = 100

    def load_concept_dict(self):
        save_path = f"{self.checkpoint_path}/concept_dict.json"
        with open(save_path, "r") as f:
            concept_dict = json.load(f)
            valid_latents = np.array(list(concept_dict.keys())).astype(int)
        return concept_dict, valid_latents

    def get_ref_image_clip_embeddings(self, imagenet_dataset, latent_idx, top_k=5, resize_size=256):
        img_indices = self.max_act_imgs[latent_idx]
        images = []
        for i in img_indices[:top_k]:
            images.append(imagenet_dataset[i.item()]["image"])

        sae_masks = get_sae_mask(
            images, self.sae, self.vit, self.cfg, latent_idx, resize_size=resize_size, device=self.device
        )
        masked_images, _ = apply_sae_mask_to_input(images, sae_masks)
        masked_image_embedding = self.get_image_embedding(masked_images)
        return masked_image_embedding.cpu().numpy()

    def load_clip_image_embedding(self, dataset_name, split, dataset, batch_size=64):
        save_dir = f"{self.save_root}/{self.dir_name}/{dataset_name}/{split}_clip_image_embedding.h5"
        if os.path.exists(save_dir):
            with h5py.File(save_dir, "r") as hf:
                clip_image_embedding = hf["clip_image_embedding"][:]
            return clip_image_embedding

        num_data = get_len_dataset(dataset)
        clip_embedding = np.zeros((num_data, self.clip_model.config.projection_dim), dtype=np.float32)
        for i in tqdm(range(0, num_data, batch_size)):
            batch_images = []
            for j in range(i, min(i + batch_size, num_data)):
                image = self.dataset[j]["image"]
                if isinstance(image, str):
                    image = Image.open(image)
                batch_images.append(image)
            image_embedding = self.get_image_embedding(batch_images)
            clip_embedding[i : i + len(batch_images)] = image_embedding.cpu().numpy()
        with h5py.File(save_dir, "w") as hf:
            hf.create_dataset("clip_image_embedding", data=clip_embedding, compression="gzip")
        return clip_embedding

    def compute_correlation_with_clip(
        self,
        attr_name,
        image_embedding,
        test_activations,
        latent_idx,
        max_chunk=100,
    ):
        attr_name = " ".join(attr_name.split("_")).lower()
        text_embedding = self.clip_processor.get_text_embedding(attr_name).cpu().numpy()

        activation = test_activations[:, latent_idx]
        selected_indices = np.where(activation > 0)[0]

        latent_scores = activation[selected_indices]
        cos_sim_all = cosine_similarity(image_embedding[selected_indices], text_embedding).flatten()

        sorted_indices = np.argsort(latent_scores)
        cos_sim_chunk = []
        scores_chunk = []

        num_chunk = min(max_chunk, len(selected_indices))
        num_sample = max(1, len(latent_scores) // num_chunk)
        for i in range(num_chunk):
            indices = sorted_indices[i * num_sample : (i + 1) * num_sample]
            cos_sim_chunk.append(float(cos_sim_all[indices].mean()))
            scores_chunk.append(float(latent_scores[indices].mean()))

        cos_np = np.array(cos_sim_chunk)
        scores_np = np.array(scores_chunk)
        pearson, _ = pearsonr(cos_np, scores_np)
        spearman, _ = spearmanr(cos_np, scores_np)
        group_scores = {
            "pearson": float(pearson),
            "spearman": float(spearman),
        }

        return group_scores

    def _process_class(self, dataset, activation_writer, class_mean_var_activations, cls_idx, batch_size=64):

        batch_iterator = BatchIterator.get_batches(dataset, batch_size)
        num_samples = get_len_dataset(dataset)
        class_activations = np.zeros((num_samples, self.cfg.d_sae), dtype=np.float32)

        for start_idx, end_idx, batch in batch_iterator:
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

            avg_sae_latents = sae_latents.mean(dim=1)
            class_activations[start_idx:end_idx] = avg_sae_latents.cpu().numpy()

        avg_sae_latents = class_activations.mean(axis=0)
        std_sae_latents = class_activations.std(axis=0)

        class_mean_var_activations[0][cls_idx] = avg_sae_latents
        class_mean_var_activations[1][cls_idx] = std_sae_latents

        activation_writer.create_cls_dataset(cls_idx, num_samples)
        activation_writer.write_cls_activation(
            class_activations,
            cls_idx=cls_idx,
        )

    def load_all_assets(self, dataset_name, split, target_attribute=None):

        if dataset_name == "waterbird":
            target_attribute = "detailed_place"
        elif dataset_name == "celeba" and target_attribute is None:
            target_attribute = "Blond_Hair"

        eval_dataset = ImageDatasetLoader.load_dataset(dataset_name, split=split, target_attribute=target_attribute)
        eval_labels = np.array(eval_dataset["label"])

        attr_list = self.eval_helper.get_attr_list(dataset_name, eval_dataset)

        num_classes = len(np.unique(eval_labels))
        class_names = load_class_names(self.data_root, dataset_name, eval_dataset, target_attribute)

        eval_sae_activations = self.get_sae_latent_activations(dataset_name, split, eval_dataset)

        if dataset_name in ["raf-db", "stanford_action"]:
            eval_dataset, indices_eval = self.eval_helper.subsample_dataset_by_class(
                eval_dataset, num_samples=self.num_samples
            )
            eval_sae_activations = eval_sae_activations[indices_eval,]

        return {
            "eval_dataset": eval_dataset,
            "eval_labels": eval_labels,
            "attr_list": attr_list,
            "num_classes": num_classes,
            "class_names": class_names,
            "eval_sae_activations": eval_sae_activations,
        }

    def get_latent(self, training_dataset, training_activation, dataset_name, cls_idx):
        binary_target = self.eval_helper.get_binary_target(training_dataset, dataset_name, cls_idx)

        target_indices = self.valid_latents
        roc_scores = np.zeros(len(target_indices))
        for i, latent_idx in enumerate(target_indices):
            latent_activation_arr = training_activation[:, latent_idx]
            if len(np.where(latent_activation_arr > 0)[0]) < 100:
                continue
            precision, recall, thresholds = precision_recall_curve(binary_target, latent_activation_arr)
            roc = auc(recall, precision)
            roc_scores[i] = roc
        max_score_idx = np.argmax(roc_scores)
        max_latent_idx = target_indices[max_score_idx]

        precision, recall, thresholds = precision_recall_curve(binary_target, training_activation[:, max_latent_idx])
        roc = auc(recall, precision)
        f1_score = 2 * (precision * recall) / (precision + recall + 1e-10)
        best_threshold_idx = np.argmax(f1_score)

        return {
            "latent_idx": max_latent_idx,
            "roc_auc": roc,
            "best_f1": f1_score.max(),
            "best_precision": precision[best_threshold_idx],
            "best_recall": recall[best_threshold_idx],
            "best_threshold": thresholds[best_threshold_idx],
        }

    def compute_metric(self, target, latent_activations, high_indices):

        roc_scores = np.zeros(len(high_indices))
        for j, latent_idx in enumerate(high_indices):
            latent_activation = latent_activations[:, latent_idx]
            precision, recall, _ = precision_recall_curve(target, latent_activation)
            precision = np.nan_to_num(precision, nan=0)
            recall = np.nan_to_num(recall, nan=0)

            roc = auc(recall, precision)
            roc_scores[j] = roc
        max_score_idx = np.argmax(roc_scores)
        latent_idx = high_indices[max_score_idx]

        latent_activation = latent_activations[:, latent_idx]
        precision, recall, thresholds = precision_recall_curve(target, latent_activation)
        f1_score = 2 * (precision * recall) / (precision + recall + 1e-10)
        best_f1_idx = np.argmax(f1_score)

        recall_interp = np.linspace(0, 1, 100)
        precision_interp = np.interp(recall_interp, recall[::-1], precision[::-1])

        return {
            "roc_auc": roc_scores[max_score_idx],
            "best_f1": float(f1_score[best_f1_idx]),
            "best_precision": float(precision[best_f1_idx]),
            "best_recall": float(recall[best_f1_idx]),
            "precision": precision_interp,
            "recall": recall_interp,
        }, latent_idx

    def run(self, dataset_name, split):
        all_data = self.load_all_assets(dataset_name, split)
        all_score_dict = {}

        mean_acts = all_data["eval_sae_activations"].mean(0)
        high_indices = np.argsort(mean_acts)[-2000:]

        for cls_idx, attr in tqdm(enumerate(all_data["attr_list"]), desc="Evaluating Attributes"):
            if dataset_name == "celeba":
                all_data = self.load_all_assets(dataset_name, split, target_attribute=attr)

            binary_target = self.eval_helper.get_binary_target(all_data["eval_dataset"], dataset_name, cls_idx)
            score_dict, latent_idx = self.compute_metric(binary_target, all_data["eval_sae_activations"], high_indices)

            if dataset_name == "celeba":
                class_name = all_data["class_names"][1]
            else:
                class_name = all_data["class_names"][cls_idx]

            all_score_dict[cls_idx] = {"class_name": class_name, "latent_idx": latent_idx}
            all_score_dict[cls_idx].update(score_dict)

        metric_keys = list(score_dict.keys())
        metrics_across_attrs = {k: [] for k in metric_keys}
        for entry in all_score_dict.values():
            for k in metric_keys:
                metrics_across_attrs[k].append(entry[k])

        mean_entry = {}
        std_entry = {}

        for k in metric_keys:
            values = np.array(metrics_across_attrs[k])
            if values.ndim == 1:
                mean_entry[k] = values.mean()
                std_entry[k] = values.std()
            else:
                mean_entry[k] = values.mean(axis=0).tolist()
                std_entry[k] = values.std(axis=0).tolist()

        all_score_dict["mean"] = mean_entry
        all_score_dict["std"] = std_entry

        save_dir = f"{self.save_root}/{self.dir_name}/{dataset_name}/{self.checkpoint_name}"
        save_path = f"{save_dir}/{split}_sae_attr_prediction_prev.json"
        json_ready_dict = make_json_serializable(all_score_dict)
        with open(save_path, "w") as f:
            json.dump(json_ready_dict, f, indent=4)


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
    args = parser.parse_args()

    sae_attr_pred_evaluator = SAEAttributePredictionEvaluator(
        save_root=args.save_root,
        dir_name=args.dir_name,
        sae_path=args.sae_path,
        device=args.device,
        backbone=args.backbone,
        clip_model_name=args.clip_model_name,
    )
    split_dict = {
        # "caltech101": "train",
        # # "celeba": "test",
        # "dtd": "test",
        # "waterbird": "test",
        "stanford_action": "test",
        "raf-db": "test",
    }
    for dataset_name, split in split_dict.items():
        print(f"Running evaluation for {dataset_name} on {split} split")
        with torch.no_grad():
            sae_attr_pred_evaluator.run(dataset_name, split)
