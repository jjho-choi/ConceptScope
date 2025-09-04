import argparse
import json
import os
import random
from glob import glob

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import auc, precision_recall_curve
from tqdm import tqdm

from src.conceptscope.conceptscope import ConceptScope
from src.utils.processor import ImageProcessor
from src.utils.utils import make_json_serializable


class SegmentationEvalHelper:
    def __init__(self, data_root):
        self.data_root = data_root
        self.mask_size = 112
        self.image_root = f"{data_root}/images"
        self.images_df = pd.read_csv(os.path.join(data_root, "index.csv"))
        self.category = pd.read_csv(os.path.join(data_root, "category.csv"))
        self.label = pd.read_csv(os.path.join(data_root, "label.csv"))
        self.category_map = {}
        for i, row in self.category.iterrows():
            category_name = row["name"]
            with open(f"{data_root}/c_{category_name}.csv", "r") as f:
                category_data = pd.read_csv(f)
                self.category_map[category_name] = category_data
        self.get_all_concept_indicators()

    def get_all_concept_indicators(self):
        all_concept_indicator = np.zeros((len(self.images_df), len(self.label)))
        for i in tqdm(range(len(self.images_df))):
            _, _, concept_indicator = self.get_image_and_mask(i)
            all_concept_indicator[i] = concept_indicator

        np.savez_compressed(
            os.path.join(self.data_root, "all_concept_indicator.npz"), all_concept_indicator=all_concept_indicator
        )

    def get_train_test_split(self, all_concept_indicator, seed=42, train_rate=0.1):
        all_train_test_splits = {}
        for i in tqdm(range(len(self.label))):
            concept_indicator = all_concept_indicator[:, i]
            valid_indices = np.where(concept_indicator > 0)[0]
            np.random.seed(seed)
            np.random.shuffle(valid_indices)
            train_indices = valid_indices[: int(len(valid_indices) * train_rate)]
            test_indices = valid_indices[int(len(valid_indices) * train_rate) :]
            all_train_test_splits[str(i)] = {"train": train_indices.tolist(), "test": test_indices.tolist()}

        with open(os.path.join(self.data_root, "train_test_split.json"), "w") as f:
            json.dump(all_train_test_splits, f)

        return all_train_test_splits

    def get_image_and_mask(self, idx):
        selected_row = self.images_df.iloc[idx]
        image_path = os.path.join(self.image_root, selected_row["image"])
        image = Image.open(image_path).convert("RGB")
        masks = {}
        concept_indicators = np.zeros(len(self.label), dtype=bool)
        for concept in self.category_map:
            if pd.isna(selected_row[concept]):
                continue
            if concept in selected_row:
                if concept in ["scene", "texture"]:
                    if isinstance(selected_row[concept], str):
                        labels = selected_row[concept].split(";")
                    else:
                        labels = [selected_row[concept]]
                    for label in labels:
                        concept_idx = int(label) - 1
                        if concept_idx == -1:
                            continue
                        concept_indicators[concept_idx] = True
                        concept_mask = np.ones((self.mask_size, self.mask_size), dtype=np.uint8)
                        masks[concept_idx] = concept_mask
                else:
                    annot_files = selected_row[concept].split(";")
                    for annot_file in annot_files:
                        mask_path = os.path.join(self.image_root, annot_file)
                        row_mask = np.array(Image.open(mask_path))[:, :, 0]
                        unique_labels = np.unique(row_mask)
                        for label in unique_labels:
                            if label == 0:
                                continue
                            concept_indicators[label - 1] = True
                            concept_mask = np.zeros((self.mask_size, self.mask_size), dtype=np.uint8)
                            concept_mask[row_mask == label] = 1
                            if label == 0:
                                print(f"Warning: Found label 0 in mask for {image_path}")
                            masks[label - 1] = concept_mask

        return image, masks, concept_indicators

    def process_image(self, image_path, resize_size=256):
        image = Image.open(image_path).convert("RGB").resize((resize_size, resize_size))
        return image

    def process_mask(self, mask_path, resize_size=256):
        mask = Image.open(mask_path.replace("jpg", "png"))
        mask = mask.resize((resize_size, resize_size), Image.NEAREST)  # Nearest to keep class indices intact
        return np.array(mask, dtype=np.uint8)

    def get_binary_mask(self, mask, class_index):
        binary_mask = np.zeros_like(mask, dtype=np.uint8)
        binary_mask[mask == class_index] = 1
        return binary_mask

    def resize_mask(self, mask, resize_size=256):
        mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-10)
        mask = (
            torch.nn.functional.interpolate(mask, (resize_size, resize_size), mode="bilinear", align_corners=False)
            .cpu()
            .numpy()
        )
        return mask

    def evaluate_auprc(self, pred_mask, gt_mask):
        precision, recall, _ = precision_recall_curve(gt_mask.flatten(), pred_mask.flatten())
        score = auc(recall, precision)
        return score

    def save_results(self, results, class_names, save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        out_dict = {}
        for i, class_name in enumerate(class_names):
            out_dict[class_name] = results[i]
        mean = np.mean(results)
        std = np.std(results)
        out_dict["mean"] = mean
        out_dict["std"] = std

        out_dict = make_json_serializable(out_dict)

        with open(save_path, "w") as f:
            json.dump(out_dict, f, indent=4)


class SAESegmentationEvaluator(ConceptScope):
    def __init__(
        self,
        save_root,
        sae_path,
        device="cuda",
        backbone="openai/clip-vit-large-patch14",
        clip_model_name="openai/clip-vit-large-patch14",
        dir_name="segmentation",
        data_root="./data",
    ):
        super().__init__(
            save_root=save_root,
            sae_path=sae_path,
            device=device,
            backbone=backbone,
            dir_name=dir_name,
            clip_model_name=clip_model_name,
        )
        self.eval_helper = SegmentationEvalHelper(data_root)
        self.resize_size = self.eval_helper.mask_size

    def get_seg_mask_pred(self, image):
        inputs = ImageProcessor.process_model_inputs({"image": [image]}, self.vit, self.device)
        sae_act = ImageProcessor.get_sae_activations(
            self.sae,
            self.vit,
            inputs,
            self.cfg.block_layer,
            self.cfg.module_name,
            self.cfg.class_token,
            get_mean=False,
        )
        sae_act = sae_act[:, 1:]
        feature_size = int(np.sqrt(sae_act.shape[1]))
        sae_act = sae_act.permute(0, 2, 1)
        masks = torch.Tensor(sae_act.reshape(1, sae_act.shape[1], feature_size, feature_size))
        return masks

    def evaluate_test_dataset(self, dataset, class_latent_mapping):
        scores_dict = {i: [] for i in range(len(class_latent_mapping))}
        for i, sample in enumerate(tqdm(dataset)):
            image = self.eval_helper.process_image(sample["image"])
            mask = self.eval_helper.process_mask(sample["mask"], resize_size=self.resize_size)
            sae_mask = self.get_seg_mask_pred(image)
            for j in range(len(class_latent_mapping)):
                binary_mask = self.eval_helper.get_binary_mask(mask, j + 1)
                if binary_mask.sum() == 0:
                    continue
                latent_idx = class_latent_mapping[str(j)]["latent_idx"]
                pred_mask = self.eval_helper.resize_mask(sae_mask[:, latent_idx : latent_idx + 1], self.resize_size)
                score = self.eval_helper.evaluate_auprc(pred_mask, binary_mask)
                scores_dict[j].append(score)

        scores_by_class = np.zeros(len(class_latent_mapping), dtype=np.float32)
        for class_idx, scores in scores_dict.items():
            scores_by_class[class_idx] = np.mean(scores) if scores else 0.0

        return scores_by_class

    def get_latent_mapping(self, dataset, class_names):
        save_path = f"{self.checkpoint_path}/latent_class_mapping_for_segmentation.json"
        if os.path.exists(save_path):
            with open(save_path, "r") as f:
                latent_class_mapping = json.load(f)
            return latent_class_mapping

        scores = np.zeros((len(dataset), len(class_names), len(self.valid_latents)), dtype=np.float32)
        for i, sample in enumerate(tqdm(dataset)):
            image = self.eval_helper.process_image(sample["image"])
            mask = self.eval_helper.process_mask(sample["mask"], resize_size=self.resize_size)

            sae_mask = self.get_seg_mask_pred(image)
            pred_masks = self.eval_helper.resize_mask(sae_mask[:, self.valid_latents], self.resize_size)
            for cls_idx, _ in enumerate((class_names)):
                class_id = cls_idx + 1
                binary_mask = self.eval_helper.get_binary_mask(mask, class_id)
                if binary_mask.sum() == 0:
                    continue
                for j, latent_idx in enumerate((self.valid_latents)):
                    pred_mask = pred_masks[:, j, :, :]
                    if len(np.where(pred_mask.flatten() > 0.5)[0]) == 0:
                        continue
                    scores[i, cls_idx, j] = self.eval_helper.evaluate_auprc(pred_mask, binary_mask)
        scores = scores.mean(axis=0)
        latent_class_mapping = {}
        for cls_idx, class_name in enumerate(class_names):
            max_score_idx = np.argmax(scores[cls_idx])
            latent_idx = self.valid_latents[max_score_idx]
            latent_class_mapping[str(cls_idx)] = {
                "class_name": class_name,
                "latent_idx": latent_idx,
                "concept": self.concept_dict[str(latent_idx)],
            }
        latent_class_mapping = make_json_serializable(latent_class_mapping)
        with open(save_path, "w") as f:
            json.dump(latent_class_mapping, f, indent=4)
        return latent_class_mapping

    def run(self):
        concepts = self.eval_helper.label["name"].tolist()
        images_df = self.eval_helper.images_df

        tot_scores = np.zeros((len(self.valid_latents), len(concepts)), dtype=np.float32)

        for i, latent_idx in tqdm(enumerate(self.valid_latents)):
            scores = np.zeros((len(images_df), len(concepts)), dtype=np.float32)
            for idx in tqdm(range(len(images_df))):
                image, masks, concept_indicator = self.eval_helper.get_image_and_mask(idx)
                sae_mask = self.get_seg_mask_pred(image)
                pred_mask = self.eval_helper.resize_mask(sae_mask[:, latent_idx : latent_idx + 1], self.resize_size)
                for concept_idx, isexists in enumerate(concept_indicator):
                    if isexists:
                        mask = masks[concept_idx]
                        score = self.eval_helper.evaluate_auprc(pred_mask, mask)
                        scores[idx, concept_idx] += score

            avg = scores.mean(axis=0)
            np.save(f"{self.checkpoint_path}/scores_latent_{latent_idx}.npy", avg)
            tot_scores[i] = avg

        np.save(f"{self.checkpoint_path}/segmentation_scores.npy", tot_scores)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate SAE segmentation predictions")
    parser.add_argument("--save_root", type=str, default="./out", help="Root directory to save results")
    parser.add_argument("--dir_name", type=str, default="segmentation", help="Name of the directory to save results")
    parser.add_argument("--sae_path", type=str, required=True, help="Path to the SAE model checkpoint")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model on")
    parser.add_argument("--backbone", type=str, default="openai/clip-vit-large-patch14", help="Backbone model name")
    parser.add_argument("--clip_model_name", type=str, default="openai/clip-vit-large-patch14", help="CLIP model name")
    parser.add_argument(
        "--data_root",
        type=str,
        default="/home/nas4_user/jinhochoi/public_repo/NetDissect/dataset/broden1_224",
        help="Root directory for the dataset",
    )
    args = parser.parse_args()

    evaluator = SAESegmentationEvaluator(
        save_root=args.save_root,
        sae_path=args.sae_path,
        device=args.device,
        backbone=args.backbone,
        clip_model_name=args.clip_model_name,
        dir_name=args.dir_name,
        data_root=args.data_root,
    )

    evaluator.run()
