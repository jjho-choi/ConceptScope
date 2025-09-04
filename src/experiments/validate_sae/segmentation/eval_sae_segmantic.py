import argparse
import json
import os

import numpy as np
import torch
from sklearn.metrics import auc, average_precision_score, precision_recall_curve
from tqdm import tqdm

from src.conceptscope.conceptscope import ConceptScope
from src.experiments.validate_sae.segmentation.eval_helper import SegmentationEvalHelper
from src.utils.processor import ImageProcessor
from src.utils.utils import make_json_serializable


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
        self.resize_size = 224

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

    # def evaluate_test_dataset(self, dataset, class_latent_mapping):
    #     scores_dict = {i: [] for i in range(len(class_latent_mapping))}
    #     for i, sample in enumerate(tqdm(dataset)):
    #         image = self.eval_helper.process_image(sample["image"])
    #         mask = self.eval_helper.process_mask(sample["mask"], resize_size=self.resize_size)
    #         sae_mask = self.get_seg_mask_pred(image)
    #         for j in range(len(class_latent_mapping)):
    #             binary_mask = self.eval_helper.get_binary_mask(mask, j + 1)
    #             if binary_mask.sum() == 0:
    #                 continue
    #             latent_idx = class_latent_mapping[str(j)]["latent_idx"]
    #             pred_mask = self.eval_helper.resize_mask(sae_mask[:, latent_idx : latent_idx + 1], self.resize_size)
    #             score = self.eval_helper.evaluate_auprc(pred_mask, binary_mask)
    #             scores_dict[j].append(score)

    #     scores_by_class = np.zeros(len(class_latent_mapping), dtype=np.float32)
    #     for class_idx, scores in scores_dict.items():
    #         scores_by_class[class_idx] = np.mean(scores) if scores else 0.0

    #     return scores_by_class

    def compute_ap_iou_per_class(self, score_map, gt_mask, threshold=0.5):

        C, H, W = score_map.shape
        ap_list = []
        iou_list = []

        for c in range(C):
            y_true = gt_mask[c].ravel()
            y_score = score_map[c].ravel()

            # --- AP ---
            if y_true.sum() == 0:
                ap = np.nan
            else:
                precision, recall, _ = precision_recall_curve(y_true, y_score)
                ap = auc(recall, precision)
            ap_list.append(ap)

            # --- IoU ---
            pred_binary = (score_map[c] >= threshold).astype(np.uint8)
            gt_binary = (gt_mask[c] > 0).astype(np.uint8)

            intersection = np.logical_and(pred_binary, gt_binary).sum()
            union = np.logical_or(pred_binary, gt_binary).sum()

            if union == 0:
                iou = np.nan
            else:
                iou = intersection / union
            iou_list.append(iou)

        return ap_list, iou_list

    def evaluate_test_dataset(self, dataset, class_latent_mapping):
        scores_dict = {i: [] for i in range(len(class_latent_mapping))}
        miou_dict = {i: [] for i in range(len(class_latent_mapping))}
        ap_score = {i: [] for i in range(len(class_latent_mapping))}

        latent_indices = np.array([info["latent_idx"] for info in class_latent_mapping.values()])
        all_ap = [[] for _ in range(len(class_latent_mapping))]
        all_iou = [[] for _ in range(len(class_latent_mapping))]

        for i, sample in enumerate(tqdm(dataset)):
            image = self.eval_helper.process_image(sample["image"])
            mask = self.eval_helper.process_mask(sample["mask"], resize_size=self.resize_size)
            gt_mask = np.zeros((len(class_latent_mapping), self.resize_size, self.resize_size), dtype=np.uint8)
            for j in range(len(class_latent_mapping)):
                gt_binary_mask = self.eval_helper.get_binary_mask(mask, j + 1)
                gt_mask[j] = gt_binary_mask
            sae_mask = self.get_seg_mask_pred(image)
            sae_mask = sae_mask[:, latent_indices]
            sae_mask = torch.softmax(sae_mask, dim=1)
            sae_mask = torch.nn.functional.interpolate(
                sae_mask, size=(self.resize_size, self.resize_size), mode="bilinear", align_corners=False
            )

            ap_list, iou_list = self.compute_ap_iou_per_class(sae_mask[0].cpu().numpy(), gt_mask)

            for c in range(len(class_latent_mapping)):
                if not np.isnan(ap_list[c]):
                    all_ap[c].append(ap_list[c])
                if not np.isnan(iou_list[c]):
                    all_iou[c].append(iou_list[c])

        mean_ap_list = [np.mean(ap) if ap else np.nan for ap in all_ap]
        mean_iou_list = [np.mean(iou) if iou else np.nan for iou in all_iou]

        return mean_ap_list, mean_iou_list, all_ap, all_iou

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

    def run(self, num_train_samples=1000, num_test_samples=None):
        train_dataset = self.eval_helper.get_image_and_mask("training", subsample=num_train_samples)
        test_dataset = self.eval_helper.get_image_and_mask("validation", subsample=num_test_samples)
        class_names = self.eval_helper.get_class_names()
        latent_class_mapping = self.get_latent_mapping(train_dataset, class_names)

        # scores = self.evaluate_test_dataset(test_dataset, latent_class_mapping)
        # save_path = f"{self.save_root}/{self.dir_name}/{self.checkpoint_name}/sae_segmentation_scores.json"
        # os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # self.eval_helper.save_results(scores, class_names, save_path)

        auprc, miou, ap = self.evaluate_test_dataset(test_dataset, latent_class_mapping)
        save_path = f"{self.save_root}/{self.dir_name}/{self.checkpoint_name}/sae_segmentation_scores_rebuttal.json"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        self.eval_helper.save_results({"auprc": auprc, "miou": miou}, class_names, save_path)


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
        default="./data/image_dataset/ade20k/ADEChallengeData2016",
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
