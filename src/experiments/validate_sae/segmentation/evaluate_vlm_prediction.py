import argparse

import numpy as np
import torch
from tqdm import tqdm

from src.experiments.validate_sae.segmentation.eval_helper import SegmentationEvalHelper
from src.experiments.validate_sae.vlm_baseline.vlm_module import (
    BLIP2Module,
    LlavaNextModule,
)


class VLMSegmentationEvaluator:
    def __init__(self, save_root, dir_name, model_name, device, data_root):
        self.save_root = save_root
        self.dir_name = dir_name
        self.model_name = model_name
        self.device = device
        self.data_root = data_root
        if model_name == "blip2":
            self.vlm_module = BLIP2Module(device=device)
        elif model_name == "llava_next":
            self.vlm_module = LlavaNextModule(device=device)
        self.eval_helper = SegmentationEvalHelper(data_root)
        self.resize_size = 224

    def evaluate_test_dataset(self, test_dataset, class_names):
        scores_dict = {i: [] for i in range(len(class_names))}
        for i, sample in enumerate(tqdm(test_dataset)):
            image = self.eval_helper.process_image(sample["image"])
            mask = self.eval_helper.process_mask(sample["mask"], resize_size=self.resize_size)
            for j, class_name in enumerate(class_names):
                binary_mask = self.eval_helper.get_binary_mask(mask, j + 1)
                if binary_mask.sum() == 0:
                    continue
                vlm_mask = self.vlm_module.get_segmentation_mask(image, class_name)
                vlm_mask = self.eval_helper.resize_mask(vlm_mask, resize_size=self.resize_size)
                score = self.eval_helper.evaluate_auprc(vlm_mask, binary_mask)
                scores_dict[j].append(score)

        scores_by_class = np.zeros(len(class_names), dtype=np.float32)
        for class_idx, scores in scores_dict.items():
            scores_by_class[class_idx] = np.mean(scores) if scores else 0.0

        return scores_by_class

    def run(self):
        test_dataset = self.eval_helper.get_image_and_mask("validation")
        class_names = self.eval_helper.get_class_names()
        scores = self.evaluate_test_dataset(test_dataset, class_names)
        save_path = f"{self.save_root}/{self.dir_name}/{self.model_name}_segmentation_scores.npy"
        self.eval_helper.save_results(scores, class_names, save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate VLM predictions for attribute prediction")
    parser.add_argument("--save_root", type=str, default="./out", help="Root directory to save results")
    parser.add_argument("--dir_name", type=str, default="segmentation", help="Directory name for saving results")
    parser.add_argument("--model_name", type=str, choices=["blip2", "llava_next"], required=True, help="VLM model name")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model on")
    parser.add_argument("--num_iter", type=int, default=5, help="Number of iterations for evaluation")
    parser.add_argument(
        "--data_root",
        type=str,
        default="./data/image_dataset/ade20k/ADEChallengeData2016",
        help="Root directory for the dataset",
    )
    args = parser.parse_args()

    evaluator = VLMSegmentationEvaluator(
        save_root=args.save_root,
        dir_name=args.dir_name,
        model_name=args.model_name,
        device=args.device,
        data_root=args.data_root,
    )

    with torch.no_grad():
        evaluator.run()
