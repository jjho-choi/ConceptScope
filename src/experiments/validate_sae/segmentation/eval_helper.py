import json
import os
import random
from glob import glob

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import auc, precision_recall_curve

from src.utils.utils import make_json_serializable


class SegmentationEvalHelper:
    def __init__(self, data_root):
        self.data_root = data_root

    def get_class_names(self):
        class_names = []
        filepath = f"{self.data_root}/objectInfo150.txt"
        with open(filepath, "r") as f:
            for line in f:
                if line.startswith("Idx") or not line.strip():
                    continue
                parts = line.strip().split("\t")
                if len(parts) >= 5:
                    name = parts[4]
                    class_names.append(name)
        return class_names

    def get_image_and_mask(self, split, subsample=None, seed=42):
        image_folder = f"{self.data_root}/images/{split}"
        mask_folder = f"{self.data_root}/annotations/{split}"

        image_files = sorted(glob(f"{image_folder}/*"))

        if subsample is not None:
            random.seed(seed)
            image_files = random.sample(image_files, subsample)

        mask_files = [
            os.path.join(mask_folder, os.path.basename(f)) for f in image_files
        ]

        out = []
        for image_file, mask_file in zip(image_files, mask_files):
            out.append({"image": image_file, "mask": mask_file})
        return out

    def process_image(self, image_path, resize_size=256):
        image = Image.open(image_path).convert("RGB").resize((resize_size, resize_size))
        return image

    def process_mask(self, mask_path, resize_size=256):
        mask = Image.open(mask_path.replace("jpg", "png"))
        mask = mask.resize(
            (resize_size, resize_size), Image.NEAREST
        )  # Nearest to keep class indices intact
        return np.array(mask, dtype=np.uint8)

    def get_binary_mask(self, mask, class_index):
        binary_mask = np.zeros_like(mask, dtype=np.uint8)
        binary_mask[mask == class_index] = 1
        return binary_mask

    def resize_mask(self, mask, resize_size=256):
        mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-10)
        mask = (
            torch.nn.functional.interpolate(
                mask, (resize_size, resize_size), mode="bilinear", align_corners=False
            )
            .cpu()
            .numpy()
        )
        return mask

    def evaluate_auprc(self, pred_mask, gt_mask):
        precision, recall, _ = precision_recall_curve(
            gt_mask.flatten(), pred_mask.flatten()
        )
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
