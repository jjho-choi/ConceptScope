from collections import defaultdict

import numpy as np


def subsample_dataset_by_class(dataset, num_samples=100):

    class_to_indices = defaultdict(list)

    for i, label in enumerate(dataset["label"]):
        class_to_indices[label].append(i)

    subset_indices = []
    for label, indices in class_to_indices.items():
        if len(indices) >= num_samples:
            subset_indices.extend(indices[:num_samples])
        else:
            print(f"Warning: Class {label} has only {len(indices)} examples.")
            subset_indices.extend(indices)

    subsampled_dataset = dataset.select(subset_indices)
    return subsampled_dataset, subset_indices


class AttrPredictionEvaluator:
    def __init__(self, seed=42):
        self.seed = seed

    def subsample_dataset(self, dataset, num_samples=1000):
        total_size = len(dataset)
        indices = np.linspace(0, total_size - 1, num=num_samples, dtype=int).tolist()
        subsampled_dataset = dataset.select(indices)
        return subsampled_dataset, indices

    def subsample_dataset_by_class(self, dataset, num_samples=100):
        np.random.seed(self.seed)
        class_to_indices = defaultdict(list)

        for i, label in enumerate(dataset["label"]):
            class_to_indices[label].append(i)

        subset_indices = []
        for label, indices in class_to_indices.items():
            num_samples = min(num_samples, len(indices))

            sampled_indices = np.random.choice(indices, size=num_samples, replace=False).tolist()
            if len(indices) >= num_samples:
                subset_indices.extend(sampled_indices)
            else:
                print(f"Warning: Class {label} has only {len(indices)} examples.")
                subset_indices.extend(indices)

        subsampled_dataset = dataset.select(subset_indices)
        return subsampled_dataset, subset_indices

    def get_attr_list(self, dataset_name, dataset):

        if dataset_name == "celeba":
            attr_list = [
                "Bald",
                "Bangs",
                "Black_Hair",
                "Blond_Hair",
                "Brown_Hair",
                "Eyeglasses",
                "Gray_Hair",
                "Male",
                "Smiling",
                "Wavy_Hair",
                "Straight_Hair",
                "Wearing_Earrings",
                "Wearing_Hat",
                "Wearing_Necklace",
                "Wearing_Necktie",
                "Wearing_Earrings",
                "Young",
            ]
        else:
            attr_list = dataset.features["label"].names

        return attr_list

    def get_binary_target(self, dataset, dataset_name, cls_idx):
        if dataset_name == "celeba":
            return np.array(dataset["label"])
        target_arr = np.array(dataset["label"]).astype(int)
        binary_target = target_arr == cls_idx
        return binary_target
