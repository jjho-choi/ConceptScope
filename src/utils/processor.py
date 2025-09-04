from typing import Dict, Iterator, Optional, Tuple

import h5py
import numpy as np
import torch
from PIL import Image

from src.sae_training.hooked_vit import HookedVisionTransformer


class ImageProcessor:
    @staticmethod
    def process_model_inputs(
        batch: Dict,
        vit: HookedVisionTransformer,
        device: str,
        process_labels: bool = False,
        image_key: str = "image",
    ) -> torch.Tensor:
        """Process input images through the ViT processor."""
        if isinstance(batch[image_key][0], Image.Image):
            images = batch[image_key]
        else:
            images = [Image.open(image_path).convert("RGB") for image_path in batch[image_key]]

        if process_labels:
            labels = [f"A photo of a {label}" for label in batch["label"]]
            return vit.processor(images=images, text=labels, return_tensors="pt", padding=True).to(device)

        return vit.processor(images=images, text="", return_tensors="pt", padding=True).to(device)

    @staticmethod
    def get_sae_activations(
        sae, vit: HookedVisionTransformer, inputs: dict, block_layer, module_name, class_token, get_mean=True
    ) -> torch.Tensor:
        """Extract activations from a specific layer of the vision transformer vitt."""
        hook_location = (block_layer, module_name)

        _, cache = vit.run_with_cache([hook_location], **inputs)
        activations = cache[hook_location]

        batch_size = inputs["pixel_values"].shape[0]
        if activations.shape[0] != batch_size:
            activations = activations.transpose(0, 1)

        if class_token is "only":
            activations = activations[0, :, :]

        _, cache = sae.run_with_cache(activations)
        sae_act = cache["hook_hidden_post"]
        if get_mean:
            sae_act = sae_act.mean(1)

        return sae_act


class BatchIterator:
    """Iterator for batching dataset."""

    @staticmethod
    def get_batches(
        dataset: Dict, batch_size: int, image_key: str = "image", max_sample=None
    ) -> Iterator[Tuple[int, Dict]]:
        """Create batch iterator from dataset."""

        if hasattr(dataset, "num_rows"):
            if dataset.num_rows > 0:
                num_samples = dataset.num_rows
        else:
            num_samples = len(dataset[image_key])

        if max_sample is not None:
            num_samples = min(num_samples, max_sample)

        indices = range(0, num_samples, batch_size)

        for start_idx in indices:
            end_idx = min(start_idx + batch_size, num_samples)
            try:
                batch = dataset[start_idx:end_idx]
            except (TypeError, KeyError):
                batch = {
                    image_key: dataset[image_key][start_idx:end_idx],
                }
            yield start_idx, end_idx, batch


class H5ActivationtWriter:
    """Handles writing data to H5 files."""

    def __init__(self, save_path: str, feature_dim: int):
        """Initialize H5 dataset writer."""
        self.save_path = save_path
        self.feature_dim = feature_dim
        self.num_chunks = 5000

    def check_dataset_exists(self, cls_idx):
        with h5py.File(self.save_path, "a") as f:
            return f"activations_{cls_idx}" in f

    def create_cls_dataset(self, cls_idx, num_samples):

        with h5py.File(self.save_path, "a") as f:
            dataset_name = f"activations_{cls_idx}"
            if dataset_name in f:
                del f[dataset_name]
            f.create_dataset(
                dataset_name,
                shape=(num_samples, self.feature_dim),
                dtype=np.float32,
                chunks=(min(self.num_chunks, num_samples), self.feature_dim),
                compression="gzip",
            )

    def create_dataset(self, num_samples):

        with h5py.File(self.save_path, "a") as f:
            f.create_dataset(
                "activations",
                shape=(num_samples, self.feature_dim),
                dtype=np.float32,
                chunks=(min(self.num_chunks, num_samples), self.feature_dim),
                compression="gzip",
            )

    def write_cls_activation_batch(
        self, start_idx: int, end_idx: int, activations: np.ndarray, cls_idx: Optional[int] = None
    ):
        with h5py.File(self.save_path, "a") as f:
            dataset_name = f"activations_{cls_idx}" if cls_idx is not None else "activations"
            f[dataset_name][start_idx:end_idx] = activations

    def write_cls_activation(self, activations: np.ndarray, cls_idx: Optional[int] = None):
        with h5py.File(self.save_path, "a") as f:
            dataset_name = f"activations_{cls_idx}"
            if dataset_name in f:
                del f[dataset_name]
            f.create_dataset(
                dataset_name,
                data=activations,
                dtype=np.float32,
                chunks=(min(self.num_chunks, len(activations)), self.feature_dim),
                compression="gzip",
            )
