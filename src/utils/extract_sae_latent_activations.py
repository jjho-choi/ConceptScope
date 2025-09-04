import argparse
from pathlib import Path
from typing import Optional, Tuple

import h5py
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from src.utils.image_dataset_loader import ImageDatasetLoader
from src.utils.processor import BatchIterator, ImageProcessor
from src.utils.sae_loader import SAELoader
from src.utils.utils import get_len_dataset


class H5ActivationtWriter:
    """Handles writing data to H5 files."""

    def __init__(self, save_path: Path, feature_dim: int):
        """Initialize H5 dataset writer."""
        self.save_path = save_path
        self.feature_dim = feature_dim
        self.num_chunks = 5000

    def create_cls_dataset(self, cls_idx, num_samples):

        with h5py.File(self.save_path, "w") as f:
            if cls_idx is not None:
                dataset_name = f"activations_{cls_idx}"
            else:
                dataset_name = "activations"
            f.create_dataset(
                dataset_name,
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


def extract_sae_latent_activations(
    dataset,
    batch_size: int,
    vit,
    sae,
    cfg,
    writer,
    device: str,
    image_key: str = "image",
    cls_idx: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Process entire dataset in batches."""

    len_dataset = get_len_dataset(dataset)
    total_batches = (len_dataset + batch_size - 1) // batch_size
    batch_iterator = BatchIterator.get_batches(dataset, batch_size, image_key)

    for start_idx, end_idx, batch in tqdm(batch_iterator, desc="Processing batches", total=total_batches):
        batch_inputs = ImageProcessor.process_model_inputs(batch, vit, device, image_key=image_key)

        sae_latents = ImageProcessor.get_sae_activations(
            sae, vit, batch_inputs, cfg.block_layer, cfg.module_name, cfg.class_token
        )

        writer.write_batch(start_idx, end_idx, sae_latents.cpu().numpy(), cls_idx=cls_idx)


def main(
    sae_path: str,
    device: str,
    dataset_name: str,
    root_dir: str,
    save_name: str,
    backbone: str,
    seed: int = 1,
    batch_size: int = 8,
    split: str = "train",
    image_key: str = "image",
    cls_idx: Optional[int] = None,
):
    """Main function to process dataset and save results."""
    dataset = ImageDatasetLoader.load_dataset(dataset_name, seed=seed, split=split)
    sae, vit, cfg = SAELoader.get_sae_vit_cfg(sae_path, device, backbone)

    sae_name = sae_path.split("/")[-2]
    save_dir = Path(root_dir) / save_name / sae_name / dataset_name
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"{split}_sae_latents.h5"
    h5_writer = H5DatasetWriter(save_path, cfg.d_sae, len(dataset[image_key]), cls_idx=cls_idx)

    extract_sae_latent_activations(
        dataset=dataset,
        vit=vit,
        sae=sae,
        cfg=cfg,
        writer=h5_writer,
        device=device,
        batch_size=batch_size,
        image_key=image_key,
        cls_idx=cls_idx,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process ViT SAE images and save feature data")
    parser.add_argument("--root_dir", type=str, default=".", help="Root directory")
    parser.add_argument("--dataset_name", type=str, default="imagenet")
    parser.add_argument("--sae_path", type=str, required=True, help="SAE ckpt path (ends with xxx.pt)")
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size to compute model activations and sae features"
    )
    parser.add_argument("--backbone", type=str, default="openai/clip-vit-base-patch16")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    main(
        sae_path=args.sae_path,
        device=args.device,
        dataset_name=args.dataset_name,
        backbone=args.backbone,
        root_dir=args.root_dir,
        save_name="out/feature_data",
        batch_size=args.batch_size,
        split=args.split,
    )
