from typing import Tuple

import torch
from transformers import CLIPModel, CLIPProcessor

from src.sae_training.config import Config
from src.sae_training.hooked_vit import HookedVisionTransformer
from src.sae_training.sparse_autoencoder import SparseAutoencoder


class SAELoader:
    """A utility class for loading SAE and ViT models."""

    @staticmethod
    def load_sae(sae_path: str, device: str) -> Tuple[SparseAutoencoder, Config]:

        checkpoint = torch.load(sae_path, map_location="cpu", weights_only=False)

        try:
            cfg = Config(checkpoint["cfg"])
        except:
            cfg = Config(checkpoint["config"])

        sae = SparseAutoencoder(cfg, device)
        sae.load_state_dict(checkpoint["state_dict"])
        sae.eval().to(device)

        return sae, cfg

    @staticmethod
    def get_sae_vit_cfg(
        sae_path: str,
        device: str,
        backbone: str,
    ) -> Tuple[SparseAutoencoder, HookedVisionTransformer, Config]:
        """
        Load both SAE and ViT models.
        """
        sae, cfg = SAELoader.load_sae(sae_path, device)
        model = CLIPModel.from_pretrained(backbone)
        processor = CLIPProcessor.from_pretrained(backbone)
        vit = HookedVisionTransformer(model, processor, device=device)

        return sae, vit, cfg
