import argparse

import torch

from src.construct_concept_dict.concept_constructor import ConceptDictConstructor


def main(
    dataset_name: str,
    sae_path: str,
    backbone: str,
    seed: int = 1,
    split: str = "train",
    device: str = "cuda",
    batch_size: int = 64,
    top_k: int = 5,
    resize_size: int = 256,
    use_gpt: bool = False,
    save_features: bool = True,
):

    concept_dict_constructor = ConceptDictConstructor(
        save_root="./out",
        sae_path=sae_path,
        device=device,
        backbone=backbone,
        dataset_name=dataset_name,
        seed=seed,
        split=split,
    )
    with torch.no_grad():
        concept_dict_constructor.run(
            batch_size=batch_size, top_k=top_k, resize_size=resize_size, use_gpt=use_gpt, save_features=save_features
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--dataset_name", type=str, default="imagenet", help="Name of the dataset to process")
    parser.add_argument("--sae_path", type=str, required=True, help="Path to the SAE model")
    parser.add_argument("--backbone", type=str, default="openai/clip-vit-large-patch14", help="Backbone model type")
    parser.add_argument("--seed", type=int, default=1, help="Random seed for reproducibility")
    parser.add_argument("--split", type=str, default="train", help="Dataset split to process (train/val/test)")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model on (cpu/cuda)")
    parser.add_argument("--use_gpt", action="store_true", help="Use GPT for naming latents")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for processing")
    parser.add_argument(
        "--save_features", action="store_true", help="Whether to save the features extracted from the dataset"
    )
    args = parser.parse_args()

    main(
        dataset_name=args.dataset_name,
        sae_path=args.sae_path,
        backbone=args.backbone,
        seed=args.seed,
        split=args.split,
        device=args.device,
        use_gpt=args.use_gpt,
        batch_size=args.batch_size,
        save_features=args.save_features,
    )
