import argparse

import torch

from src.conceptscope.conceptscope import ConceptScope

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_root", type=str, default="./out", help="Root directory to save results")
    parser.add_argument(
        "--dir_name", type=str, default="dataset_analysis", help="Name of the directory to save results"
    )
    parser.add_argument("--sae_path", type=str, required=True, help="Path to the SAE model checkpoint")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model on")
    parser.add_argument("--backbone", type=str, default="openai/clip-vit-large-patch14", help="Backbone model name")
    parser.add_argument("--clip_model_name", type=str, default="openai/clip-vit-large-patch14", help="CLIP model name")
    parser.add_argument("--dataset_name", type=str, default="imagenet", help="Name of the dataset to analyze")
    parser.add_argument("--split", type=str, default="train", help="Dataset split to analyze (train/val/test)")
    parser.add_argument("--target_attribute", type=str, default=None, help="Target attribute for analysis")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for processing")
    parser.add_argument("--num_samples", type=int, default=256, help="Number of samples to process")
    parser.add_argument(
        "--target_threshold",
        type=float,
        default=0.0,
        help="Threshold for the target attribute to consider a sample as posstive",
    )
    args = parser.parse_args()

    concept_scope = ConceptScope(
        save_root=args.save_root,
        dir_name=args.dir_name,
        sae_path=args.sae_path,
        device=args.device,
        backbone=args.backbone,
        clip_model_name=args.clip_model_name,
        num_samples_for_alignment=args.num_samples,
    )
    with torch.no_grad():
        concept_scope.run(
            dataset_name=args.dataset_name,
            split=args.split,
            target_attribute=args.target_attribute,
            batch_size=args.batch_size,
            target_threshold=args.target_threshold,
        )
