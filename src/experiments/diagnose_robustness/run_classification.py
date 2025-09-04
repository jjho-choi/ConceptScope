import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torchvision import transforms
from tqdm import tqdm

from src.experiments.diagnose_robustness.dataloader import create_dataloaders
from src.experiments.diagnose_robustness.trainer import MultiClassClassifierTrainer
from src.utils.image_dataset_loader import ImageDatasetLoader


def get_model(backbone, pretrained="IMAGENET1K_V1"):
    if backbone == "resnet18":
        model = models.resnet18(weights=pretrained)
    elif backbone == "resnet50":
        model = models.resnet50(weights=pretrained)
    elif backbone == "resnet34":
        model = models.resnet34(weights=pretrained)
    elif backbone == "resnet101":
        model = models.resnet101(weights=pretrained)
    elif backbone == "resnext50":
        model = models.resnext50_32x4d(weights=pretrained)
    elif backbone == "resnext101":
        model = models.resnext101_32x8d(weights=pretrained)
    elif backbone == "resnet152":
        model = models.resnet152(weights=pretrained)
    elif backbone == "vit_b_16":
        model = models.vit_b_16(weights=pretrained)
    elif backbone == "vit_b_32":
        model = models.vit_b_32(weights=pretrained)
    elif backbone == "vit_l_16":
        model = models.vit_l_16(weights=pretrained)
    elif backbone == "vit_l_32":
        model = models.vit_l_32(weights=pretrained)
    elif backbone == "convnext_tiny":
        model = models.convnext_tiny(weights=pretrained)
    elif backbone == "convnext_base":
        model = models.convnext_base(weights=pretrained)
    elif backbone == "convnext_small":
        model = models.convnext_small(weights=pretrained)
    elif backbone == "convnext_large":
        model = models.convnext_large(weights=pretrained)
    elif backbone == "alexnet":
        model = models.alexnet(weights=pretrained)
    elif backbone == "vgg11":
        model = models.vgg11(weights=pretrained)
    elif backbone == "vgg16":
        model = models.vgg16(weights=pretrained)
    elif backbone == "vgg19":
        model = models.vgg19(weights=pretrained)
    elif backbone == "efficientnet_b0":
        model = models.efficientnet_b0(weights=pretrained)
    elif backbone == "efficientnet_b1":
        model = models.efficientnet_b1(weights=pretrained)
    elif backbone == "efficientnet_b2":
        model = models.efficientnet_b2(weights=pretrained)
    elif backbone == "efficientnet_b3":
        model = models.efficientnet_b3(weights=pretrained)
    elif backbone == "efficientnet_b4":
        model = models.efficientnet_b4(weights=pretrained)
    elif backbone == "efficientnet_b5":
        model = models.efficientnet_b5(weights=pretrained)
    elif backbone == "efficientnet_b6":
        model = models.efficientnet_b6(weights=pretrained)
    elif backbone == "efficientnet_b7":
        model = models.efficientnet_b7(weights=pretrained)
    elif backbone == "densenet121":
        model = models.densenet121(weights=pretrained)
    elif backbone == "densenet169":
        model = models.densenet169(weights=pretrained)
    elif backbone == "densenet201":
        model = models.densenet201(weights=pretrained)
    elif backbone == "wide_resnet50_2":
        model = models.wide_resnet50_2(weights=pretrained)
    elif backbone == "wide_resnet101_2":
        model = models.wide_resnet101_2(weights=pretrained)
    elif backbone == "swin_b":
        model = models.swin_b(weights=pretrained)
    elif backbone == "swin_t":
        model = models.swin_t(weights=pretrained)
    elif backbone == "swin_s":
        model = models.swin_s(weights=pretrained)

    return model


def get_pred_results(
    save_root,
    model_name,
    dataset_name,
    dataloader,
    device,
    split="test",
    pretrained=False,
):

    save_dir = f"{save_root}/{dataset_name}/{model_name}.csv"
    if os.path.exists(save_dir):
        return pd.read_csv(save_dir)
    os.makedirs(os.path.dirname(save_dir), exist_ok=True)

    model = get_model(model_name, pretrained=pretrained).to(device)

    trainer = MultiClassClassifierTrainer(model, nn.CrossEntropyLoss(), None, device, dataloader)
    eval_gt, eval_pred, eval_logit, eval_metrics = trainer.evaluate(split=split)

    out_dict = {"pred_label": eval_pred, "gt_label": eval_gt}
    df_test_preds = pd.DataFrame(out_dict)
    df_test_preds.to_csv(save_dir, index=False)

    return df_test_preds


def main(
    save_root,
    model_name,
    dataset_name,
    split="test",
    device="cuda",
    pretrained=False,
    target_attribute=None,
    batch_size=64,
):

    dataloader = create_dataloaders(
        batch_size=batch_size,
        dataset_name=dataset_name,
        attribute=target_attribute,
        test_only=True,
    )

    pred_results = get_pred_results(
        save_root,
        model_name,
        dataset_name,
        dataloader,
        device,
        split=split,
        pretrained=pretrained,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a ResNet-50 for single-attribute classification on CelebA.")
    parser.add_argument("--model_name", type=str, default="base")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--target_attribute", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training and evaluation.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train the model.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for the optimizer.")
    parser.add_argument(
        "--device", type=str, default="cuda:5", help="Device to use for training (e.g., 'cpu' or 'cuda')."
    )
    parser.add_argument(
        "--save_root",
        type=str,
        default="./out/classification",
        help="Directory where to save model checkpoints and logs.",
    )
    parser.add_argument("--dataset_name", type=str, default="imagenet", help="Name of the dataset to use.")
    parser.add_argument("--pretrained", type=str, default="IMAGENET1K_V1", help="Pretrained model to use.")
    parser.add_argument("--backbone", type=str, default="resnet50", help="Backbone architecture of the model.")
    parser.add_argument("--num_classes", type=int, default=1000, help="Number of classes for the classifier.")
    args = parser.parse_args()

    model_names = [
        "alexnet",
        "vgg11",
        "vgg16",
        "vgg19",
        "resnet18",
        "resnet34",
        "resnet50",
        "resnet101",
        "resnet152",
        "resnext50",
        "resnext101",
        "vit_b_16",
        "vit_b_32",
        "vit_l_16",
        "vit_l_32",
        "convnext_tiny",
        "convnext_base",
        "convnext_small",
        "convnext_large",
        "efficientnet_b0",
        "efficientnet_b1",
        "efficientnet_b2",
        "efficientnet_b3",
        "efficientnet_b4",
        "efficientnet_b5",
        "efficientnet_b6",
        "efficientnet_b7",
        "densenet121",
        "densenet169",
        "densenet201",
        "wide_resnet50_2",
        "wide_resnet101_2",
        "swin_b",
        "swin_t",
        "swin_s",
    ]
    for dataset_name in ["imagenet", "imagenet-v2", "imagenet-sketch"]:
        args.dataset_name = dataset_name
        for model_name in model_names:
            args.model_name = model_name
            print(f"Testing model: {model_name}")

            try:
                main(
                    args.save_root,
                    model_name,
                    dataset_name,
                    split=args.split,
                    device=args.device,
                    pretrained=args.pretrained,
                    target_attribute=args.target_attribute,
                    batch_size=args.batch_size,
                )
            except Exception as e:
                print(f"Error occurred while testing model {model_name} on dataset {dataset_name}: {e}")
                continue
