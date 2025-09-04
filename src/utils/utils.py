import json

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from src.utils.processor import ImageProcessor


def get_sae_mask(images, sae, vit, cfg, idx, resize_size=256, device="cpu"):
    inputs = ImageProcessor.process_model_inputs({"image": images}, vit, device)
    sae_act = ImageProcessor.get_sae_activations(
        sae,
        vit,
        inputs,
        cfg.block_layer,
        cfg.module_name,
        cfg.class_token,
        get_mean=False,
    )
    selected_act = sae_act[:, :, idx]
    feature_size = int(np.sqrt(selected_act.shape[1] - 1))

    masks = torch.Tensor(selected_act[:, 1:].reshape(sae_act.shape[0], 1, feature_size, feature_size))
    masks = torch.nn.functional.interpolate(masks, (resize_size, resize_size), mode="bilinear").squeeze(1).cpu().numpy()
    return masks


def apply_sae_mask_to_input(images, masks, resize_size=256, blend_rate=0.0, gamma=0.001, reverse=False):
    masked_images = []
    processed_masks = []
    for i, image in enumerate(images):
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        image_array = np.array(image.resize((resize_size, resize_size)))[..., :3].astype(np.float32)

        mask = (masks[i] - masks[i].min()) / (masks[i].max() - masks[i].min() + 1e-10)
        mask = np.expand_dims(mask, axis=-1)  # Make shape (H, W, 1) to broadcast over RGB

        mask = mask**gamma

        if reverse:
            mask = 1 - mask

        blended = image_array * (blend_rate + (1 - blend_rate) * mask)
        blended = np.clip(blended, 0, 255).astype(np.uint8)

        masked_images.append(Image.fromarray(blended))
        processed_masks.append(mask.squeeze(-1))
    return masked_images, processed_masks


def plot_images(
    images,
    num_cols=5,
    show_plot=True,
    resize_size=256,
    title=None,
):
    num_images = len(images)
    images = [img.resize((resize_size, resize_size)) for img in images]

    num_cols = min(num_cols, 5, num_images)
    num_rows = (num_images + num_cols - 1) // num_cols
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(4.5 * num_cols, 5 * num_rows))

    # Make axes iterable
    if num_rows == 1 and num_cols == 1:
        axes = [axes]
    elif num_rows == 1 or num_cols == 1:
        axes = axes if isinstance(axes, (list, np.ndarray)) else [axes]
    else:
        axes = axes.flatten()

    for i in range(num_images):
        axes[i].imshow(images[i])
        axes[i].axis("off")
    if title is not None:
        fig.suptitle(title, fontsize=16)

    plt.tight_layout()
    if show_plot:
        plt.show()
    plt.close()

    return fig


def get_len_dataset(dataset):
    """Get the length of the dataset."""
    if hasattr(dataset, "num_rows"):
        if dataset.num_rows > 0:
            return dataset.num_rows
    else:
        return len(dataset["image"])


def load_class_names(root, dataset_name, test_dataset=None, target_attribute=None):
    if dataset_name == "imagenet":
        with open(f"{root}/class_names/{dataset_name}_classnames.txt", "r") as f:
            class_names = []
            for i, line in enumerate(f.readlines()):
                class_name = " ".join(line.strip().split(" ")[1:])
                class_names.append(class_name)
    elif dataset_name == "ms_coco":
        class_names = ["person", "bird", "dog", "cat", "car", "airplane"]
    elif dataset_name == "waterbird":
        if target_attribute == "detailed_place":
            return test_dataset.features["label"].names
        class_names = [
            [
                "woodpecker",
                "hummingbird",
                "parrot",
                "owl",
                "finch",
                "blackbird",
                "cormorant",
                "crow",
                "flycatcher",
                "goldfinch",
                "jay",
                "kingfisher",
                "raven",
                "magpie",
                "sparrow",
                "warbler",
                "woodpecker",
                "wren",
                "cardinal",
            ],
            ["duck", "seagull", "pelican", "albatross", "gull", "loon", "mallard", "merganser", "pelican", "puffin"],
        ]
    elif dataset_name == "celeba":
        if target_attribute is None:
            target_attribute = "Blond_Hair"
        attr_name = " ".join(target_attribute.split("_")).lower()
        class_names = [f"without {attr_name}", f"{attr_name}"]
        # class_names = [["dark hair", "brown hair", "red hair", "gray hair"], ["blond hair"]]
    elif dataset_name in ["nico_75", "nico_90", "nico_95"]:
        class_names = [
            [
                "sheep",
                "wolf",
                "lion",
                "fox",
                "elephant",
                "kangaroo",
                "cat",
                "rabbit",
                "dog",
                "monkey",
                "squirrel",
                "tiger",
                "giraffe",
                "horse",
                "bear",
                "cow",
            ],
            ["bird", "owl", "goose", "ostrich"],
            ["flower", "sunflower", "cactus"],
            ["hot air balloon", "airplane", "helicopter"],
            ["sailboat", "ship", "lifeboat"],
            [
                "bicycle",
                "motorcycle",
                "train",
                "bus",
                "scooter",
                "truck",
                "car",
            ],
        ]

    elif dataset_name == "sun397":
        org_class_names = test_dataset.features["label"].names
        class_names = []
        for path in org_class_names:
            processed = [part.replace("_", " ") for part in path.split("/") if part]
            result = ", ".join(processed[1:])
            class_names.append(result)
    else:
        class_names = test_dataset.features["label"].names

    return class_names


def make_json_serializable(obj):
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    else:
        return obj
