import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
from PIL import Image

from src.utils.utils import apply_sae_mask_to_input, get_sae_mask, plot_images


def get_class_sae_activation(
    dataset_name, split, class_idx, checkpoint_name, dir_name="dataset_analysis", save_root="./out"
):
    save_path = f"{save_root}/{dir_name}/{dataset_name}/{checkpoint_name}/{split}_sae_latents.h5"
    with h5py.File(save_path, "r") as hf:
        if f"activations_{class_idx}" in hf:
            activations = hf[f"activations_{class_idx}"][:]
        else:
            raise ValueError(f"No activations found for class index {class_idx} in {save_path}")
    return activations


def plot_ref_images(checkpoint_name, latent_idx, concept_name=None, save_root="./out", show_title=False):
    save_dir = f"{save_root}/checkpoints/{checkpoint_name}/reference_images/"
    file_path = f"{save_dir}{latent_idx}.png"
    image = Image.open(file_path)
    plt.imshow(image)
    title_text = f"latent {latent_idx}"
    if concept_name is not None:
        title_text += f" - concept: {concept_name}"
    if show_title:
        plt.title(title_text)
    plt.axis("off")
    plt.show()
    plt.close()


def plot_concepts(concept_dict, selected_class, target_threshold=0.1, bias_sigma=1, max_concepts=20, save_fig=False):
    data = {
        "Class aligned": [],
        "Mean": [],
        "Concept Type": [],
        "latent_idx": [],
        "latent_name": [],
        "slice_idx": [],
    }

    context_means = []
    len_target = len(concept_dict[str(selected_class)]["target"])
    len_context = max_concepts - len_target
    all_slices = (
        concept_dict[str(selected_class)]["target"] + concept_dict[str(selected_class)]["context"][:len_context]
    )
    for i, slice_info in enumerate(all_slices):
        data["latent_idx"].append(slice_info["latent_idx"])
        data["latent_name"].append(slice_info["latent_name"])
        data["Mean"].append(slice_info["mean_activation"])
        data["Class aligned"].append(slice_info["normalized_alignment_score"])
        data["slice_idx"].append(i)
        if "bias" not in slice_info:
            data["Concept Type"].append("target")
        else:
            context_means.append(slice_info["mean_activation"])
            if slice_info["bias"] == True:
                data["Concept Type"].append("bias")
            else:
                data["Concept Type"].append("context")

    context_means = np.array(context_means)
    bias_threshold = np.mean(context_means) + bias_sigma * np.std(context_means)

    df = pd.DataFrame(data)
    color_map = {"target": "green", "context": "orange", "bias": "red"}
    df["Color"] = df["Concept Type"].map(color_map)

    df = df.sort_values("Class aligned", ascending=True).reset_index(drop=True)

    fig = px.bar(
        df,
        x="slice_idx",
        y="Mean",
        color="Concept Type",
        color_discrete_map=color_map,
        hover_data={
            "Mean": True,
            "latent_name": True,
            "latent_idx": True,
            "Concept Type": True,
            "Class aligned": False,  # hide duplicate
        },
    )

    fig.update_layout(
        xaxis_title="Concepts",
        yaxis_title="Mean Activations",
        xaxis_tickangle=-30,
        xaxis=dict(
            tickmode="array",
            tickvals=df["slice_idx"],
            ticktext=df["latent_name"],
        ),
        yaxis_title_font=dict(size=18, family="Arial", color="black"),  # increase size
        yaxis=dict(tickfont=dict(size=14), nticks=5),  # Limit to 5 y-axis ticks
        template="plotly_white",
        width=1200,
        height=600,
        font=dict(size=14),
    )
    fig.add_shape(
        type="line",
        x0=len_target - 0.5,
        x1=len(df) - 1,
        y0=bias_threshold,
        y1=bias_threshold,
        line=dict(color="red", width=2, dash="dash"),
    )
    fig.show()

    if save_fig:
        save_dir = "output"
        os.makedirs(save_dir, exist_ok=True)
        pio.write_image(fig, f"{save_dir}/classs_{selected_class}_concepts.svg", format="svg")
    return data


def get_high_activating_images(
    activations, idx, class_indices, dataset, resize_size=256, top_k=10, percentile=0, reverse=False
):
    latent_activation = activations[:, idx]

    start_point = int(len(latent_activation) * percentile)

    if reverse:
        latent_activation = -latent_activation
    sort_indices = np.argsort(latent_activation)[::-1][start_point : start_point + top_k]

    images = []
    subset = dataset[class_indices]["image"]
    for idx in sort_indices:
        image = subset[idx]
        if isinstance(image, str):
            image = Image.open(image)
        images.append(image.resize((resize_size, resize_size)))
    return images, sort_indices


def plot_high_low_images(
    latent_idx,
    class_indices,
    class_activation,
    dataset,
    sae,
    vit,
    cfg,
    top_k=10,
    plot_ref=True,
    blend_rate_high=0.5,
    blend_rate_low=0.0,
    gamma=0.5,
    resize_size=256,
    percentile=0,
    device="cuda",
):

    high_images, _ = get_high_activating_images(
        class_activation,
        latent_idx,
        class_indices,
        dataset,
        resize_size=resize_size,
        top_k=top_k,
        percentile=percentile,
        reverse=False,
    )

    low_images, _ = get_high_activating_images(
        class_activation,
        latent_idx,
        class_indices,
        dataset,
        resize_size=resize_size,
        top_k=top_k,
        percentile=percentile,
        reverse=True,
    )

    high_images_masks = get_sae_mask(high_images, sae, vit, cfg, latent_idx, resize_size=resize_size, device=device)
    low_images_masks = get_sae_mask(low_images, sae, vit, cfg, latent_idx, resize_size=resize_size, device=device)

    masked_high_images, _ = apply_sae_mask_to_input(
        high_images, high_images_masks, resize_size=resize_size, blend_rate=blend_rate_high, gamma=gamma, reverse=False
    )
    masked_low_images, _ = apply_sae_mask_to_input(
        low_images, low_images_masks, resize_size=resize_size, blend_rate=blend_rate_low, gamma=gamma, reverse=False
    )

    plot_images(
        high_images, resize_size=resize_size, show_plot=True, title=f"High Activating Images for Latent {latent_idx}"
    )
    plot_images(
        masked_high_images,
        resize_size=resize_size,
        show_plot=True,
        title=f"Masked High Activating Images for Latent {latent_idx}",
    )
    plot_images(
        low_images, resize_size=resize_size, show_plot=True, title=f"Low Activating Images for Latent {latent_idx}"
    )
    plot_images(
        masked_low_images,
        resize_size=resize_size,
        show_plot=True,
        title=f"Masked Low Activating Images for Latent {latent_idx}",
    )
