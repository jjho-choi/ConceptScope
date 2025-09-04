from glob import glob

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from src.utils.image_dataset_loader import ImageDatasetLoader


def cal_z(train_mean_var_activations, label, activation, selected_class, latent_indices):
    train_means = train_mean_var_activations[0][selected_class][latent_indices]
    train_stds = train_mean_var_activations[1][selected_class][latent_indices]

    test_class_indices = np.where(label == selected_class)[0]
    test_class_activation = activation[test_class_indices, :][:, latent_indices]

    z_scores = (test_class_activation - train_means) / train_stds
    z_scores = np.clip(z_scores, -3, 3)
    z_scores = z_scores.max(-1)
    return z_scores, test_class_indices


def get_test_stat(
    test_activation,
    test_label,
    num_classes,
    train_mean_var_activation,
    categorization_dict,
    pred_results,
    threshold=0.0,
    verbose=False,
):

    all_z_target = np.zeros(len(test_activation))
    all_z_bias = np.zeros(len(test_activation))
    for selected_class in range(num_classes):
        target_latent_indices = []
        for slice_info in categorization_dict[str(selected_class)]["target"]:
            latent_idx = slice_info["latent_idx"]
            target_latent_indices.append(latent_idx)
        target_latent_indices = np.array(target_latent_indices)
        if len(target_latent_indices) != 0:
            z_scores, test_class_indices = cal_z(
                train_mean_var_activation,
                test_label,
                test_activation,
                selected_class,
                target_latent_indices,
            )
            all_z_target[test_class_indices] = z_scores

        bias_latent_indices = []
        for slice_info in categorization_dict[str(selected_class)]["context"]:
            if slice_info["bias"] is False:
                continue
            latent_idx = slice_info["latent_idx"]
            bias_latent_indices.append(latent_idx)
        bias_latent_indices = np.array(bias_latent_indices)
        if len(bias_latent_indices) != 0:
            z_scores, test_class_indices = cal_z(
                train_mean_var_activation,
                test_label,
                test_activation,
                selected_class,
                bias_latent_indices,
            )
            all_z_bias[test_class_indices] = z_scores

    high_threshold = threshold
    low_threshold = -high_threshold

    prototype_examples = np.where((all_z_target > high_threshold) & (all_z_bias > high_threshold))[0]
    confounded_examples = np.where((all_z_target > high_threshold) & (all_z_bias <= low_threshold))[0]
    shortcut_example = np.where((all_z_target <= low_threshold) & (all_z_bias > high_threshold))[0]
    ambiguous_example = np.where((all_z_target <= low_threshold) & (all_z_bias <= low_threshold))[0]

    ratio_prototype = len(prototype_examples) / len(all_z_target)
    ratio_confounded = len(confounded_examples) / len(all_z_target)
    ratio_shortcut = len(shortcut_example) / len(all_z_target)
    ratio_ambiguous = len(ambiguous_example) / len(all_z_target)

    prototype_acc = accuracy_score(
        pred_results["gt_label"][prototype_examples], pred_results["pred_label"][prototype_examples]
    )
    confounded_acc = accuracy_score(
        pred_results["gt_label"][confounded_examples], pred_results["pred_label"][confounded_examples]
    )
    shortcut_acc = accuracy_score(
        pred_results["gt_label"][shortcut_example], pred_results["pred_label"][shortcut_example]
    )
    ambiguous_acc = accuracy_score(
        pred_results["gt_label"][ambiguous_example], pred_results["pred_label"][ambiguous_example]
    )
    all_acc = accuracy_score(pred_results["gt_label"], pred_results["pred_label"])

    if verbose:
        print(
            f"Prototype examples: {len(prototype_examples)}, Confounded examples: {len(confounded_examples)}, Shortcut examples: {len(shortcut_example)}, Ambiguous examples: {len(ambiguous_example)}"
        )

        print(
            f"Prototype ratio: {ratio_prototype:.4f}, Confounded ratio: {ratio_confounded:.4f}, Shortcut ratio: {ratio_shortcut:.4f}, Ambiguous ratio: {ratio_ambiguous:.4f}"
        )

        print(
            f"Prototype acc: {prototype_acc:.4f}, Confounded acc: {confounded_acc:.4f}, Shortcut acc: {shortcut_acc:.4f}, Ambiguous acc: {ambiguous_acc:.4f}"
        )
        print(f"All acc: {all_acc:.4f}")

    return {
        "avg_acc": all_acc,
        "prototype_acc": prototype_acc,
        "confounded_acc": confounded_acc,
        "shortcut_acc": shortcut_acc,
        "ambiguous_acc": ambiguous_acc,
        "prototype_ratio": ratio_prototype,
        "confounded_ratio": ratio_confounded,
        "shortcut_ratio": ratio_shortcut,
        "ambiguous_ratio": ratio_ambiguous,
        "prototype_examples": prototype_examples,
        "confounded_examples": confounded_examples,
        "shortcut_examples": shortcut_example,
        "ambiguous_examples": ambiguous_example,
        "prototype_indices": prototype_examples,
        "confounded_indices": confounded_examples,
        "shortcut_indices": shortcut_example,
        "ambiguous_indices": ambiguous_example,
        "all_z_target": all_z_target,
        "all_z_bias": all_z_bias,
    }


def get_scores(
    save_dir,
    test_activation,
    test_label,
    num_classes,
    train_mean_var_activation,
    categorization_dict,
    threshold=0.0,
    verbose=False,
):

    pred_results = glob(save_dir)

    base_acc = np.zeros(len(pred_results))
    prototype_acc = np.zeros(len(pred_results))
    confounded_acc = np.zeros(len(pred_results))
    shortcut_acc = np.zeros(len(pred_results))
    ambiguous_acc = np.zeros(len(pred_results))

    for i, pred_result in tqdm(enumerate(pred_results), desc="Processing prediction results"):
        pred_result = pd.read_csv(pred_result)
        out = get_test_stat(
            test_activation,
            test_label,
            num_classes,
            train_mean_var_activation,
            categorization_dict,
            pred_result,
            threshold=threshold,
            verbose=verbose,
        )
        base_acc[i] = out["avg_acc"]
        prototype_acc[i] = out["prototype_acc"]
        confounded_acc[i] = out["confounded_acc"]
        shortcut_acc[i] = out["shortcut_acc"]
        ambiguous_acc[i] = out["ambiguous_acc"]
    return {
        "base_acc": base_acc,
        "prototype_acc": prototype_acc,
        "confounded_acc": confounded_acc,
        "shortcut_acc": shortcut_acc,
        "ambiguous_acc": ambiguous_acc,
        "all_z_target": out["all_z_target"],
        "all_z_bias": out["all_z_bias"],
        "prototype_indices": out["prototype_indices"],
        "confounded_indices": out["confounded_indices"],
        "shortcut_indices": out["shortcut_indices"],
        "ambiguous_indices": out["ambiguous_indices"],
    }


def add_scatter_and_regression(fig, x, y, name, color="blue", opacity=0.6, show_legend=False):
    fig.update_layout(
        title="Model Accuracy by Group with Regression Lines",
        xaxis_title="Base Accuracy",
        yaxis_title="Group Accuracy",
        template="plotly_white",
    )
    fig.update_layout(
        xaxis_title_font=dict(size=16),  # X-axis label font size
        yaxis_title_font=dict(size=16),  # Y-axis label font size
        font=dict(size=14),  # General tick and hover font size
        width=800,
        height=600,
        showlegend=show_legend,
        legend=dict(
            title="Group", orientation="h", yanchor="bottom", y=1.1, xanchor="center", x=0.5  # horizontal orientation
        ),
    )

    fig.add_trace(go.Scatter(x=x, y=y, mode="markers", name=name, marker=dict(color=color, opacity=opacity)))

    coeffs = np.polyfit(x, y, 1)
    x_line = np.linspace(min(x), max(x), 100)
    y_line = coeffs[0] * x_line + coeffs[1]

    fig.add_trace(
        go.Scatter(
            x=x_line,
            y=y_line,
            mode="lines",
            name=f"{name} (regression)",
            line=dict(color="black", dash="dash"),
            opacity=0.5,
        )
    )

    identity_min = min(min(x), min(y))
    identity_max = max(max(x), max(y))
    identity_line = np.linspace(identity_min, identity_max, 100)

    fig.add_trace(
        go.Scatter(
            x=identity_line,
            y=identity_line,
            mode="lines",
            name="y = x",
            line=dict(color="gray", dash="dot"),
            opacity=0.5,
        )
    )
    fig.update_traces(marker=dict(size=10), selector=dict(mode="markers"))


def load_all(dataset_name, split, conceptscope, train_datast_name="imagenet"):

    dataset = ImageDatasetLoader.load_dataset(dataset_name=dataset_name, split=split)
    sae_activations = conceptscope.get_sae_latent_activations(dataset_name, split, dataset)
    mean_var_activations = conceptscope.get_train_mean_var_activations(train_datast_name, "train")
    concept_dict = conceptscope.get_concept_dict(train_datast_name, "train")
    labels = np.array(dataset["label"])
    num_classes = len(np.unique(labels))

    save_dir = f"../../../out/classification/results/{dataset_name}/*.csv"
    out = get_scores(save_dir, sae_activations, labels, num_classes, mean_var_activations, concept_dict)

    out["dataset"] = dataset
    out["sae_activations"] = sae_activations
    out["mean_var_activations"] = mean_var_activations
    out["concept_dict"] = concept_dict
    out["labels"] = labels
    out["num_classes"] = num_classes

    return out
