# %%

import os

import numpy as np

os.environ["HOME"] = "/home/nas4_user/jinhochoi"
import sys

from sklearn.metrics import auc, precision_recall_curve

sys.path.append("/home/nas4_user/jinhochoi/repo/ConceptScope")

import cv2
import matplotlib.pyplot as plt
from PIL import Image

from src.experiments.validate_sae.segmentation.evaluate_vlm_prediction import (
    VLMSegmentationEvaluator,
)


# %%
def gradcam_style_overlay(image, attn, colormap="jet", alpha=0.5):
    """
    Overlays an attention map on an image like Grad-CAM.

    Args:
        image_path (str): Path to the input image.
        attention_map (torch.Tensor): Tensor of shape (H, W) or (1, H, W).
        colormap (str): Matplotlib colormap name (e.g., 'jet', 'inferno').
        alpha (float): Transparency of the heatmap overlay.
    """
    image_np = np.array(image)

    attn = (attn - attn.min()) / (attn.max() - attn.min() + 1e-8)
    attn_resized = cv2.resize(attn, (image_np.shape[1], image_np.shape[0]))

    # Apply colormap
    cmap = plt.get_cmap(colormap)
    attn_colored = cmap(attn_resized)[:, :, :3]  # drop alpha

    # Convert to 0-255 for overlay
    attn_colored_uint8 = (attn_colored * 255).astype(np.uint8)
    overlay = cv2.addWeighted(image_np, 1 - alpha, attn_colored_uint8, alpha, 0)

    # Plot
    plt.figure(figsize=(6, 6))
    plt.imshow(overlay)
    plt.title("Grad-CAM Style SAE Attention")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


# %%

save_root = "/home/nas4_user/jinhochoi/repo/ConceptScope/out"
dir_name = "segmenmtation"
model_name = "llava_next"
device = "cuda:7"
data_root = "/home/nas4_user/jinhochoi/repo/ConceptScope/data/image_dataset/ade20k/ADEChallengeData2016"


evaluator = VLMSegmentationEvaluator(
    save_root=save_root,
    dir_name=dir_name,
    model_name=model_name,
    device=device,
    data_root=data_root,
)

# %%


test_dataset = evaluator.eval_helper.get_image_and_mask("validation")
class_names = evaluator.eval_helper.get_class_names()


# %%

scores = np.zeros((len(test_dataset), len(class_names)), dtype=np.float32)
for i, sample in enumerate((test_dataset)):
    image = evaluator.eval_helper.process_image(sample["image"])
    mask = evaluator.eval_helper.process_mask(sample["mask"], resize_size=224)
    for j, class_name in enumerate(class_names):
        binary_mask = evaluator.eval_helper.get_binary_mask(mask, j + 1)
        if binary_mask.sum() == 0:
            continue
        vlm_mask = evaluator.vlm_module.get_segmentation_mask(image, class_name)
        vlm_mask = evaluator.eval_helper.resize_mask(vlm_mask, resize_size=224)
        break
    break
# %%
for i, sample in enumerate(test_dataset):
    mask = evaluator.eval_helper.process_mask(sample["mask"])
    binary_mask = evaluator.eval_helper.get_binary_mask(mask, class_idx + 1)
    if binary_mask.sum() > 0:
        print(i)
        continue

# %%

sample = test_dataset[490]
sample = test_dataset[78]

image = evaluator.eval_helper.process_image(sample["image"])
mask = evaluator.eval_helper.process_mask(sample["mask"], resize_size=224)

class_idx = 98
class_name = class_names[class_idx]
binary_mask = evaluator.eval_helper.get_binary_mask(mask, class_idx + 1)
vlm_mask = evaluator.vlm_module.get_segmentation_mask(image, class_name)
vlm_mask = evaluator.eval_helper.resize_mask(vlm_mask, resize_size=224)
print(class_name)


precision, recall, _ = precision_recall_curve(binary_mask.flatten(), vlm_mask.flatten())
score = auc(recall, precision)
print(score)


fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Original image
axes[0].imshow(image)
axes[0].set_title("Original Image")
axes[0].axis("off")

# Ground-truth mask overlay (COCO-style: transparent red)
axes[1].imshow(image)
axes[1].imshow(binary_mask, cmap="Reds", alpha=0.7)
axes[1].set_title("GT Mask Overlay")
axes[1].axis("off")

# Attention map overlay (e.g., using viridis colormap)

vlm_mask = vlm_mask.squeeze()
axes[2].imshow(image)
axes[2].imshow(vlm_mask, cmap="viridis", alpha=0.7)
axes[2].set_title("SAE Attention Overlay")
axes[2].axis("off")

plt.tight_layout()
plt.show()


# %%

import torch

conversation = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": class_name},
            {"type": "image"},
        ],
    },
]
prompt = evaluator.vlm_module.processor.apply_chat_template(
    conversation,
    add_generation_prompt=True,
)

inputs = evaluator.vlm_module.processor(images=image, text=prompt, return_tensors="pt").to(device)

with torch.no_grad():
    outputs = evaluator.vlm_module.model.generate(
        **inputs,
        output_attentions=True,
        return_dict_in_generate=True,
        max_new_tokens=1,
        pad_token_id=evaluator.vlm_module.processor.tokenizer.eos_token_id,
    )
self_attn = outputs.attentions[0]

# %%
all_attn = []
# Iterate over layers and heads
for layer in range(len(self_attn)):
    layer_attn = self_attn[layer][0]  # shape: (heads, tokens, tokens)
    for head in range(layer_attn.shape[0]):
        # Extract attention: question tokens (rows) to image tokens (columns)
        attn_map = layer_attn[head]
        all_attn.append(attn_map)

# Stack all maps, shape: (num_layers * num_heads, num_question_tokens, num_image_tokens)
all_attn = torch.stack(all_attn, dim=0)

# Max over layers and heads: shape (num_question_tokens, num_image_tokens)
self_attn = all_attn.max(dim=0).values

# %%
import torch.nn.functional as F

token_masks = evaluator.vlm_module.get_token_masks(
    inputs,
    128256,
    list(evaluator.vlm_module.processor.tokenizer.added_tokens_decoder.keys()),
    self_attn.shape[-1],  # image_token_id
)
input_toknes = inputs["input_ids"][:, token_masks["question"]]

input_attn = self_attn[token_masks["question"],][2:-5,].mean(0)[token_masks["image"]][:576]
attn_map = input_attn.reshape(1, 1, 24, 24)
attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)

attn_map = F.interpolate(attn_map.float(), size=(224, 224), mode="bilinear", align_corners=False)
plt.imshow(attn_map.squeeze().cpu().numpy())
plt.show()
# for i in range(len(input_toknes[0])):
#     input_attn = self_attn[token_masks["question"],][i,][token_masks["image"]][:576]

#     attn_map = input_attn.reshape(1, 1, 24, 24)

#     attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)

#     # Upsample to match the input image size
#     # attn_map = F.interpolate(attn_map, size=(224, 224), mode="bilinear", align_corners=False)
#     attn_map = F.interpolate(attn_map, size=(224, 224), mode="nearest")
#     attn_map = attn_map.squeeze().cpu().numpy()
#     # attn_map = np.where(attn_map > 0.5, 1,0 )
#     plt.imshow(attn_map)
#     plt.show()
#     plt.close()

pred_mask = attn_map.squeeze().cpu().numpy()
gt_mask = binary_mask.squeeze()
from sklearn.metrics import auc, precision_recall_curve

precision, recall, _ = precision_recall_curve(gt_mask.flatten(), pred_mask.flatten())
score = auc(recall, precision)
print(score)

# %%
gradcam_style_overlay(image, pred_mask, colormap="jet", alpha=0.5)
