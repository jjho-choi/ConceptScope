import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor


class CLIPPreprocessor:
    def __init__(self, clip_model_name, device="cuda"):
        self.device = device
        self.clip_model_name = clip_model_name

        self.clip_model = CLIPModel.from_pretrained(clip_model_name).to(device)
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)

    def get_image_text_cos(self, image, label_embedding, batch_size=64):
        image_embedding = self.get_image_embedding(image, batch_size=batch_size)
        cos_sim_matrix = image_embedding @ label_embedding.T
        label_masked_high = cos_sim_matrix.max(dim=1).values.cpu().numpy()
        return label_masked_high

    def get_text_embedding(self, text):
        inputs = self.clip_processor(text, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.clip_model.get_text_features(**inputs)
        norm_embedding = F.normalize(outputs, p=2, dim=1)
        return norm_embedding

    def get_image_embedding(self, images, batch_size=64):
        out_embedding = torch.zeros((len(images), self.clip_model.config.projection_dim), device=self.device)
        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size]
            if isinstance(batch, torch.Tensor):
                inputs = {"pixel_values": batch}
            else:
                inputs = self.clip_processor(images=batch, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.clip_model.get_image_features(**inputs)
            norm_embedding = F.normalize(outputs, p=2, dim=1)
            out_embedding[i : i + batch_size] = norm_embedding
        return out_embedding
