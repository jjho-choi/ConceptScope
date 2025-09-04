import json

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import (
    Blip2ForConditionalGeneration,
    Blip2Processor,
    LlavaNextForConditionalGeneration,
    LlavaNextProcessor,
)


class VLMModule:
    def __init__(self, model_name, device):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.processor = None

    def process_image(self, image):
        if isinstance(image, str):
            image = Image.open(image)
        return image.convert("RGB").resize((256, 256))

    def generate_captions_all(self, dataset, caption_path):
        out = []
        for sample in tqdm(dataset):
            image = self.process_image(sample["image"])
            caption = self.generate_caption(image)
            out.append(caption)

            with open(caption_path, "w") as f:
                json.dump(out, f, indent=4)

        return out


class BLIP2Module(VLMModule):
    def __init__(self, model_name="Salesforce/blip2-opt-2.7b", device="cuda"):
        super().__init__(model_name, device)
        self.model, self.processor = self.get_model(model_name, device)

    def get_model(self, model_name, device):
        processor = Blip2Processor.from_pretrained(model_name)
        model = Blip2ForConditionalGeneration.from_pretrained(model_name)
        model.to(device)
        return model.eval(), processor

    def generate_caption(self, image):
        inputs = self.processor(image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                do_sample=True,
                max_new_tokens=256,
            )
        caption = self.processor.decode(output_ids[0], skip_special_tokens=True)
        return caption

    def get_segmentation_mask(self, image, text_label):
        inputs = self.processor(images=image, text=text_label, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            vision_outputs = self.model.vision_model(
                pixel_values=inputs["pixel_values"], output_hidden_states=True, return_dict=True
            )
            image_embeds = vision_outputs.last_hidden_state

        query_tokens = self.model.query_tokens.expand(image_embeds.shape[0], -1, -1)
        with torch.no_grad():
            qformer_outputs = self.model.qformer(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=self.device),
                output_attentions=True,
                return_dict=True,
            )

        all_attn = []
        attns = qformer_outputs.cross_attentions
        for attn in attns:
            layer_attn = attn[0]
            for head in layer_attn:
                all_attn.append(head)
        all_attn = torch.stack(all_attn, dim=0)
        self_attn = all_attn.max(dim=0).values
        attn_map = self_attn.max(dim=0).values[1:].view(1, 1, 16, 16)
        return attn_map


class LlavaNextModule(VLMModule):
    def __init__(self, model_name="llava-hf/llama3-llava-next-8b-hf", device="cuda"):
        super().__init__(model_name, device)
        self.model, self.processor = self.get_model(model_name, device)

    def get_model(self, model_name, device):
        processor = LlavaNextProcessor.from_pretrained(model_name)
        model = LlavaNextForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch.float16, low_cpu_mem_usage=True
        ).to(device)
        return model.eval(), processor

    def generate_caption(self, image, max_new_tokens=256):
        prompt = "Describe this image."
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image"},
                ],
            },
        ]
        prompt = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
        )

        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.processor.tokenizer.eos_token_id,
                do_sample=True,
            )
        caption = self.processor.decode(output_ids[0], skip_special_tokens=True)
        return caption

    def get_token_masks(self, inputs, image_token_id, special_token_ids, input_ids_len):
        """
        Generate masks for different token types in the model input.

        Args:
            inputs: Model inputs containing input_ids
            image_token_id: Token ID for image tokens
            special_token_ids: List of special token IDs
            input_ids_len: Length of input IDs

        Returns:
            Dictionary containing boolean masks for different token types
        """
        image_token_mask = (inputs["input_ids"] == image_token_id)[0].detach().cpu().numpy()

        system_prompt_mask = np.zeros((input_ids_len,))
        system_prompt_mask[: np.where(image_token_mask)[0][0]] = 1
        system_prompt_mask = system_prompt_mask.astype(bool)

        question_token_mask = np.zeros((input_ids_len,))
        question_token_mask[np.where(image_token_mask)[0][-1] :] = 1
        question_token_mask = question_token_mask.astype(bool)

        assert (question_token_mask * image_token_mask * system_prompt_mask).sum() == 0

        special_token_mask = inputs["input_ids"][0].detach().cpu().numpy()
        special_token_mask = np.isin(special_token_mask, special_token_ids).astype(bool)

        return {
            "image": image_token_mask,
            "system": system_prompt_mask,
            "question": question_token_mask,
            "special": special_token_mask,
        }

    def get_segmentation_mask(self, image, text_label):
        image = image.resize((336, 336))
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text_label},
                    {"type": "image"},
                ],
            },
        ]
        prompt = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
        )

        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                output_attentions=True,
                return_dict_in_generate=True,
                max_new_tokens=1,
                pad_token_id=self.processor.tokenizer.eos_token_id,
            )

        self_attn = outputs.attentions[0]
        all_attn = []

        for layer_attn in self_attn:
            layer_attn = layer_attn[0]  #
            for head in layer_attn:
                all_attn.append(head)
        all_attn = torch.stack(all_attn, dim=0)
        self_attn = all_attn.max(dim=0).values

        token_masks = self.get_token_masks(
            inputs,
            128256,
            list(self.processor.tokenizer.added_tokens_decoder.keys()),
            self_attn.shape[-1],  # image_token_id
        )

        input_attn = self_attn[token_masks["question"],][2:-5,].mean(0)[token_masks["image"]][:576]
        attn_map = input_attn.reshape(1, 1, 24, 24)
        return attn_map
