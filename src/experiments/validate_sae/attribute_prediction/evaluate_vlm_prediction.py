import argparse
import ast
import asyncio
import json
import os
import time

import google.generativeai as genai
import numpy as np
import openai
import spacy
import torch
from nltk.stem import PorterStemmer
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm

from credential import GEMINIKEY, OPENAIKEY
from src.experiments.validate_sae.attribute_prediction.attr_prediction_evaluator import (
    AttrPredictionEvaluator,
)
from src.experiments.validate_sae.vlm_baseline.vlm_module import (
    BLIP2Module,
    LlavaNextModule,
)
from src.utils.image_dataset_loader import ImageDatasetLoader

try:
    nlp = spacy.load("en_core_web_sm", exclude=["lemmatizer"])
except OSError:
    from spacy.cli import download

    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm", exclude=["lemmatizer"])

nlp.add_pipe("lemmatizer", config={"mode": "lookup"}, last=True)
nlp.initialize()
ps = PorterStemmer()

openai.api_key = OPENAIKEY
genai.configure(api_key=GEMINIKEY)


class VLMAttributePredictionEvaluator:
    def __init__(self, save_root, dir_name, model_name, device, eval_method="gemini", caption_dir="vlm_captions2"):
        self.save_root = save_root
        self.dir_name = dir_name
        self.model_name = model_name
        self.device = device
        if model_name == "blip2":
            self.vlm_module = BLIP2Module(device=device)
        elif model_name == "llava_next":
            self.vlm_module = LlavaNextModule(device=device)
        self.eval_helper = AttrPredictionEvaluator()
        self.num_samples = 100
        self.caption_dir = caption_dir
        self.eval_method = eval_method

    async def generate_async(self, model, prompt: str):

        try:
            response = await model.generate_content_async(prompt)
            return response.text
        except Exception as e:
            print(f"Error processing prompt '{prompt[:30]}...': {e}")
            return None

    def match_keywords(self, caption, lemmatized_keywords):
        caption_doc = nlp(caption.lower())
        caption_lemmas = [token.lemma_ for token in caption_doc]
        caption_stems = [ps.stem(token.text) for token in caption_doc]

        for keyword in lemmatized_keywords:
            phrase_lemmas = keyword["lemmas"]
            phrase_stems = keyword["stems"]
            n = len(phrase_lemmas)

            for i in range(len(caption_lemmas) - n + 1):
                if all(caption_lemmas[i + j] == phrase_lemmas[j] for j in range(n)):
                    return 1

            for i in range(len(caption_stems) - n + 1):
                if all(caption_stems[i + j] == phrase_stems[j] for j in range(n)):
                    return 1

        return 0

    def evaluate_keyword_matching(self, captions, keyword_map, binary_target):
        def lemmatize_phrase(phrase):
            tokens = nlp(phrase.lower())
            return {"lemmas": [token.lemma_ for token in tokens], "stems": [ps.stem(token.text) for token in tokens]}

        lemmatized_keywords = [lemmatize_phrase(k) for k in keyword_map]

        scores = np.zeros(len(captions), dtype=int)
        for i, caption in enumerate((captions)):
            is_match = self.match_keywords(caption, lemmatized_keywords)
            scores[i] = is_match

        precision = precision_score(binary_target, scores)
        recall = recall_score(binary_target, scores)
        f1 = f1_score(binary_target, scores)

        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
        }

    def match_by_gpt(self, caption, attr_name, model_name="gpt-3.5-turbo", max_retries=3):
        caption = caption.lower()
        attr_name = " ".join(attr_name.split("_")).lower()
        prompt = f"Does the caption: '{caption}' contain the attribute: '{attr_name}'? Answer 'yes' if the meaning is semantically included. Answer with 'yes' or 'no'."

        for attempt in range(max_retries):
            try:
                response = openai.chat.completions.create(
                    model=model_name,
                    messages=[
                        {
                            "role": "user",
                            "content": [{"type": "text", "text": prompt}],
                        }
                    ],
                    max_tokens=10,
                )
                result = response.choices[0].message.content.strip().lower()
                return 1 if result == "yes" else 0

            except Exception as e:
                print(f"[Attempt {attempt + 1}] Error: {e}")
                time.sleep(1)
        return 0

    async def match_batch_by_gemini(
        self, captions, attr_name, model_name="gemini-1.5-flash", batch_size=16, max_retries=3
    ):
        attr_name = " ".join(attr_name.split("_")).lower()

        def build_prompt(caption):
            caption = caption.lower()
            return (
                f"Given the following image caption:\n"
                f'"{caption}"\n\n'
                f'Determine whether the attribute "{attr_name}" is present in the caption, either explicitly or implicitly. '
                f"Consider synonyms, paraphrases, or implied meanings.\n"
                f"Answer strictly with 'yes' if the caption includes the attribute in a semantically meaningful way. "
                f"Otherwise, answer 'no'.\n\n"
                f"Answer with 'yes' or 'no'."
            )

        prompts = [build_prompt(c) for c in captions]
        model = genai.GenerativeModel(model_name)
        results = []

        for i in tqdm(range(0, len(prompts), batch_size)):
            batch = prompts[i : i + batch_size]
            for attempt in range(max_retries):
                try:

                    tasks = [self.generate_async(model, prompt) for prompt in batch]
                    out = await asyncio.gather(*tasks)
                    # batch_results = [1 if r.strip().lower() == "yes" else 0 for r in out if r is not None]
                    batch_results = [1 if (r and r.strip().lower() == "yes") else 0 for r in out]
                    if len(batch_results) < len(batch):
                        print(f"Warning: Some prompts in batch {i // batch_size + 1} returned None. Filling with 0s.")
                    results.extend(batch_results)
                    break
                except Exception as e:
                    print(f"[Attempt {attempt + 1}] Error in batch {i // batch_size + 1}: {e}")
                    time.sleep(1)
            # else:
            #     print(f"Failed to process batch {i // batch_size + 1} after {max_retries} attempts.")
            #     results.extend([0] * len(batch))
        return np.array(results)

    async def evaluate(self, captions, keyword_map, binary_target):
        def lemmatize_phrase(phrase):
            tokens = nlp(phrase.lower())
            return {"lemmas": [token.lemma_ for token in tokens], "stems": [ps.stem(token.text) for token in tokens]}

        lemmatized_keywords = [lemmatize_phrase(k) for k in keyword_map]
        print(f"evaluating {len(captions)} captions for {keyword_map[-1]}")

        if self.eval_method == "gemini":
            scores = await self.match_batch_by_gemini(
                captions, keyword_map[-1], model_name="gemini-2.0-flash-lite", batch_size=16
            )
        else:
            scores = np.zeros(len(captions), dtype=int)

            for i, caption in enumerate(tqdm(captions)):
                if self.eval_method == "keyword":
                    is_match = self.match_keywords(caption, lemmatized_keywords)
                elif self.eval_method == "gpt":
                    is_match = self.match_by_gpt(caption, keyword_map[-1], model_name="gpt-3.5-turbo")
                scores[i] = is_match

        precision = precision_score(binary_target, scores)
        recall = recall_score(binary_target, scores)
        f1 = f1_score(binary_target, scores)

        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
        }

    async def run(self, dataset_name, split, num_iter=5):
        dataset, attr_list = self.get_dataset_and_attrs(dataset_name, split)

        for idx in range(num_iter):
            save_path = f"{self.save_root}/{self.dir_name}/{dataset_name}/{self.caption_dir}/{self.eval_method}_scores_{self.model_name}_{idx}.json"
            if os.path.exists(save_path):
                with open(save_path, "r") as f:
                    score_dict = json.load(f)
            else:
                score_dict = {}
            print(f"Iteration {idx+ 1}/{num_iter} for dataset: {dataset_name}, split: {split}")
            keyword_mapping = self.get_keyword_mapping(dataset_name, attr_list, idx)
            if dataset_name != "celeba":
                all_captions = self.get_captions(dataset, dataset_name, split, idx)
            for cls_idx, attr in tqdm(enumerate(attr_list)):
                if attr in score_dict:
                    continue
                if dataset_name == "celeba":
                    dataset = ImageDatasetLoader.load_dataset(dataset_name, split=split, target_attribute=attr)
                    dataset, indices = self.eval_helper.subsample_dataset_by_class(
                        dataset, num_samples=self.num_samples
                    )
                    all_captions = self.get_captions(dataset, dataset_name, split, idx, target_attribute=attr)

                binary_target = self.eval_helper.get_binary_target(dataset, dataset_name, cls_idx)
                scores = await self.evaluate(all_captions, keyword_mapping[attr], binary_target)
                score_dict[attr] = scores

                with open(save_path, "w") as f:
                    json.dump(score_dict, f, indent=4)

    def get_dataset_and_attrs(self, dataset_name, split, target_attribute=None):
        if dataset_name == "waterbird":
            target_attribute = "detailed_place"
        dataset = ImageDatasetLoader.load_dataset(dataset_name, split=split, target_attribute=target_attribute)
        if dataset_name != "celeba":
            dataset, _ = self.eval_helper.subsample_dataset_by_class(dataset, num_samples=self.num_samples)
        attr_list = self.eval_helper.get_attr_list(dataset_name, dataset)
        return dataset, attr_list

    def get_captions(self, dataset, dataset_name, split, num_iter, target_attribute=None):
        if dataset_name == "celeba":
            save_path = f"{self.save_root}/{self.dir_name}/{dataset_name}/{self.caption_dir}/{split}_set_{self.model_name}_{target_attribute}_{num_iter}.json"
        else:
            save_path = f"{self.save_root}/{self.dir_name}/{dataset_name}/{self.caption_dir}/{split}_set_{self.model_name}_{num_iter}.json"
        if os.path.exists(save_path):
            with open(save_path, "r") as f:
                captions = json.load(f)
        else:
            captions = self.vlm_module.generate_captions_all(dataset, save_path)
        return captions

    def get_keyword_mapping(self, dataset_name, attr_list, num_iter):
        save_path = (
            f"{self.save_root}/{self.dir_name}/{dataset_name}/{self.caption_dir}/keyword_mapping_{num_iter}.json"
        )
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if os.path.exists(save_path):
            with open(save_path, "r") as f:
                mapping_dict = json.load(f)
        else:
            mapping_dict = {}
            for attr in tqdm(attr_list):
                keyword_list = self.get_synonyms(attr)
                mapping_dict[attr] = keyword_list
            with open(save_path, "w") as f:
                json.dump(mapping_dict, f, indent=4)
        return mapping_dict

    def get_synonyms(self, attr, model_name="gpt-4o", max_retries=3):
        attr = " ".join(attr.split("_")).lower()
        prompt = f"For the given class label: {attr}, provide synonyms or phrases that are semantically equivalent. Return the output in JSON array format."

        for attempt in range(max_retries):
            try:
                response = openai.chat.completions.create(
                    model=model_name,
                    messages=[
                        {
                            "role": "user",
                            "content": [{"type": "text", "text": prompt}],
                        }
                    ],
                    max_tokens=300,
                )
                result = response.choices[0].message.content
                clean_str = result.strip("`").split("\n", 1)[-1].rsplit("\n", 1)[0]
                parsed_list = json.loads(clean_str)
                parsed_list += [attr]
                return parsed_list

            except json.JSONDecodeError:
                print(f"[Attempt {attempt + 1}] JSON parsing failed. Retrying...")
                time.sleep(1)
            except Exception as e:
                print(f"[Attempt {attempt + 1}] Unexpected error: {e}")
                break
        print("Failed to parse after retries.")
        return [attr]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate VLM predictions for attribute prediction")
    parser.add_argument("--save_root", type=str, default="./out", help="Root directory to save results")
    parser.add_argument("--dir_name", type=str, default="dataset_analysis", help="Directory name for saving results")
    parser.add_argument("--model_name", type=str, choices=["blip2", "llava_next"], required=True, help="VLM model name")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model on")
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset to evaluate")
    parser.add_argument("--split", type=str, default="test", help="Dataset split to evaluate")
    parser.add_argument("--num_iter", type=int, default=5, help="Number of iterations for evaluation")
    parser.add_argument(
        "--eval_method", type=str, default="gemini", choices=["gemini", "gpt", "keyword"], help="Evaluation method"
    )

    args = parser.parse_args()

    evaluator = VLMAttributePredictionEvaluator(
        save_root=args.save_root,
        dir_name=args.dir_name,
        model_name=args.model_name,
        device=args.device,
        eval_method="gemini",
    )

    split_dict = {
        # "waterbird": "test",
        # "caltech101": "test",
        # "dtd": "test",
        # "stanford_action": "test",
        # "raf-db": "test",
        "celeba": "test",
    }
    for dataset_name, split in split_dict.items():
        args.dataset_name = dataset_name
        args.split = split
        with torch.no_grad():
            print(f"Running evaluation for {dataset_name} on split {split}")
            if args.eval_method == "gemini":
                asyncio.run(evaluator.run(args.dataset_name, args.split, num_iter=args.num_iter))
            else:
                evaluator.run(args.dataset_name, args.split, num_iter=args.num_iter)
