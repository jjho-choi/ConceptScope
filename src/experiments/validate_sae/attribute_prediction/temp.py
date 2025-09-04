import argparse
import ast
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


async def generate_async(model, prompt: str):
    """
    Asynchronously calls the Gemini API for a single prompt.
    Includes basic error handling.
    """
    try:
        # The async version of the 'generate_content' method
        response = await model.generate_content_async(prompt)
        return response.text
    except Exception as e:
        print(f"Error processing prompt '{prompt[:30]}...': {e}")
        return None  # Or return a custom error message


class VLMAttributePredictionEvaluator:
    def __init__(self, save_root, dir_name, model_name, device):
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
        self.caption_dir = "vlm_captions2"



    async def match_batch_by_gemini(self, captions, attr_name, model_name="gemini-1.5-flash", batch_size=16, max_retries=3):
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

                    tasks = [generate_async(model, prompt) for prompt in prompts]
                    out = await asyncio.gather(*tasks)

                    responses = model.batch_generate_content(batch)
                    batch_results = [1 if r.strip().lower() == "yes" else 0 for r in responses.text.strip().split("\n")]
                    results.extend(batch_results)
                    break
                except Exception as e:
                    print(f"[Attempt {attempt + 1}] Error in batch {i // batch_size + 1}: {e}")
                    time.sleep(1)
            else:
                print(f"Failed to process batch {i // batch_size + 1} after {max_retries} attempts.")
                results.extend([0] * len(batch))
        return np.array(results)

    def evaluate(self, captions, keyword_map, binary_target):
        def lemmatize_phrase(phrase):
            tokens = nlp(phrase.lower())
            return {"lemmas": [token.lemma_ for token in tokens], "stems": [ps.stem(token.text) for token in tokens]}

        lemmatized_keywords = [lemmatize_phrase(k) for k in keyword_map]
        print(f"evaluating {len(captions)} captions for {keyword_map[-1]}")
        scores = np.zeros(len(captions), dtype=int)
        scores = await self.match_batch_by_gemini(
            captions, keyword_map[-1], model_name="gemini-2.0-flash-lite", batch_size=32
        )

        precision = precision_score(binary_target, scores)
        recall = recall_score(binary_target, scores)
        f1 = f1_score(binary_target, scores)

        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
        }

    def run(self, dataset_name, split, num_iter=5):
        dataset, attr_list = self.get_dataset_and_attrs(dataset_name, split)
        all_captions = self.get_captions(dataset, dataset_name, split, idx)


        for idx in range(num_iter):
            score_dict = {}
                binary_target = self.eval_helper.get_binary_target(dataset, dataset_name, cls_idx)
                scores = self.evaluate(all_captions, keyword_mapping[attr], binary_target)
                score_dict[attr] = scores

            save_path = f"{self.save_root}/{self.dir_name}/{dataset_name}/{self.caption_dir}/gemini_scores_{self.model_name}_{idx}.json"
            with open(save_path, "w") as f:
                json.dump(score_dict, f, indent=4)

  



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate VLM predictions for attribute prediction")
    parser.add_argument("--save_root", type=str, default="./out", help="Root directory to save results")
    parser.add_argument("--dir_name", type=str, default="dataset_analysis", help="Directory name for saving results")
    parser.add_argument("--model_name", type=str, choices=["blip2", "llava_next"], required=True, help="VLM model name")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model on")
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset to evaluate")
    parser.add_argument("--split", type=str, default="test", help="Dataset split to evaluate")
    parser.add_argument("--num_iter", type=int, default=5, help="Number of iterations for evaluation")

    args = parser.parse_args()

    evaluator = VLMAttributePredictionEvaluator(
        save_root=args.save_root,
        dir_name=args.dir_name,
        model_name=args.model_name,
        device=args.device,
    )

    split_dict = {
        # "waterbird": "test",
        "caltech101": "test",
        "dtd": "test",
        "stanford_action": "test",
        "raf-db": "test",
        "celeba": "test",
    }
    for dataset_name, split in split_dict.items():
        args.dataset_name = dataset_name
        args.split = split
        with torch.no_grad():
            print(f"Running evaluation for {dataset_name} on split {split}")
            evaluator.run(args.dataset_name, args.split, num_iter=args.num_iter)
