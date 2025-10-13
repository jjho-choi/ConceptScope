import os

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.demo.backend.processor import Processor

load_dotenv("../../../.env")

app = FastAPI()
processor = Processor(
    root=os.getenv("VISUALIZER_ROOT"), device=os.getenv("DEVICE"), checkpoint_name=os.getenv("CHECKPOINT_NAME")
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/load_base_info")
def load_base_info(dataset_name: str = Query(...)):
    dataset_name = dataset_name.lower()
    processor.dataset_name = dataset_name
    processor.get_concept_categorization_dict(dataset_name=dataset_name)
    return {
        "class_names": processor.class_names,
        "concept_dict": processor.concept_categorization_dict,
    }


@app.get("/get_concept_distribution")
def get_concept_distribution(class_idx: int = Query(...), dataset_name: str = Query(...)):
    df = processor.get_concept_distribution(class_idx)
    return df.to_dict()


@app.get("/get_concept_info")
def get_concept_info(latent_idx: int = Query(...), top_k_images: int = Query(...)):
    return processor.get_concept_info(latent_idx, top_k_images=top_k_images)


@app.get("/get_images_from_class")
def get_images_from_class(
    class_idx: int = Query(...), latent_idx: int = Query(...), top_k: int = Query(...), dataset_name: str = Query(...)
):
    return processor.get_images_from_class(class_idx, latent_idx, top_k=top_k)


@app.get("/get_images_with_prediction")
def get_images_with_prediction(
    class_idx: int = Query(...),
    latent_idx: int = Query(...),
    top_k: int = Query(...),
    dataset_name: str = Query(...),
    threshold: float = Query(...),
):
    return processor.get_images_with_prediction(
        class_idx, latent_idx, top_k=top_k, threshold=threshold, dataset_name=dataset_name.lower()
    )
