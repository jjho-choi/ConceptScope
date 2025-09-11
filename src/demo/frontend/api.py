import os

import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()


@st.cache_data(show_spinner="Loading dataset")
def load_base_info(dataset_name: str) -> pd.DataFrame:
    res = requests.get(f"http://localhost:{os.getenv('PORT')}/load_base_info?dataset_name={dataset_name}")
    res.raise_for_status()
    data = res.json()
    return data


@st.cache_data(show_spinner="Loading concept distribution")
def get_concept_distribution(class_idx: int) -> pd.DataFrame:
    res = requests.get(f"http://localhost:{os.getenv('PORT')}/get_concept_distribution?class_idx={class_idx}")
    res.raise_for_status()
    data = res.json()
    return pd.DataFrame(data)


@st.cache_data(show_spinner="Loading concept info")
def get_concept_info(latent_idx: int, top_k_images: int) -> dict:
    res = requests.get(
        f"http://localhost:{os.getenv('PORT')}/get_concept_info?latent_idx={latent_idx}&top_k_images={top_k_images}"
    )
    res.raise_for_status()
    data = res.json()
    return data


@st.cache_data(show_spinner="Loading class images")
def get_images_from_class(class_name: str, latent_idx: int, top_k: int) -> dict:
    class_idx = st.session_state.class_names.index(class_name)
    res = requests.get(
        f"http://localhost:{os.getenv('PORT')}/get_images_from_class?class_idx={class_idx}&latent_idx={latent_idx}&top_k={top_k}"
    )
    res.raise_for_status()
    data = res.json()
    return data
