import numpy as np
import streamlit as st


def save_base_info(base_info: dict):
    st.session_state.class_names = base_info["class_names"]
    st.session_state.concept_dict = base_info["concept_dict"]


def get_image_data_url(path):
    return f"data:image/png;base64,{path}"


def process_concept_distribution(df):

    df["thumbnail_image"] = df["thumbnail_image"].apply(lambda x: get_image_data_url(x))
    df["full_image"] = df["full_image"].apply(lambda x: get_image_data_url(x))
    df.sort_values(by="Mean", ascending=False, inplace=True)

    df = df.reset_index(drop=True)
    target_indices = df[df["Concept Type"] == "target"].index
    context_indices = df[df["Concept Type"] == "context"].index
    bias_indices = df[df["Concept Type"] == "bias"].index
    display_indices = np.concatenate([target_indices, bias_indices, context_indices])
    return df.iloc[display_indices]
