from io import BytesIO

import numpy as np
import streamlit as st
from PIL import Image

import src.demo.frontend.api as api
import src.demo.frontend.figures as figures
import src.demo.frontend.utils as utils
from src.demo.config import (
    AppConfig,
    ClassOverviewConfig,
    ClassSampleViewConfig,
    ConceptViewConfig,
)


def set_page():
    st.set_page_config(
        layout="wide", page_title="ConceptScope Demo", page_icon=":dog:", initial_sidebar_state="collapsed"
    )
    st.markdown("### ConceptScope Demo :dog:")


def set_config_view():
    st.subheader("Dataset and Model Configuration")
    selected_dataset = st.selectbox(
        "Select Dataset",
        AppConfig.dataset_options,
        index=AppConfig.dataset_options.index(AppConfig.default_dataset),
        key="dataset_select",
    )

    st.slider(
        "Top-k Concepts for Class",
        min_value=ClassOverviewConfig.min_top_k_dist,
        max_value=ClassOverviewConfig.max_top_k_dist,
        value=ClassOverviewConfig.default_top_k_dist,
        step=ClassOverviewConfig.step_top_k_dist,
        key="top_k_for_concept_dist",
    )

    st.slider(
        "Top-k samples for Concepts",
        min_value=ConceptViewConfig.min_top_k_samples,
        max_value=ConceptViewConfig.max_top_k_samples,
        value=ConceptViewConfig.default_top_k_samples,
        step=ConceptViewConfig.step_top_k_samples,
        key="top_k_sample_for_concept",
    )

    st.slider(
        "Top-k samples for Class",
        min_value=ClassSampleViewConfig.min_top_k_samples,
        max_value=ClassSampleViewConfig.max_top_k_samples,
        value=ClassSampleViewConfig.default_top_k_samples,
        step=ClassSampleViewConfig.step_top_k_samples,
        key="top_k_sample_for_class",
    )

    return selected_dataset


def render_class_view():
    selected_class = st.selectbox(
        "Select Class",
        st.session_state.class_names,
        index=0,
        key="class_select",
    )

    if selected_class != st.session_state.selected_class:
        st.session_state.selected_class = selected_class
        class_idx = st.session_state.class_names.index(selected_class)

        concept_distribution = api.get_concept_distribution(class_idx)
        concept_distribution = utils.process_concept_distribution(concept_distribution)
        st.session_state.concept_distribution = concept_distribution

    col1, col2 = st.columns(2)
    with col1:
        fig = figures.plot_concept_bar(
            st.session_state.concept_distribution,
            num_concepts=st.session_state.top_k_for_concept_dist,
            height=ClassOverviewConfig.height,
        )
        st.plotly_chart(fig)
    with col2:
        show_class_concepts(st.session_state.concept_distribution)

    return selected_class


def show_horizontal_image_row(b64_images, captions=None, height=100, font_size=12, highlight_idx=None):
    """
    Render images in a horizontal scrollable row.
    """
    html = '<div style="display: flex; overflow-x: auto; gap: 10px; margin-top: 0; padding: 3px;">'
    for i, img in enumerate(b64_images):
        border_color = "#4CAF50" if (highlight_idx is not None and i == highlight_idx) else "#ccc"
        html += (
            f'<div style="flex-shrink: 0; text-align: center;">'
            f'<img src="data:image/png;base64,{img}" '
            f'style="height: {height}px; border-radius: 4px; border: 3px solid {border_color};" />'
        )
        if captions:
            html += f'<div style="font-size:{font_size}px; white-space:nowrap;">{captions[i]}</div>'
        html += "</div>"
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def show_class_concepts(df):

    st.dataframe(
        df[
            [
                "full_image",
                "latent_name",
                "Concept Type",
                "Mean",
                "Class aligned",
            ]
        ],
        column_config={
            "full_image": st.column_config.ImageColumn(
                "Reference Image", help="Reference image of the concept", pinned=False, width="medium"
            ),
            "latent_name": st.column_config.TextColumn("Concept Name"),
            "Concept Type": st.column_config.TextColumn("Concept Type"),
            "Mean": st.column_config.ProgressColumn(
                "Concept Strength",
                format="%.2f",
            ),
            "Class aligned": st.column_config.ProgressColumn(
                "Alignment Score",
                format="%.2f",
                help="Alignment score for the concept",
            ),
        },
        hide_index=True,
    )


def render_concepts_list():
    df = st.session_state.concept_distribution

    for i in range(len(df)):
        with st.container(border=True):
            row = df.iloc[i]
            cols = st.columns([2, 2], vertical_alignment="top")
            with cols[0]:
                if st.button(f"Concept {i+1}", key=f"select_{i + 1}"):
                    if st.session_state.selected_concept != i:
                        st.session_state.selected_concept = i
                st.image(row["thumbnail_image"], width=ConceptViewConfig.thumbnail_width)

            with cols[1]:
                st.markdown(f"**Id**: {row['latent_idx']}")
                st.markdown(f"**Name**: {row['latent_name']}")
                st.markdown(f"**Type**: {row['Concept Type']}")


def render_concept_info():
    concept_info = st.session_state.concept_distribution.iloc[st.session_state.selected_concept]
    latent_idx = int(concept_info["latent_idx"])
    concept_images = api.get_concept_info(latent_idx, top_k_images=st.session_state.top_k_sample_for_concept)
    st.markdown(f"#### {concept_info['latent_name']} - {latent_idx}")

    st.markdown("**Reference Images**")
    show_mask = st.checkbox("Show masked images", value=True, key="show_masked_images_concept")

    if show_mask:
        show_horizontal_image_row(concept_images["masked_high_images"])
    else:
        show_horizontal_image_row(concept_images["high_activating_images"])

    class_images = api.get_images_from_class(
        st.session_state.selected_class, latent_idx, top_k=st.session_state.top_k_sample_for_concept
    )
    st.markdown("**High activating class images**")
    show_mask = st.checkbox("Show masked images", value=True, key="show_masked_images_class")
    if show_mask:
        show_horizontal_image_row(class_images["masked_highest_images"])
    else:
        show_horizontal_image_row(class_images["highest_images"])

    st.markdown("**Low activating class images**")
    show_horizontal_image_row(class_images["lowest_images"])
