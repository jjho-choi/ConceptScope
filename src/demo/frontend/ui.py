import pandas as pd
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
    st.subheader("VisualizationConfiguration")

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


def render_option_view():
    col1, col2 = st.columns(2)
    with col1:
        selected_dataset = st.selectbox(
            "Select Dataset",
            AppConfig.dataset_options,
            index=AppConfig.dataset_options.index(AppConfig.default_dataset),
            key="dataset_select",
        )

    if selected_dataset != st.session_state.selected_dataset:
        base_info = api.load_base_info(selected_dataset)
        utils.save_base_info(base_info)

    with col2:
        selected_class = st.selectbox(
            "Select Class",
            st.session_state.class_names,
            index=0,
            key="class_select",
        )
    return selected_dataset, selected_class


def render_class_view():
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


def show_horizontal_image_row(b64_images, captions=None, height=100, font_size=12, colors=None):
    """
    Render images in a horizontal scrollable row.
    """
    html = '<div style="display: flex; overflow-x: auto; gap: 10px; margin-top: 0; padding: 3px;">'
    for i, img in enumerate(b64_images):
        border_color = "#ccc" if colors is None else colors[i]
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
                "concept strength",
                "alignment score",
            ]
        ],
        column_config={
            "full_image": st.column_config.ImageColumn(
                "Reference Image", help="Reference image of the concept", pinned=False, width="medium"
            ),
            "latent_name": st.column_config.TextColumn("Concept Name"),
            "Concept Type": st.column_config.TextColumn("Concept Type"),
            "concept strength": st.column_config.ProgressColumn(
                "Concept Strength",
                format="%.3f",
            ),
            "alignment score": st.column_config.ProgressColumn(
                "Alignment Score",
                format="%.3f",
                help="Alignment score for the concept",
                min_value=df["alignment score"].min(),
                max_value=df["alignment score"].max(),
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
    st.session_state.latent_avg_activations = concept_images["latent_avg_activations"]
    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown(f"#### {concept_info['latent_name']} - {latent_idx}")
        fig, info = figures.plot_top_class_for_concept(
            concept_images["latent_avg_activations"],
            st.session_state.selected_class,
            st.session_state.class_names,
        )
        st.plotly_chart(fig)
        st.info(info)

    with col2:

        show_referfence_images(concept_images)
        show_train_sample_images(latent_idx)

        class_dict = show_test_sample_images(latent_idx)

    with col1:
        show_high_low_diff(class_dict)


def show_referfence_images(concept_images):
    st.markdown(
        f"<p style='font-size:{AppConfig.mid_font_size}px; font-weight:bold;'>Reference Images</p>",
        unsafe_allow_html=True,
    )
    show_mask = st.checkbox("Show masked images", value=True, key="show_masked_images")

    if show_mask:
        show_horizontal_image_row(concept_images["masked_high_images"])
    else:
        show_horizontal_image_row(concept_images["high_activating_images"])


def show_train_sample_images(latent_idx):
    class_images = api.get_images_from_class(
        st.session_state.selected_class,
        latent_idx,
        top_k=st.session_state.top_k_sample_for_concept,
        dataset_name=st.session_state.selected_dataset,
    )
    st.markdown(
        f"<p style='font-size:{AppConfig.mid_font_size}px; font-weight:bold;'>High activating class images from train split</p>",
        unsafe_allow_html=True,
    )
    if st.session_state.show_masked_images:
        show_horizontal_image_row(class_images["masked_highest_images"], captions=class_images["high_activations"])
    else:
        show_horizontal_image_row(class_images["highest_images"], captions=class_images["high_activations"])

    st.markdown(
        f"<p style='font-size:{AppConfig.mid_font_size}px; font-weight:bold;'>Low activating class images from train split</p>",
        unsafe_allow_html=True,
    )
    show_horizontal_image_row(class_images["lowest_images"], captions=class_images["low_activations"])


def show_test_sample_images(latent_idx):
    class_idx = st.session_state.class_names.index(st.session_state.selected_class)
    threshold = st.session_state.latent_avg_activations[class_idx]
    class_dict = api.get_images_with_prediction(
        st.session_state.selected_class,
        latent_idx,
        top_k=st.session_state.top_k_sample_for_concept,
        dataset_name=st.session_state.selected_dataset,
        threshold=threshold,
    )
    high_colors = ["#4CAF50" if x else "#FF0000" for x in class_dict["high_correct"]]
    low_colors = ["#4CAF50" if x else "#FF0000" for x in class_dict["low_correct"]]

    st.markdown(
        f"<p style='font-size:{AppConfig.mid_font_size}px; font-weight:bold;'>High activating class images from test split</p>",
        unsafe_allow_html=True,
    )
    if st.session_state.show_masked_images:
        high_images = class_dict["masked_highest_images"]
    else:
        high_images = class_dict["highest_images"]
    show_horizontal_image_row(high_images, captions=class_dict["high_activations"], colors=high_colors)
    st.markdown(
        f"<p style='font-size:{AppConfig.mid_font_size}px; font-weight:bold;'>Low activating class images from test split</p>",
        unsafe_allow_html=True,
    )
    show_horizontal_image_row(class_dict["lowest_images"], captions=class_dict["low_activations"], colors=low_colors)

    st.markdown(
        f"<span style='display:inline-block;width:16px;height:16px;background-color:#4CAF50;margin-right:4px;vertical-align:middle;'></span>"
        f"<span style='font-size:{AppConfig.small_font_size}px;'>Correct</span> &nbsp;&nbsp; "
        f"<span style='display:inline-block;width:16px;height:16px;background-color:#FF0000;margin-right:4px;vertical-align:middle;'></span>"
        f"<span style='font-size:{AppConfig.small_font_size}px;'>Incorrect</span>",
        unsafe_allow_html=True,
    )
    return class_dict


def show_high_low_diff(class_dict):
    high_acc = class_dict["high_acc"]
    low_acc = class_dict["low_acc"]
    mean_acc = class_dict["mean_acc"]
    num_high = class_dict["num_high"]
    num_low = class_dict["num_low"]

    acc_diff_pct = ((high_acc - low_acc) / low_acc) * 100 if low_acc != 0 else 0

    st.markdown(
        f"<p style='font-size:{AppConfig.mid_font_size}px; font-weight:bold;'>High vs Low Group Comparison (Test Split)</p>",
        unsafe_allow_html=True,
    )
    # Side-by-side columns
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**High Group**")
        st.markdown(f"- Accuracy: **{high_acc:.3f}**")
        st.markdown(f"- Count: **{num_high}**")

    with col2:
        st.markdown("**Low Group**")
        st.markdown(f"- Accuracy: **{low_acc:.3f}**")
        st.markdown(f"- Count: **{num_low}**")

    with col3:
        st.markdown("**All Group**")
        st.markdown(f"- Accuracy: **{mean_acc:.3f}**")
        st.markdown(f"- Count: **{num_high + num_low}**")

    st.markdown("---")
    st.markdown(
        f"<p style='color:{'green' if acc_diff_pct > 0 else 'red'};'>"
        f"Relative Accuracy Difference: <b>{acc_diff_pct:+.2f}%</b> (High vs. Low)"
        "</p>",
        unsafe_allow_html=True,
    )
