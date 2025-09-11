import streamlit as st

import src.demo.frontend.api as api
import src.demo.frontend.ui as ui
import src.demo.frontend.utils as utils
from src.demo.config import ConceptListViewConfig


def init_state():
    st.session_state.selected_dataset = None
    st.session_state.selected_class = None
    st.session_state.selected_concept = 0
    st.session_state.refresh_key = 0


def load_class_distribution(
    selected_dataset,
    selected_class,
):
    class_idx = st.session_state.class_names.index(selected_class)

    concept_distribution = api.get_concept_distribution(class_idx, selected_dataset)

    st.session_state.selected_class = selected_class
    st.session_state.selected_dataset = selected_dataset
    concept_distribution = utils.process_concept_distribution(concept_distribution)
    st.session_state.concept_distribution = concept_distribution


if __name__ == "__main__":
    ui.set_page()
    init_state()
    with st.sidebar:
        ui.set_config_view()

    selected_dataset, selected_class = ui.render_option_view()
    if selected_class != st.session_state.selected_class or selected_dataset != st.session_state.selected_dataset:
        load_class_distribution(selected_dataset, selected_class)

    with st.container(border=True):
        ui.render_class_view()

    with st.container(border=True):
        col1, col2 = st.columns([1, 4])
        with col1:
            with st.container(border=True, height=ConceptListViewConfig.height):
                ui.render_concepts_list()
        with col2:
            ui.render_concept_info()
