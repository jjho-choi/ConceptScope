import streamlit as st

import src.demo.frontend.api as api
import src.demo.frontend.ui as ui
import src.demo.frontend.utils as utils
from src.demo.config import ConceptListViewConfig


def init_state():
    if "selected_dataset" not in st.session_state:
        st.session_state.selected_dataset = None
    if "selected_class" not in st.session_state:
        st.session_state.selected_class = None
    if "selected_concept" not in st.session_state:
        st.session_state.selected_concept = 0


if __name__ == "__main__":
    ui.set_page()
    init_state()
    with st.sidebar:
        selected_dataset = ui.set_config_view()

    if selected_dataset != st.session_state.selected_dataset:
        st.session_state.selected_dataset = selected_dataset
        st.session_state.selected_class = None
        base_info = api.load_base_info(selected_dataset)
        utils.save_base_info(base_info)

    with st.container(border=True):
        ui.render_class_view()

    with st.container(border=True):
        col1, col2 = st.columns([1, 4])
        with col1:
            with st.container(border=True, height=ConceptListViewConfig.height):
                ui.render_concepts_list()
        with col2:
            ui.render_concept_info()
