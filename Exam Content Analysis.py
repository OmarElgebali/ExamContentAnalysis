import time

import streamlit as st
from ECA_PreModule import *
from ECA_BERTopic_v4 import bertopic_model_call

if __name__ == '__main__':
    st.title("Exam Content Analysis")

    tab1, tab2, tab3, tab4 = st.tabs(["Train", "Visualize", "Analytics", "Predict"])
    with tab1:
        st.subheader("Train BERTopic")
        txt_model_name = st.text_input("BERTopic Model Name *(Course Name)*")
        file = st.file_uploader("Upload Materials", type=["pdf"],
                                help="Upload File and will train the model on file's folder")
        col1_checks, col2_checks, col3_checks = st.columns(3)
        col1_checks.checkbox("Apply Descriptor", key='ck_descriptor', disabled=True)
        col2_checks.checkbox("Apply Augmentor", key='ck_augmentor')
        col3_checks.checkbox("Calculate Coherence", key='ck_coherence')
        if st.session_state.ck_augmentor:
            col1_aug, col2_aug = st.columns(2)
            col1_aug.number_input("Number of Augmentations:", step=1, min_value=1, max_value=100,
                                  key='number_of_augmentation_outputs')
            col2_aug.selectbox("Type of Augmentation", options=['contextualWord', 'Synonym', 'word2vec'],
                               key='augmentation_type', disabled=True)
        else:
            st.session_state.augmentation_type = ''
            st.session_state.number_of_augmentation_outputs = 0

        submit_material_btn = st.button("Start Training")
        if submit_material_btn:
            if file:
                folder_path = extract_last_folder(search_file(file.name))
                with st.status("Training BERTopic ...", expanded=True) as status:
                    dataset = process_material(
                        folder_path=folder_path,
                        # is_describe=st.session_state.ck_descriptor,
                        is_describe=False,
                        is_augment=st.session_state.ck_augmentor,
                        # augmentation_type=st.session_state.augmentation_type,
                        augmentation_type='contextualWord',
                        number_of_augmentation_outputs=st.session_state.number_of_augmentation_outputs
                    )
                    st.session_state.bertopic_dict = bertopic_model_call(
                        dataset=dataset,
                        model_name=txt_model_name,
                        calc_coherence=st.session_state.ck_coherence
                    )

                    status.update(label="Training Completed!", state="complete", expanded=False)
                st.toast("Model Trained Successfully", icon="✔️")
            else:
                st.warning("No Material Uploaded")

    with tab2:
        st.subheader("Visualization Section")
        try:
            st.markdown(f"BERTopic Model Name *(Course Name)*: {st.session_state.bertopic_dict['model_name']}")
            fig_visualize_topics = st.session_state.bertopic_dict['model'].visualize_topics()
            st.write(fig_visualize_topics)
            st.divider()
            fig_visualize_barchart = st.session_state.bertopic_dict['model'].visualize_barchart()
            st.write(fig_visualize_barchart)
            st.toast("Visualization Generated Successfully", icon="📈")
        except AttributeError:
            st.error("Model not learned yet")

    with tab3:
        st.subheader("Analytics Section")
        try:
            st.markdown(f"BERTopic Model Name *(Course Name)*: {st.session_state.bertopic_dict['model_name']}")
            if 'coherence' in st.session_state.bertopic_dict:
                st.markdown("#### Coherence")
                st.write("Coherence calculated for this model is", float(st.session_state.bertopic_dict['coherence']))
            st.markdown("#### Document Info")
            st.write(st.session_state.bertopic_dict['document_info'])
            st.markdown("#### Topic Info")
            st.write(st.session_state.bertopic_dict['topic_info'])
            st.markdown("#### Topics")
            st.write(st.session_state.bertopic_dict['topics'])
            st.toast("Analytics Generated Successfully", icon="📊")
        except AttributeError:
            st.error("Model not learned yet")

    with tab4:
        st.subheader("Prediction Section")
        try:
            st.markdown(f"BERTopic Model Name *(Course Name)*: {st.session_state.bertopic_dict['model_name']}")
            st.toast("Ready to Predict", icon="💭")
            txt_question = st.text_area("Provide your question")
            predict_btn = st.button("Classify Question")
            if predict_btn:
                _topic, _prob = st.session_state.bertopic_dict['model'].transform([str(txt_question)])
                if _topic[0] == -1:
                    st.info("Question not covered in given material")
                else:
                    st.success("Question covered")
        except AttributeError:
            st.error("Model not learned yet")
