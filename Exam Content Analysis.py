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
        col1_checks.checkbox("Apply Descriptor", key='ck_descriptor')
        col2_checks.checkbox("Apply Augmentor", key='ck_augmentor')
        col3_checks.checkbox("Calculate Coherence", key='ck_coherence')
        if st.session_state.ck_augmentor:
            col1_aug, col2_aug = st.columns(2)
            col1_aug.number_input("Number of Augmentations:", step=1, min_value=1, max_value=100,
                                  key='number_of_augmentation_outputs')
            col2_aug.selectbox("Type of Augmentation", options=['Synonym', 'word2vec', 'contextualWord'],
                               key='augmentation_type')
        else:
            st.session_state.augmentation_type = ''
            st.session_state.number_of_augmentation_outputs = 0

        submit_material_btn = st.button("Start Training")
        if submit_material_btn:
            if file:
                folder_path = extract_last_folder(search_file(file.name))
                print(f"@ ck_descriptor: {st.session_state.ck_descriptor}")
                print(f"@ ck_augmentor: {st.session_state.ck_augmentor}")
                print(f"@ augmentation_type: {st.session_state.augmentation_type}")
                print(f"@ number_of_augmentation_outputs: {st.session_state.number_of_augmentation_outputs}")
                with st.status("Training BERTopic ...", expanded=True) as status:
                    dataset = process_material(
                        folder_path=folder_path,
                        is_describe=st.session_state.ck_descriptor,
                        is_augment=st.session_state.ck_augmentor,
                        augmentation_type=st.session_state.augmentation_type,
                        number_of_augmentation_outputs=st.session_state.number_of_augmentation_outputs
                    )
                    st.session_state.bertopic_dict = bertopic_model_call(
                        dataset=dataset,
                        model_name=txt_model_name,
                        calc_coherence=st.session_state.ck_coherence
                    )

                    status.update(label="Training Completed!", state="complete", expanded=False)
                # """################################# Visualization ###########################################"""

                # """################################# Transform ###############################################"""
                # print("#" * 200)
                # doc = " What is the concept of Digital Signal Processing"
                # print(doc)
                # print("*" * 200)
                #
                # _topic, _prob = bertopic_dict['model'].transform([doc])
                # print("Topic: ", _topic, "Prob: ", _prob)
                # print("*" * 200)
                st.toast("Model Trained Successfully :D")
            else:
                st.warning("No Material Uploaded !!")

    with tab2:
        st.subheader("Visualization Section")
        try:
            st.markdown(f"BERTopic Model Name *(Course Name)*: {st.session_state.bertopic_dict['model_name']}")
            fig_visualize_topics = st.session_state.bertopic_dict['model'].visualize_topics()
            fig_visualize_barchart = st.session_state.bertopic_dict['model'].visualize_barchart()
            # col1_visualize, col2_visualize = st.columns(2)
            st.write(fig_visualize_topics)
            st.divider()
            st.write(fig_visualize_barchart)
        except AttributeError:
            st.error("Model not learned yet")

    with tab3:
        st.subheader("Analytics Section")
        try:
            st.markdown(f"BERTopic Model Name *(Course Name)*: {st.session_state.bertopic_dict['model_name']}")
            st.markdown("#### Coherence")
            st.write("Coherence calculated for this model is", float(st.session_state.bertopic_dict['coherence']))
            # st.divider()
            st.markdown("#### Document Info")
            st.write(st.session_state.bertopic_dict['document_info'])
            # st.divider()
            st.markdown("#### Topic Info")
            st.write(st.session_state.bertopic_dict['topic_info'])
            # st.divider()
            st.markdown("#### Topics")
            st.write(st.session_state.bertopic_dict['topics'])
        except AttributeError:
            st.error("Model not learned yet")

    with tab4:
        st.subheader("Prediction Section")
        try:
            st.markdown(f"BERTopic Model Name *(Course Name)*: {st.session_state.bertopic_dict['model_name']}")
            txt_question = st.text_area("Provide your question")
            predict_btn = st.button("Classify Question")
            if predict_btn:
                _topic, _prob = st.session_state.bertopic_dict['model'].transform([str(txt_question)])
                if _topic[0] == -1:
                    st.info("Question not covered in given material")
                else:
                    st.success("Question covered")
                print(st.session_state.bertopic_dict['model'].get_topic(_topic[0]))
                print("*" * 200)
                print(_topic)
                print("*" * 200)
                print(_prob)
                print("*" * 200)

        except AttributeError:
            st.error("Model not learned yet")
