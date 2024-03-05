import time

import streamlit as st
from ECA_PreModule import *
from ECA_BERTopic_v4 import bertopic_model_call

if __name__ == '__main__':
    st.title("Exam Content Analysis *(User-Side)*")

    # st.subheader("Train BERTopic")
    txt_model_name = st.text_input("Course Name")
    file = st.file_uploader("Upload Materials", type=["pdf"],
                            help="Upload file and will use file's folder")
    submit_material_btn = st.button("Submit Material")
    if submit_material_btn:
        if file and txt_model_name:
            folder_path = extract_last_folder(search_file(file.name))
            with st.spinner("Analysing Materials ...") as status:
                dataset = process_material(
                    folder_path=folder_path,
                    # is_describe=st.session_state.ck_descriptor,
                    is_describe=False,
                    is_augment=1,
                    # augmentation_type=st.session_state.augmentation_type,
                    augmentation_type='Synonym',
                    number_of_augmentation_outputs=1
                )
                st.session_state.bertopic_dict = bertopic_model_call(
                    dataset=dataset,
                    model_name=txt_model_name,
                    calc_coherence=0
                )

            st.toast("Model Trained Successfully", icon="✔️")
            st.toast("Ready to Classify", icon="💭")
        else:
            st.warning("Material not uploaded or Model name not provided")

    try:
        st.divider()
        txt_question = st.text_area("Provide your question")
        predict_btn = st.button("Classify Question")
        if predict_btn:
            _topic, _prob = st.session_state.bertopic_dict['model'].transform([str(txt_question)])
            if _topic[0] == -1:
                st.info("Question not covered in given material", icon="❌")
            else:
                st.success("Question covered", icon="✔️")
    except AttributeError:
        pass
