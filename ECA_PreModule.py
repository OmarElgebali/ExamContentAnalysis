import pandas as pd
from nltk import word_tokenize, WordNetLemmatizer
import streamlit as st
from ECA_TextDescriptor import describe_dataset
from ECA_PdfExtraction import *
from ECA_TextAugmentation import augmentation


def collect_material(folder_path, augmentation_type='Synonym', number_of_augmentation_outputs=1, is_describe=False, is_augment=False):
    """
    Extract documents from PDF files within a folder, caching the result using pickle.
    :param folder_path: The path to the folder containing PDF files.
    :param augmentation_type: The type of text augmentation to apply.
                              Options: 'Synonym', 'word2vec', 'contextualWord'.
                              Default is 'Synonym'.
    :param number_of_augmentation_outputs: Number of augmented outputs to generate for each input.
                                           Default is 1.
    :param is_augment: boolean if augmentation will be used in data or not
    :param is_describe: boolean if descriptor-api will be used in data or not
    :return: A list of extracted text for each PDF file in the folder.
    """
    file_path = extract_last_folder(folder_path)
    docs_raw_file = f'{file_path}-raw.pkl'
    docs_des_file = f'{file_path}-des.pkl'
    docs_aug_file = f'{file_path}-aug-{number_of_augmentation_outputs}-{augmentation_type}.pkl'

    docs_aug_file_path = find_pdf_in_data_folder(docs_aug_file)
    if is_augment and docs_aug_file_path:
        st.write("✅ Data augmented successfully")
        return load_extracted_docs(docs_aug_file_path)

    docs_des_file_path = find_pdf_in_data_folder(docs_des_file)
    if is_describe and docs_des_file_path:
        docs_des = load_extracted_docs(docs_des_file_path)
        st.write("✅ Data described successfully")
        if is_augment:
            # with st.spinner("Augmenting Data"):
            docs_aug = augmentation(docs_des, number_of_outputs=number_of_augmentation_outputs, aug_type=augmentation_type)
            save_extracted_docs(docs_aug_file, docs_aug)
            st.write("✅ Data augmented successfully")
            return docs_aug
        return docs_des

    docs_raw_file_path = find_pdf_in_data_folder(docs_raw_file)
    if docs_raw_file_path:
        docs_raw = load_extracted_docs(docs_raw_file_path)
        st.write("✅ Data extracted successfully")
    else:
        # with st.spinner("Extracting Data"):
        docs_raw = extract_folder_pdf_text(folder_path, True)
        save_extracted_docs(docs_raw_file, docs_raw)
        st.write("✅ Data extracted successfully")

    if is_describe:
        # with st.spinner("Describing Data"):
        docs_des = describe_dataset(docs_raw)
        save_extracted_docs(docs_des_file, docs_des)
        st.write("✅ Data described successfully")
        if is_augment:
            # with st.spinner("Augmenting Data"):
            docs_aug = augmentation(docs_des, number_of_outputs=number_of_augmentation_outputs, aug_type=augmentation_type)
            save_extracted_docs(docs_aug_file, docs_aug)
            st.write("✅ Data augmented successfully")
            return docs_aug
        return docs_des

    if is_augment:
        # with st.spinner("Augmenting Data"):
        docs_aug = augmentation(docs_raw, number_of_outputs=number_of_augmentation_outputs, aug_type=augmentation_type)
        save_extracted_docs(docs_aug_file, docs_aug)
        st.write("✅ Data augmented successfully")
        return docs_aug

    return docs_raw


def clean_text(text):
    # remove urls
    text = re.sub(r"http\S+", " link ", text)

    # replace any digit with num
    text = re.sub(r"\d+", "", text)

    # set space before and after any punctuation
    text = re.sub(r"([^\w\s])", r" \1 ", text)

    # remove extra spaces
    text = re.sub(r"\s+", " ", text)

    words = word_tokenize(text)
    text = " ".join([w for w in words if len(w) > 1])

    return text.lower().strip()


def process_material(folder_path, augmentation_type='Synonym', number_of_augmentation_outputs=1, is_describe=False, is_augment=False):
    dataset = collect_material(
        folder_path=folder_path,
        augmentation_type=augmentation_type,
        number_of_augmentation_outputs=number_of_augmentation_outputs,
        is_describe=is_describe,
        is_augment=is_augment
    )

    clean_docs = [clean_text(raw) for raw in dataset]

    lemmatizer = WordNetLemmatizer()
    filtered_text = [lemmatizer.lemmatize(doc) for doc in clean_docs]

    return filtered_text
