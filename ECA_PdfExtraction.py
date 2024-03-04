import io
import os
import pickle

import PIL
import PyPDF2
import cv2
import pytesseract
from PIL import Image
import re
import fitz  # PyMuPDF library


def search_file(file_name):
    current_folder = os.getcwd()  # Get the current working directory

    for folder_name, _, filenames in os.walk(current_folder):
        if file_name in filenames:
            return folder_name

    return None


def extract_last_folder(path):
    """
    Extract the last folder from a given path.
    :param path: The path from which to extract the last folder.
    :return: The name of the last folder in the path.
    """
    last_folder = os.path.split(path)[-1]
    return last_folder


def find_file_in_current_folder(filename):
    """
    Find a file in the current working directory.
    :param filename: The name of the file to search for.
    :return: The absolute path to the file if found, otherwise None.
    """
    current_folder = os.getcwd()  # Get the current working directory
    files_in_current_folder = os.listdir(current_folder)

    if filename in files_in_current_folder:
        return os.path.join(current_folder, filename)
    else:
        return None  # File not found in the current folder


def find_pdf_in_data_folder(filename):
    """
    Find a file in the 'Models/Data/' subdirectory of the current working directory.
    :param filename: The name of the file to search for.
    :return: The absolute path to the file if found, otherwise None.
    """
    current_folder = os.getcwd()  # Get the current working directory
    data_folder = os.path.join(current_folder, 'Data')  # Path to 'Models/Data/' folder

    files_in_data_folder = os.listdir(data_folder)

    if filename in files_in_data_folder:
        return os.path.join(data_folder, filename)
    else:
        return None  # File not found in the 'Models/Data/' folder


def clean_text(text):
    """
    Clean the text by replacing newlines with spaces.
    :param text: The input text to be cleaned.
    :return: The cleaned text with newlines replaced by spaces.
    """
    # Regular expression to match newlines across platforms
    newline_pattern = r"\r?\\n|\\xe|\\x|\\x|\\xa|\\xe"  # Matches carriage return + newline or just newline
    # Replace newlines with spaces
    cleaned_text = re.sub(newline_pattern, " ", text)
    # cleaned_text = re.sub(r"[0-9]", " ", cleaned_text)
    return str(cleaned_text)


def has_images(pdf_path):
    """
    Check if a PDF document contains images.
    :param pdf_path: The path to the PDF document.
    :return: True if the PDF contains images, False otherwise.
    """
    try:
        doc = fitz.open(pdf_path)
        for page_num in range(doc.page_count):
            page = doc[page_num]
            images = page.get_images(full=True)
            if images:
                return True
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False


def extract_pdf_text_one_chunk(pdf_path):
    """
    Extract text from a PDF document considering both text and images.
    :param pdf_path: The path to the PDF document.
    :return: Extracted text from the PDF, or None in case of an error.
    """

    with open(pdf_path, 'rb') as pdf_file:
        reader = PyPDF2.PdfReader(pdf_file)

        if not has_images(pdf_path):
            # print("******************** Text *********************")
            text = ""
            for page in reader.pages:
                text += page.extract_text()
            return clean_text(str(text.encode('utf-8')))
        else:
            # print("******************** Image *********************")
            try:
                doc = fitz.open(pdf_path)
                extracted_text = ""
                for page_num in range(doc.page_count):
                    page = doc[page_num]
                    images = page.get_images(full=True)
                    if images:
                        for img_index, img in enumerate(images):
                            try:
                                xref = img[0]
                                base_image = doc.extract_image(xref)
                                image_data = base_image["image"]
                                img = PIL.Image.open(io.BytesIO(image_data))
                                # img.show()
                                ocr_text = pytesseract.image_to_string(img)
                                extracted_text += ocr_text
                            except Exception as e:
                                print(f"Error in image processing: {e}")
                    extracted_text += page.get_text("text")
                return clean_text(str(extracted_text.encode('utf-8')))
            except Exception as e:
                print(f"Error in extract_pdf_text: {e}")
                return None


def extract_pdf_text_page_chunks(pdf_path):
    """
    Extract text from a PDF document page by page, considering both text and images.
    :param pdf_path: The path to the PDF document.
    :return: A list of extracted text for each page, or None in case of an error.
    """
    with open(pdf_path, 'rb') as pdf_file:
        reader = PyPDF2.PdfReader(pdf_file)

        if not has_images(pdf_path):
            # print("******************** Text *********************")
            pages_text = []
            for page in reader.pages:
                pages_text.append(clean_text(str(page.extract_text().encode('utf-8'))))
            return pages_text
        else:
            # print("******************** Image *********************")
            try:
                doc = fitz.open(pdf_path)
                pages_extracted_text = []
                for page_num in range(doc.page_count):
                    page = doc[page_num]
                    images = page.get_images(full=True)
                    extracted_text = ""
                    if images:
                        for img_index, img in enumerate(images):
                            try:
                                xref = img[0]
                                base_image = doc.extract_image(xref)
                                image_data = base_image["image"]
                                img = PIL.Image.open(io.BytesIO(image_data))
                                # img.show()
                                ocr_text = pytesseract.image_to_string(img)
                                extracted_text += ocr_text
                            except Exception as e:
                                print(f"Error in image processing: {e}")
                    extracted_text += page.get_text("text")
                    pages_extracted_text.append(clean_text(str(extracted_text.encode('utf-8'))))
                return pages_extracted_text
            except Exception as e:
                print(f"Error in extract_pdf_text: {e}")
                return None


def extract_folder_pdf_text(folder_path, is_chunks):
    """
    Extract text from PDF files within a folder.
    :param folder_path: The path to the folder containing PDF files.
    :param is_chunks: A boolean indicating whether to extract text in chunks (page by page) or not.
    :return: A list of extracted text for each PDF file in the folder.
    """
    folder_output = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        file_output = extract_pdf_text_page_chunks(file_path) if is_chunks else extract_pdf_text_one_chunk(file_path)
        folder_output.extend(file_output)
    return folder_output


def save_extracted_docs(file_path, docs):
    """
    Save extracted documents to a file using pickle.
    :param file_path: The path to the file where documents will be saved.
    :param docs: The documents to be saved.
    """
    with open('Data/' + file_path, 'wb') as file:
        pickle.dump(docs, file)


def load_extracted_docs(file_path):
    """
    Load extracted documents from a file using pickle.
    :param file_path: The path to the file from which documents will be loaded.
    :return: The loaded documents.
    """
    with open(file_path, 'rb') as file:
        docs = pickle.load(file)
    return docs
