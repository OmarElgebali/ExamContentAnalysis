from ECA_PdfExtraction import *

# pdf_path = "Resources/TestDoc2.pdf"
# if has_images(pdf_path):
#     print("The PDF contains images.")
# else:
#     print("The PDF does not contain any images.")
# pdf_text_1 = extract_pdf_text_one_chunk(pdf_path)
# # pdf_text_2 = extract_pdf_text_page_chunks(pdf_path)
#
# print("PDF-1 Text:\n", pdf_text_1)
# print("=" * 300)
# print("PDF-2 Text:\n")
# for i, p in enumerate(pdf_text_2):
#     print(f"Page-{i+1}: {p}\n")

# folder = "Resources/nlp_lectures"
# folder_pdf_text = extract_folder_pdf_text(folder, 1)
# for i, p in enumerate(folder_pdf_text):
#     print(f"Page-{i + 1}: {p}\n")
#
# file_saved = 'nlp_lectures'
# save_extracted_docs(file_saved, folder_pdf_text)
# loaded_docs = load_extracted_docs(file_saved)
#
# print("******************************  Loaded List  ******************************")
# for i, p in enumerate(loaded_docs):
#     print(f"Page-{i + 1}: {p}\n")

folder_path = "Resources/nn_lectures"
extract_docs(folder_path)
