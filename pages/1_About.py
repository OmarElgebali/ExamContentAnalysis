import streamlit as st

info = """
# Exam Content Analysis Module

## Overview

The Exam Content Analysis module is designed to analyze a large number of PDFs in specific topics, such as neural networks or image processing. The module follows a step-by-step process to generate a comprehensive dataset for further examination.

---
## Phases

### **1. PDF Text Extraction:** 
The module takes a collection of PDFs in specified topics and extracts their text content. If there are images, OCR is applied to convert them into text.

### **2. Data Augmentation:** 
The extracted data is augmented by incorporating synonyms and relevant words, expanding the dataset.

### **3. API Integration for Descriptive Text: *(Under-Maintenance)*** 
Each row in the dataset is sent to an API, which generates more descriptive text without replacing the original content. This enhances the dataset within the boundaries of the specified topics.

### **4. BERTopic Model:** 
The dataset is fed into the BERTopic model, which applies embeddings, dimensionality reduction, and clustering. Topics are represented by centroids or a centroid with its closer points.

### **5. Syllabus Determination Options:**
Transform the question using the model's approach and check if it corresponds to a generated topic. Syllabus inclusion is based on whether the transformed question belongs to the range `[0, n-1]` or is an outlier topic `-1`.

---
## Run

To start the UI run the following command line:
```console
streamlit run '.\Exam Content Analysis.py'
```

---

## Workflow

### *Inputs*

##### [1] Course Name
- Used to save the 2 models `(BERTopic and Coherence)` so to load it later.

##### [2] PDF file
- The file must be in a directory in your current folder.
- **Example:** Current directory `/project/` and file named `lecture.pdf` is in `/project/course/`.

##### [3] Descriptor
- Checker that determine whether to use *Descriptor API* (ex: ChatGPT) to Augment the data.\n
- > **Note:** Currently disabled due to maintenance.

##### [4] Augmentor
- Checker that determine whether to use *Substitute Augmentation* to Augment the data.
- 1. **Number of Augmentations `L`:** increases dataset of `N` rows to `N*(L+1)`.
- 2. **Augmentation Type:** selection between 3 types:
   - 1. `word2vec`: using Google-news model.
   - 2. `contextualWord`: using BERT model.
   - 3. `Synonym`: using PPDB Model.
- > **Note:** (Augmentation Type) currently disabled with value due to maintenance.

##### [5] Calculate Coherence
- Checker that determine whether to calculate *Coherence* in Analytics tab or not.

##

### *Outputs*
If the model is trained then the another 3 tabs will be available.

##### [1] Visualizations:
- 1. Visualize Topics
- 2. Visualize Documents

##### [2] Analytics:
- 1. Coherence 
- 2. Document Info
- 3. Topic Info
- 4. Topics

##### [3] Prediction:
Takes a question and determine whether the question is covered in the material learned or not.
"""

st.write(info)
