# Exam Content Analysis Module (UNRELEASED)

## Overview

The Exam Content Analysis module is designed to analyze a large number of PDFs in specific topics, such as neural networks or image processing. The module follows a step-by-step process to generate a comprehensive dataset for further examination.

## Workflow

1. **PDF Text Extraction:** The module takes a collection of PDFs in specified topics and extracts their text content. If there are images, OCR is applied to convert them into text.

2. **Data Augmentation:** The extracted data is augmented by incorporating synonyms and relevant words, expanding the dataset.

3. **API Integration for Descriptive Text:** Each row in the dataset is sent to an API, which generates more descriptive text without replacing the original content. This enhances the dataset within the boundaries of the specified topics.

4. **BERTopic Model:** The dataset is fed into the BERTopic model, which applies embeddings, dimensionality reduction, and clustering. Topics are represented by centroids or a centroid with its closer points.

5. **Syllabus Determination Options:**
   - **Option 1:** Generate a topic from a given question, compare it with the model-generated topics using similarity metrics (e.g., USE or Siamese network), and determine syllabus inclusion based on a threshold.
   - **Option 2:** Transform the question using the model's approach and check if it corresponds to a generated topic. Syllabus inclusion is based on whether the transformed question belongs to the range [0, n-1] or is an outlier topic -1.
