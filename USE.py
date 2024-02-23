from sentence_transformers import SentenceTransformer, util

# Load the Universal Sentence Encoder
model_name = 'paraphrase-MiniLM-L6-v2'  # You can experiment with different pre-trained models
model = SentenceTransformer(model_name)

# Example Questions and Curriculum Texts
question = "France Capital"
# curriculum_text_relevant = "Baguette Eiffel France Louvre"
# curriculum_text_relevant = "Paris"
# curriculum_text_relevant = "Baguette Eiffel Europe Louvre"
curriculum_text_relevant = "Paris is the capital of France."
curriculum_text_non_relevant = "The sun is a star that provides light and heat."
# curriculum_text_non_relevant = "The sun is shinny in Italy."

# Encode the questions and curriculum texts
question_embedding = model.encode(question, convert_to_tensor=True)
curriculum_embedding_relevant = model.encode(curriculum_text_relevant, convert_to_tensor=True)
curriculum_embedding_non_relevant = model.encode(curriculum_text_non_relevant, convert_to_tensor=True)

# Calculate cosine similarity between question and curriculum texts
similarity_relevant = util.pytorch_cos_sim(question_embedding, curriculum_embedding_relevant)[0][0].item()
similarity_non_relevant = util.pytorch_cos_sim(question_embedding, curriculum_embedding_non_relevant)[0][0].item()

# Set a similarity threshold
threshold = 0.7

# Determine relevance based on the threshold
is_relevant_relevant = similarity_relevant >= threshold
is_relevant_non_relevant = similarity_non_relevant >= threshold

# Print Results
print(f"Similarity (Relevant): {similarity_relevant:.4f}")
print(f"Is Relevant? {is_relevant_relevant}")

print(f"\nSimilarity (Non-Relevant): {similarity_non_relevant:.4f}")
print(f"Is Relevant? {is_relevant_non_relevant}")
