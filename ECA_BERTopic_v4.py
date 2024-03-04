import os
import re
import nltk
import pandas as pd
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from hdbscan import HDBSCAN
from nltk import WordNetLemmatizer, word_tokenize
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
import gensim.corpora as corpora
from gensim.models.coherencemodel import CoherenceModel
from umap import UMAP
from bertopic.representation import KeyBERTInspired
from ECA_PreModule import collect_material
import streamlit as st

english_stemmer = nltk.stem.SnowballStemmer('english')


def clean_text(text):
    """
    Clean the text by replacing newlines with spaces.
    :param text: The input text to be cleaned.
    :return: The cleaned text with newlines replaced by spaces.
    """
    # Regular expression to match newlines across platforms
    newline_pattern = r"\r?\\n|\\xe|\\x|\\x|\\xa|\\xe"  # Matches carriage return + newline or just newline
    # Replace newlines with spaces
    cleaned_text = re.sub(newline_pattern, "", text)
    # cleaned_text = re.sub(r"[0-9]", " ", cleaned_text)
    return str(cleaned_text)


class StemmedCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedCountVectorizer, self).build_analyzer()
        return lambda doc: ([english_stemmer.stem(w) for w in analyzer(doc)])


def train_bert(docs, model_path):
    model_id = "all-mpnet-base-v2"
    embedding_model = SentenceTransformer(model_id)

    # Dimensionality Reduction
    # umap_model = UMAP(n_neighbors=10, n_components=10, min_dist=0.0, metric='cosine', random_state=101)

    # Clustering model: See [2] for more details
    cluster_model = HDBSCAN(min_cluster_size=5, metric='euclidean', cluster_selection_method='eom',
                            prediction_data=True)

    # vectorizer_model = CountVectorizer(ngram_range=(1, 2),stop_words=stopwords)
    stopwords_list = stopwords.words('english')
    vectorizer_model = StemmedCountVectorizer(analyzer="word", stop_words=stopwords_list, ngram_range=(1, 3))

    # Explicitly define, use, and adjust the ClassTfidfTransformer with new parameters,
    # bm25_weighting and reduce_frequent_words, to potentially improve the topic representation
    ctfidf_model = ClassTfidfTransformer(bm25_weighting=True)

    # BERTopic model
    topic_model = BERTopic(
        # Pipeline models
        embedding_model=embedding_model,
        # umap_model=umap_model,
        hdbscan_model=cluster_model,
        vectorizer_model=vectorizer_model,
        ctfidf_model=ctfidf_model,

        # Hyperparameters
        language="english",
        top_n_words=5,
        nr_topics=5,
        verbose=True
    )

    # Fit the model on a corpus
    topic_model.fit_transform(docs)
    topic_model.save(model_path, serialization="safetensors",
                     save_ctfidf=True, save_embedding_model=model_id)
    return topic_model


def coherence_evaluate(model, model_path, docs, method='u_mass'):
    documents = pd.DataFrame({"Document": docs,
                              "ID": range(len(docs)),
                              "Topic": model.topics_})
    documents_per_topic = documents.groupby(['Topic'], as_index=False).agg({'Document': ' '.join})
    cleaned_docs = model._preprocess_text(documents_per_topic.Document.values)

    # Extract vectorizer and analyzer from BERTopic
    vectorizer = model.vectorizer_model
    analyzer = vectorizer.build_analyzer()

    # Extract features for Topic Coherence evaluation
    words = vectorizer.get_feature_names_out()
    tokens = [analyzer(doc) for doc in cleaned_docs]
    dictionary = corpora.Dictionary(tokens)
    corpus = [dictionary.doc2bow(token) for token in tokens]
    topic_words = [[words for words, _ in model.get_topic(topic)]
                   for topic in range(len(set(model.topics_)) - 1)]

    # Evaluate
    coherence_model = CoherenceModel(topics=topic_words,
                                     texts=tokens,
                                     corpus=corpus,
                                     dictionary=dictionary,
                                     coherence=method
                                     )
    if os.path.exists(model_path):
        coherence_model.load(fname=model_path)
    else:
        coherence_model.save(fname_or_handle=model_path)
    return coherence_model.get_coherence()


def bertopic_model_call(dataset, model_name, calc_coherence=False):
    model_path = 'Models/BERTopic-' + model_name
    results = {'model_name': model_name}
    if os.path.exists(model_path):
        results['model'] = BERTopic.load(model_path)
    else:
        results['model'] = train_bert(dataset, model_path)
    st.write("✅ Model trained successfully")

    results['document_info'] = results['model'].get_document_info(dataset)
    results['topic_info'] = results['model'].get_topic_info()

    results['topic_info']['Percentage'] = round(results['topic_info']['Count'] / results['topic_info']['Count'].sum() * 100, 2)
    results['topic_info'] = results['topic_info'].iloc[:, [0, 1, 3, 2]]
    results['topics'] = results['model'].get_topics()

    # print(results['topic_info'].head())
    # print('*' * 100)

    # for topic_id, topic in results['topics'].items():
    #     print("Topic ID:", topic_id)
    #     print("Words:", topic[0])

    if calc_coherence:
        coherence_res_path = f'Models/CoherenceRES-{model_name}.txt'
        if os.path.exists(coherence_res_path):
            with open(coherence_res_path, 'r') as coh_res_file:
                results['coherence'] = coh_res_file.read()
        else:
            coherence_model_path = 'Models/Coherence-' + model_name
            results['coherence'] = coherence_evaluate(model=results['model'], model_path=coherence_model_path, docs=dataset, method='c_v')
            with open(coherence_res_path, 'w') as coh_res_file:
                coh_res_file.write(str(results['coherence']))
        print('Coherence: ', results['coherence'])
    st.write("✅ Analysis calculated successfully")
    return results
