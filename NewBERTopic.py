import nltk
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.datasets import fetch_20newsgroups
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
from bertopic.representation import KeyBERTInspired

# Built Modules
from ECA_PdfExtraction import extract_folder_pdf_text

english_stemmer = nltk.stem.SnowballStemmer('english')


class StemmedCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedCountVectorizer, self).build_analyzer()
        return lambda doc: ([english_stemmer.stem(w) for w in analyzer(doc)])


def clean_text(text):
    # remove urls
    text = re.sub(r"http\S+", " link ", text)

    # replace any digit with num
    # remove digits
    text = re.sub(r"\d+", "", text)

    # set space before and after any punctuation
    text = re.sub(r"([^\w\s])", r" \1 ", text)

    # remove extra spaces
    text = re.sub(r"\s+", " ", text)

    words = word_tokenize(text)
    text = " ".join([w for w in words if len(w) > 1])

    return text.lower().strip()


def train_bert(docs, model_path):
    model_id = "all-MiniLM-L6-v2"
    embedding_model = SentenceTransformer(model_id)
    news_embeddings = embedding_model.encode(docs, show_progress_bar=True)

    # Dimintionality Rwductuion
    umap_model = UMAP(n_neighbors=15, n_components=15, min_dist=0.0, metric='cosine', random_state=101)

    # Clustering model (A higher min_cluster_size will generate fewer topics - A lower min_cluster_size will generate more topics.)
    cluster_model = HDBSCAN(min_cluster_size=20, metric='euclidean', cluster_selection_method='eom',
                            prediction_data=True)

    # vectorizer_model = CountVectorizer(ngram_range=(1, 2),stop_words=stopwords)
    stopwords_list = stopwords.words('english')
    vectorizer_model = StemmedCountVectorizer(analyzer="word", stop_words=stopwords_list, ngram_range=(1, 3), min_df=2,
                                              max_df=0.95)

    # Explicitly define, use, and adjust the ClassTfidfTransformer with new parameters,
    # bm25_weighting and reduce_frequent_words, to potentially improve the topic representation
    ctfidf_model = ClassTfidfTransformer(bm25_weighting=True)

    # Topic Representer
    keybert_model = KeyBERTInspired()

    # BERTopic model
    topic_model = BERTopic(
        # Pipeline models
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=cluster_model,
        vectorizer_model=vectorizer_model,
        ctfidf_model=ctfidf_model,
        representation_model=keybert_model,

        # Hyperparameters
        language="english",
        top_n_words=20,
        verbose=True
    )

    # Fit the model on a corpus
    topics, probs = topic_model.fit_transform(docs, news_embeddings)
    topic_model.save(model_path, serialization="s-afetensors",
                     save_ctfidf=True, save_embedding_model=model_id)
    return topic_model


model_path = 'Models/NewBERTModel'

docs = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))['data']

# docs_path = "Resources/nn_lectures"
# docs = extract_folder_pdf_text(docs_path, 1)

# clean_docs = clean_text(str(docs))

# topic_model = train_bert(docs, model_path)
topic_model = BERTopic.load(model_path)

print(topic_model.get_topic_info())
print("*" * 200)

topics = topic_model.get_topics()
for topic_id, topic in topics.items():
    print("Topic ID:", topic_id)
    print("Words:", topic[0])


""""""""""""""""""""""""""""""
# loaded_model = BERTopic.load(model_path)
#
# doc = " It's like your\nbrakes, something you don't want to take chances with. I waited too long\nto take care of my front tire once and it went flat on me, doing 70 MPH\ngoing down the grapevine towards Bakersfield.  At that instance, I would\nof given any amount of money for a new tire."
# new_doc = clean_text(doc)
# print(new_doc)
# print("*"*200)
#
# _topic, _prob = loaded_model.transform([new_doc])
#
# print("Topic: ", _topic, "Prob: ", _prob)
# print("*"*200)
# print(loaded_model.get_topic(_topic[0]))
