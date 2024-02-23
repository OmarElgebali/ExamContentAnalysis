import nltk
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from hdbscan import HDBSCAN
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import fetch_20newsgroups

english_stemmer = nltk.stem.SnowballStemmer('english')


class StemmedCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedCountVectorizer, self).build_analyzer()
        return lambda doc: ([english_stemmer.stem(w) for w in analyzer(doc)])


def train_bert(docs, model_path):
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    # Clustering model: See [2] for more details
    cluster_model = HDBSCAN(min_cluster_size=20, metric='euclidean', cluster_selection_method='eom',
                            prediction_data=True)

    # Explicitly define, use, and adjust the ClassTfidfTransformer with new parameters,
    # bm25_weighting and reduce_frequent_words, to potentially improve the topic representation
    ctfidf_model = ClassTfidfTransformer(bm25_weighting=True)
    # vectorizer_model = CountVectorizer(ngram_range=(1, 2),stop_words=stopwords)
    stopwords_list = stopwords.words('english')
    vectorizer_model = StemmedCountVectorizer(analyzer="word", stop_words=stopwords_list, ngram_range=(1, 3))

    # BERTopic model
    topic_model = BERTopic(embedding_model=embedding_model, hdbscan_model=cluster_model, ctfidf_model=ctfidf_model,
                           vectorizer_model=vectorizer_model, language="english", top_n_words=20)

    # Fit the model on a corpus
    topics, probs = topic_model.fit_transform(docs)
    topic_model.save(model_path)
    return topic_model


# docs = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))['data']
#
# print(len(docs))

docs = [
    "State the difference between following:\n A. Activation and Objective Function(Mention Examples)\n B. Training and Generalization Error\n C. Feedforward and Recurrent Neural Network"
    , "Calculate the padding used in the first layer?"
    , "If we used a Fully Connected Layer instead of the first Convolutional Layer, how many neurons are needed?"
    , "Yolo object detector is trained to detect up to 3 overlapped bounding boxes of 10 classes. The input image is divided into 7x7 cells. What is the dimensionality of the output layer tensor?"
    , "Given a two-input neuron with the following parameters: b = 1.2, w=[3 2], and X=[1 2], calculate the neuron output for the following transfer functions:\n 1. A symmetrical Threshold transfer function\n 2. A linear transfer function"
    , "Different Types of Artificial Neural Networks"
    , "How do Neural Networks Work?"
    , "What is Perceptron Learning Algorithm?"
    , "What is Delta Rule?"
    , "What is the Back Propagation Learning Algorithm?"
    , "What is a Neural Network?"
    , "Applications Of ANN Or NN"
    , "What do you mean by Perceptron?"
    , "What are the different types of Perceptrons?"
    , "Neural Network (NN) and Artificial Intelligence (AI)"
    , "What do you mean by Cost Function?"
    , "Activation Functions used in Neural Networks"
    , "What is the use of Loss Functions?"
    , "What are Weights and Biases in Neural Networks?"
    , "What is Hyperparameters or Hyperparameters Tuning?"
    , "Backpropagation Algorithm & Error Function"
    , "What is Data Normalization?"
    , "What are Feedforward Neural Networks?"
    , "What are Recurrent Neural Networks (RNN)?"
    , "What are Convolutional Neural Networks (CNN)?"
    , "What is AutoEncoders?"
    , "What do you mean by Boltzmann Machine?"
    , "What are Optimizers?"
    , "Epoch, Batch, and Iteration in Neural Networks"
    , "What is the difference between a FNN and RNN?"
    , "What is the curse of Dimensionality, and how can it be addressed in neural networks?"
    , "What is deep learning, and how is it different from other types of machine learning?"
    , "What is reinforcement learning, and how is it used in neural networks?"
    , "What is the difference between a convolutional neural network and a fully connected neural network?"
    , "What are some applications of neural networks in industry?"
    , "What is the difference between supervised and unsupervised learning?"
    , "What is the role of activation functions in neural networks?"
    , "What is overfitting, and how can it be prevented?"
    , "What is transfer learning, and how is it used in neural networks?"
    , "What is a loss function, and how is it used in neural networks?"
    , "What is a dropout layer, and how is it used in neural networks?"
    , "What is batch normalization, and how is it used in neural networks?"
    , "What is a generative adversarial network (GAN)?"
    , "What is a deep belief network (DBN)?"
    , "What is the vanishing gradient problem, and how can it be addressed?"

    , "What is Digital Signal Processing (DSP) and how does it differ from analog signal processing?"
    , "Can you explain the fundamental concepts of sampling and quantization in DSP?"
    , "How is the Fourier Transform used in DSP, and what is its significance?"
    , "What role does the Z-transform play in the analysis and processing of discrete-time signals?"
    , "How does DSP contribute to the enhancement of audio and image signals?"
    , "What are the key differences between FIR (Finite Impulse Response) and IIR (Infinite Impulse Response) filters in DSP?"
    , "How is the concept of convolution applied in the context of DSP?"
    , "Can you elaborate on the applications of DSP in telecommunications and speech processing?"
    , "What is the significance of the Fast Fourier Transform (FFT) algorithm in DSP?"
    , "How do digital filters contribute to noise reduction and signal enhancement in DSP?"
    , "What are the challenges and techniques involved in real-time implementation of DSP algorithms?"
    , "Discuss the importance of DSP in modern audio processing systems, such as music production and digital audio effects."
    , "How is DSP utilized in the design and implementation of digital communication systems?"
    , "Explain the concept of signal representation using complex numbers in the context of DSP."
    , "What are the advantages of using DSP processors in comparison to general-purpose processors for signal processing tasks?"

    , "Describe and analyze the processes by which a distinctively American identity was created and changed over time. What historical forces and events helped to forge this identity? In what specific ways has this identity been contested? Has this identity been broad and inclusive, or has it tended to create un-American or non-American “Others?”"
    , "For decades now, US History survey classes have been divided by the Civil War with the first half covering up to 1865 and the second half covering 1865 to the present.  Discuss why you think historians settled upon this demarcation and why it persists to this day.  What are some other turning points in American history that historians have emphasized or should emphasize?  How might these other turning points influence where new American history surveys begin and end?"
    , "One way in which Americans have tended to differ from people in other cultures is our relative rootlessness. Migrations and movement have punctuated American history. Outline the key eras of popular mobility, and how these were similar to or different from each other (who moved, where they moved, why they moved). What have been the implications (social, political, economic) for American history as a whole? Throughout history, powerful elites have dominated (socially, politically, economically). How do the relatively less powerful and the truly disenfranchised (as variously defined, depending on the era and region in question) fit into history? To what degree are they merely victims or puppets of the elite? To what extent are they active shapers of history? Explain the most salient examples over time."
    , "Some historians view wars as catalysts for profound social, political, and economic change. Others believe that wars entrench the status quo. Looking at the major wars fought by the US (Indian wars may be considered collectively as a single war), with which side (if either) do you most agree, and why? [If you agree with neither, what conclusions would you draw about the impact of wars, and why?"
    , "Identify and examine key social, political, cultural and economic movements in American history.  To what extent did each of these movements promote reform, rebellion, or both reform and rebellion?  Be clear in the ways you define reform and rebellion, pointing out how your definitions might differ from that of other Americans. How and why have certain Americans tended to view themselves and their nation as “exceptional”?  How has this notion shaped American culture, politics and economics?"
    , "Which has been more significant in shaping American history: race or class? [Note: It is acceptable to argue that the significance of race or class has varied in different eras, but make sure you give concrete evidence."
    , "Design a new American History survey course.Write a course description explaining what you hope to cover and what you hope the class will learn from this course.What key people, places and events would you highlight?  Why would you highlight these?  What books and/or other readings would you use when developing lectures and discussions?  What books and/or other sources would you have the class read?  Defend and explain these choices."
    , "How have historians tended to define “politics” over the course of American history? When and why have these definitions changed and varied?  What impact do these definitions have on how historians have approached and written about the nation’s past? Europeans have long defined themselves in vis-à-vis “internal others” (i.e. minority groups within Europe, such as (but not limited to) heretics, immigrants, and Jews). Explain how Europeans have defined themselves in distinction from these internal “others” in two time periods (defined above). How has this process shaped European identity?"
    , "European history can be divided and understood in many ways. Write a history of Europe in two periods (defined above) using environmental and technological OR scientific, intellectual, and cultural events as dividing points or moments of change [Note: on the exam, only one of these sets of themes will be offered; please prepare answers for both sets]. How does this approach redefine European history in significant ways? What is revealed when we periodize history according to these turning points rather than more traditional moments of change defined by politics, religion, or war?"
    , "In discussing nationalism, historians are divided over whether or not people had any sort of national identity prior to the nineteenth century—many say that it had its roots in much earlier periods. Do you agree? What other kinds of identity competed with national identity in two periods (defined above) of European history?"
    , "A complex relationship exists between secular power and sacred authority in European history—a relationship of competition, cooperation, and ultimately separation. Focusing on two periods of history (defined above), argue for the dominance of one power over the other as the driving factor in European history."
    , "European history is often conceived of in terms of renaissances and revolutions. Identify what are commonly thought of as revolutions and renaissances in two different periods of European history (as defined above). What characterizes these renaissances and revolutions? Are they, in fact, renaissances and revolutions? Why or why not? Defining Europe has long vexed historians, politicians, philosophers, others. Present-day debates over membership in the EU attest to this conundrum. Address the development of “Europe” in two different historical periods (defined above). How have its geopolitical, geographical, and cultural parameters shifted? What do these shifting assignations signify? How are they useful for historians? How do they illuminate or obscure important themes in European development?"
    , "Some historians views wars as catalysts for profound social, political, and economic change. Others believe that wars entrench the status quo. Looking at the major wars in two periods of European history , with which side (if any) do you most agree and why?"
    , "Examine the role of institutions of authority (i.e. religious, political, economic, and cultural institutions) in the lives of ordinary people (i.e. citizens, subjects, and estates) in two different periods of European history (defined above). What is the balance of power between institutions and individuals? How are individual lives shaped by institutions of authority? Has the balance changed over time? If so, how?"
    , "Contact with people beyond European borders (however those are defined) has long been a feature of European history. How have Europeans redefined their understandings of themselves and the world as a result of global exploration and trade? How have the expansion and contractions of empires affected European self-understandings? Please consider this question across two major periods of European history (defined above)"
    , "How have people, individuals, or groups who have not had formal institutional power (i.e. because of their religion, gender, class, or economic or social status) shaped events in two major periods of European history (defined above)? How significant is their informal power? What accounts for its relative strength or weakness?"
]

model_path = 'OldModels/bert_model_NN'

topic_model = BERTopic.load(model_path)
# topic_model = train_bert(docs, model_path)


docs_df = topic_model.get_document_info(docs)
docs_df.to_csv('topic_model - get_document_info - BERT_NN.csv')
print(docs_df)
print('*' * 100)

freq_df = topic_model.get_topic_info()
freq_df.to_csv('topic_model - get_topic_info - BERT_NN.csv')

print("Number of topics: {}".format(len(freq_df)))
print('*' * 100)

print(freq_df)
print('*' * 100)

freq_df['Percentage'] = round(freq_df['Count'] / freq_df['Count'].sum() * 100, 2)
freq_df.to_csv('topic_model - get_topic_info_PERCENTAGE - BERT_NN.csv')
freq_df = freq_df.iloc[:, [0, 1, 3, 2]]
print(freq_df.head())
print('*' * 100)

topics = topic_model.get_topics()
for topic_id, topic in topics.items():
    print("Topic ID:", topic_id)
    print("Words:", topic[0])

# fig1 = topic_model.visualize_topics()
# fig1.show()
#
# fig2 = topic_model.visualize_barchart()
# fig2.show()
