# extracting and standardizing review texts, 
# tokenizing feedback, 
# dividing the data into training, validation, and test sets.

# asap-aes https://www.kaggle.com/code/nandaprasetia/automated-essay-scoring-mlp-regression
print("\nASAP...\n")
import numpy as np 
import pandas as pd
import torch

from sklearn.model_selection import train_test_split

import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
nltk.download('stopwords')
nltk.download('punkt')
stemmer = PorterStemmer()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s.,\']', '', text)
    tokens = word_tokenize(text)
    stopwords_set = set(stopwords.words('english'))
    tokens = [stemmer.stem(word) for word in tokens if word not in stopwords_set]
    preprocessed_text = ' '.join(tokens)
    return preprocessed_text

def preprocess_text2(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s.,\']', '', text)
    tokens = word_tokenize(text)
    stopwords_set = set(stopwords.words('english'))
    tokens = [stemmer.stem(word) for word in tokens if word not in stopwords_set]
    return tokens

def document_vector(word2vec_model, doc_tokens):
    doc_vector = np.zeros(word2vec_model.vector_size)
    num_words = 0
    for word in doc_tokens:
        try:
            doc_vector += word2vec_model.wv[word]
            num_words += 1
        except KeyError:
            continue
    if num_words != 0:
        doc_vector /= num_words
    return doc_vector

#Load the training,test and validation data
train_df = pd.read_csv("./input/asap-aes/training_set_rel3.tsv",sep='\t', encoding='ISO-8859-1',
                       usecols = ['essay_id', 'essay_set', 'essay','domain1_score']).dropna(axis=1)
preprocess_data=lambda text:preprocess_text(text)
train_df["essay_prepro"]=train_df["essay"].apply(preprocess_data)


X = train_df.drop(["essay_id","essay_set","essay","domain1_score",],axis=1)
y = train_df["domain1_score"]
y = np.asarray(y)


tokenized_documents = [preprocess_text2(doc) for doc in X["essay_prepro"]]


from gensim.models import Word2Vec
ukuran_vektor=100
word2vec_model = Word2Vec(sentences=tokenized_documents, 
                          min_count=1, vector_size=ukuran_vektor,sg=1)

all_words = word2vec_model.wv.index_to_key

X_train,X_val,y_train,y_val = train_test_split(X, y, test_size=0.2,random_state=42)

X_train_vec = np.array([document_vector(word2vec_model, doc.split())
                        for doc in X_train["essay_prepro"]])
X_val_vec = np.array([document_vector(word2vec_model, doc.split()) 
                      for doc in X_val["essay_prepro"]])

# Printing the shapes of the resulting datasets
print("Shape of training data:", X_train_vec.shape)
print("Shape of testing data:", X_val_vec.shape)





# PaperRead
print("\nPaperRead...\n")
from models.Paper import Paper
from models.Review import Review
data_dir = "./input/PeerRead/data/"

# datas = ["arxiv/cs.cl",
#          "arxiv/cs.lg",
#          "arxiv/cs.ai",
#          "nips/nips_2013",
#          "nips/nips_2014",
#          "nips/nips_2015",
#          "nips/nips_2016",
#          "openreview/ICLR.cc_2017_conference",
#          "conll16",
#          "acl17"]

datas = ['']

for data in datas:
    print("preparing dataset ... ", data)
