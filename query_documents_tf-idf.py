from read_pdf import *
from cli import *
from gensim.utils import simple_preprocess
import pandas as pd
import numpy as np
import os 
import re
import operator
import nltk 
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
import seaborn as sns
import itertools
from itertools import product

'''------Code to find documents most fitting for input query-------
run this code by typing and altering the path:
    python3 query_documents_tf-idf.py -d '/Users/klara/Downloads/*.pdf'
    python3 query_documents_tf-idf.py -d '/Users/klara/Documents/Uni/bachelorarbeit/data/0/*.pdf'
'''

def preprocess_text(path):
    tokens = simple_preprocess(pdf_to_str(path))
    filtered_tokens = remove_stop_words(tokens)
    stemmed_filtered_tokens = stemming(filtered_tokens)
    return stemmed_filtered_tokens

def get_vocab_word_per_doc(df_clean_token):
    vocabulary = set() # contains all unique tokens of all documents
    words_per_doc = {}  # contains the number of occurences of each token in each document
    for doc in df_clean_token:
        words_per_doc[doc] = {token: df_clean_token[doc].count(token) for token in set(df_clean_token[doc])}
        vocabulary.update(df_clean_token[doc])
    vocabulary = list(vocabulary)
    return vocabulary, words_per_doc

def get_tfidf_per_doc(tfidf, doc_num):
    # place tf-idf values in a pandas data frame 
    tfidf_first_document_vector = document_term_matrix[doc_num]
    df = pd.DataFrame(tfidf_first_document_vector.T.todense(), index=tfidf.get_feature_names_out(), columns=["tfidf"]) 
    df.sort_values(by=["tfidf"],ascending=False, inplace=True)
    return df

def get_tf_idf_per_doc(file_paths, tfidf, document_term_matrix):
    # document vectorization using tf-idf model's token encoding
    # tf-idf matrix; entry: tf-idf value of token (collumn/ second entry in access tuple) in document (row/ first index in access tuple)
    D = np.zeros((len(file_paths), len(list(tfidf.vocabulary_.values()))))  
    for comb in itertools.product(list(range(len(file_paths))), list(tfidf.vocabulary_.values())):
        D[comb] = document_term_matrix[comb]
    return D

def print_info_abt_doc_term_mat(document_term_matrix):
    print(document_term_matrix)
    print(document_term_matrix[(0,tfidf.vocabulary_['redit'])])
    print(tfidf.vocabulary_)
    print(tfidf.idf_)
    print(tfidf.get_feature_names_out())

def get_docs_from_file_paths(file_paths):
    docs = []
    for path in file_paths:
        docs.append(pdf_to_str(path))
    return docs

def get_preprocessed_tokens_from_file_paths(file_paths):
    df_clean_token = {} # contains preprocessed tokens for each document
    for path in file_paths:
        df_clean_token[path] = preprocess_text(path)
    return df_clean_token

if __name__ == '__main__':
    args = arguments()
    file_paths = get_input_filepath(args)

    #df_clean_token = get_preprocessed_tokens_from_file_paths(file_paths)
    docs = get_docs_from_file_paths(file_paths)

    #vocabulary, words_per_doc = get_vocab_word_per_doc(df_clean_token)
    #print('vocab', vocabulary)
    #print('words_per_doc ', words_per_doc)

    # tfIdf model
    tfidf = TfidfVectorizer(input='content', lowercase=True, analyzer='word', stop_words='english', token_pattern="\w+")
    document_term_matrix = tfidf.fit_transform(docs)
    #print_info_abt_doc_term_mat(document_term_matrix)

    # returns tf-idf values for the first document with token human readable
    #print(get_tfidf_per_doc(tfidf, 1))

    D = get_tf_idf_per_doc(file_paths, tfidf, document_term_matrix)
    print(f'tfidf of {list(tfidf.vocabulary_.keys())[10]} in the first Document is {D[0,list(tfidf.vocabulary_.values())[10]]}')
    # returns tf-idf values for the first document with token NOT human readable
    #print('tf-idf of first document: ', D[0])

    # document search engine using TF-IDF
    #print(cosine_similarity_T(10,'lion com', df_clean_token, vocabulary, tfidf))

