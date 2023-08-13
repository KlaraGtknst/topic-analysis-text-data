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

def gen_vector_T(tokens, vocabulary, tfidf):
    # Create a vector for Query/search keywords
    Q = np.zeros((len(vocabulary)))
    x= tfidf.transform(tokens)
    #print(tokens[0].split(','))
    for token in tokens:
        #print(token)
        try:
            ind = vocabulary.index(token)
            Q[ind]  = x[0, tfidf.vocabulary_[token]]
        except:
            pass
    return Q

def cosine_sim(a, b):
    cos_sim = np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))
    return cos_sim

def cosine_similarity_T(k, query, df_clean_token, vocabulary, tfidf):
    # TODO: do not use voacb (list of filenmames), but df_clean_token.values (list of tokens)
    # Cosine Similarity b/w document to query function
    preprocessed_query = re.sub("\W+", " ", query).strip() # remove one or more non-alphanumeric characters
    tokens = word_tokenize(str(preprocessed_query))
    #q_df = pd.DataFrame(columns=['q_clean'])
    #q_df.loc[0,'q_clean'] = tokens
    #print(type(q_df.loc[0,'q_clean']))
    #print('\n' , q_df['q_clean'].values.tolist()[0])
    #q_df['q_clean'] = list(stemming(remove_stop_words(q_df.q_clean.values.tolist()[0])))
    #rint(q_df)
    d_cosines = []
    
    query_vector = gen_vector_T(stemming(remove_stop_words(tokens)), vocabulary, tfidf)
    #gen_vector_T(q_df['q_clean'])
    for d in tfidf_tran.A:
        print(d, query_vector)
        print(cosine_sim(query_vector, d))
        d_cosines.append(cosine_sim(query_vector, d))
                    
    out = np.array(d_cosines).argsort()[-k:][::-1]
    #print("")
    d_cosines.sort()
    print(d_cosines)
    a = pd.DataFrame()
    #print(df_clean_token)
    #print(out)
    for i,index in enumerate(out):
        #print(i, index)
        a.loc[i,'index'] = str(index)
        #print(df_clean_token[file_paths[index]])
        a.loc[i,'Subject'] = file_paths[index]
    for j,simScore in enumerate(d_cosines[-k:][::-1]):
        a.loc[j,'Score'] = simScore
    return a

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


if __name__ == '__main__':
    args = arguments()
    file_paths = get_input_filepath(args)

    df_clean_token = {} # contains preprocessed tokens for each document
    docs = []
    for path in file_paths:
        df_clean_token[path] = preprocess_text(path)
        docs.append(pdf_to_str(path))

    vocabulary, words_per_doc = get_vocab_word_per_doc(df_clean_token)
    #print('vocab', vocabulary)
    #print('words_per_doc', words_per_doc)

    # tfIdf model
    tfidf = TfidfVectorizer(input='content', lowercase=True, analyzer='word', stop_words='english', token_pattern="\w+")
    document_term_matrix = tfidf.fit_transform(docs)
    #print(document_term_matrix)
    #print(document_term_matrix[(0,tfidf.vocabulary_['redit'])])
    #print(tfidf.vocabulary_)
    #print(tfidf.idf_)
    #print(tfidf.get_feature_names_out())

    df = get_tfidf_per_doc(tfidf, 1)
    #print(df)

    # document vectorization
    D = np.zeros((len(file_paths), len(list(tfidf.vocabulary_.values()))))    # tf-idf matrix; entry: tf-idf value of token (collumn/ second entry in access tuple) in document (row/ first index in access tuple)
    #print(list(range(len(file_paths))),list(tfidf.vocabulary_.values()))
    for comb in itertools.product(list(range(len(file_paths))), list(tfidf.vocabulary_.values())):
        D[comb] = document_term_matrix[comb]

    print(f'tfidf of {list(tfidf.vocabulary_.keys())[10]} in the first Document is {D[0,list(tfidf.vocabulary_.values())[10]]}')
    #print(D[0])
    
    # document search engine using TF-IDF
    #print(cosine_similarity_T(10,'lion com', df_clean_token, vocabulary, tfidf))

