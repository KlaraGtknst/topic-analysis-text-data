import re
import string
import pdb # for debugging

from sklearn.pipeline import Pipeline
from text_embeddings.preprocessing.read_pdf import *
from user_interface.cli import *
from gensim.utils import simple_preprocess
import pandas as pd
import numpy as np
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
import itertools
from sklearn.metrics.pairwise import cosine_similarity
import unidecode
from nltk.stem import WordNetLemmatizer
from text_embeddings.TFIDF.preprocessing.TfidfTextPreprocessor import *
from text_embeddings.TFIDF.preprocessing.TfidfPreprocessingSteps import *

'''------Code to find documents most fitting for input query-------
run this code by typing and altering the path:
    python3 query_documents_tfidf.py -d '/Users/klara/Downloads/*.pdf'
    python3 query_documents_tfidf.py -d '/Users/klara/Documents/Uni/bachelorarbeit/data/0/*.pdf'
'''

def preprocess_text(path: str) -> list:
    '''
    :param path: path to pdf file
    :return: list of preprocessed tokens
    
    This function preprocesses the text from a pdf file.
    The preprocessing steps are:
        1. extract text from pdf file, make it lowercase and tokenize it, ignoring very short and very long tokens
        2. remove stop words
        3. stem tokens
    '''
    tokens = simple_preprocess(pdf_to_str(path))
    filtered_tokens = remove_stop_words(tokens)
    stemmed_filtered_tokens = stemming(filtered_tokens)
    return stemmed_filtered_tokens

def get_vocab_word_per_doc(df_clean_token: dict) -> tuple:
    '''
    :param df_clean_token: dictionary with document id as key and list of tokens as value
    :return: vocabulary: list of all unique tokens of all documents
        words_per_doc: dictionary with document id as key and dictionary with token as key and number of occurences of token in document as value
    '''
    vocabulary = set() # contains all unique tokens of all documents
    words_per_doc = {}  # contains the number of occurences of each token in each document
    for doc in df_clean_token:
        words_per_doc[doc] = {token: df_clean_token[doc].count(token) for token in set(df_clean_token[doc])}
        vocabulary.update(df_clean_token[doc])
    vocabulary = list(vocabulary)
    return vocabulary, words_per_doc

def get_tfidf_per_doc(tfidf: TfidfVectorizer, doc_num: int, document_term_matrix: np.ndarray) -> pd.DataFrame:
    '''
    :param tfidf: trained tf-idf model
    :param doc_num: document number (starting at index 0)
    :param document_term_matrix: document term matrix, entries are tf-idf values of tokens per document
    :return: pandas data frame with tf-idf values for each token in a specific document doc_num, whereas the tokens are sorted by their tf-idf value. The row's name is the token.
    
    DO NOT USE for representation of document as vector, since the return value is sorted and therefore not suitable as an identifier.
    for more information cf. https://kavita-ganesan.com/tfidftransformer-tfidfvectorizer-usage-differences/
    '''
    # place tf-idf values in a pandas data frame 
    tfidf_document_vector = document_term_matrix[doc_num]
    tfidf_per_token = pd.DataFrame(tfidf_document_vector.T, index=tfidf.get_feature_names_out(), columns=["tfidf"]) 
    tfidf_per_token.sort_values(by=["tfidf"], ascending=False, inplace=True)
    return tfidf_per_token


def print_info_abt_doc_term_mat_model(document_term_matrix: np.ndarray, tfidf: TfidfVectorizer) -> None:
    '''
    :param document_term_matrix: document term matrix, entries are tf-idf values of tokens per document
    :param tfidf: trained tf-idf model
    :return: None

    prints information about the document term matrix and the tf-idf model
    '''
    print(document_term_matrix)
    print(document_term_matrix[(0,tfidf.vocabulary_['redit'])])
    print(tfidf.vocabulary_)
    print(tfidf.idf_)
    print(tfidf.get_feature_names_out())

def get_docs_from_file_paths(file_paths: list) -> list:
    '''
    :param file_paths: list of file paths
    :return: list of documents, where each document is a (not further processed) string of the pdf file's text
    '''
    docs = []
    for path in file_paths:
        docs.append(pdf_to_str(path))
    return docs

def get_preprocessed_tokens_from_file_paths(file_paths: list) -> dict:
    '''
    :param file_paths: list of file paths
    :return: dictionary with document id as key and list of preprocessed tokens as value
    '''
    df_clean_token = {} # contains preprocessed tokens for each document
    for path in file_paths:
        df_clean_token[path] = preprocess_text(path)
    return df_clean_token

def print_cosine_similarity_examples(transformed_query: np.ndarray, document_term_matrix: np.ndarray):
    '''
    :param transformed_query: tf-idf vector of query string
    :param document_term_matrix: document term matrix, entries are tf-idf values of tokens per document; 
        use return of tfidf.(fit_)transform(docs)- don't use .todense()!
    :return: None

    prints the cosine similarity between documents in the training corpus and the query string, as well as between documents in the training corpus.
    '''
    # similarity between documents (https://goodboychan.github.io/python/datacamp/natural_language_processing/2020/07/17/04-TF-IDF-and-similarity-scores.html)
    kernel_matrix = cosine_similarity(document_term_matrix, document_term_matrix)   
    print('similarity between documents in trainings corpus:\n',kernel_matrix)
    kernel_matrix = cosine_similarity(document_term_matrix, transformed_query)
    # TODO: do not understand matrix below
    print('similarity between documents in trainings corpus and query:\n',kernel_matrix)

def print_tfidf_transformation_example(tfidf: TfidfVectorizer,query: str = 'human readable Bahamas credit system'):
    '''
    :param tfidf: trained tf-idf model
    :param query: query string
    :return: None
    
    prints the transformation of the query string to a tf-idf vector.
    The default query string is 'human readable Bahamas credit system'.
    It contains tokens that are not in the vocabulary'''
    t = tfidf.transform([query])
    print('transformation to (document number, token encoding) tf-idf score\n', t)
    return t
    

def get_num_all_zero_tfidf_embeddings(sim_docs_document_term_matrix: TfidfVectorizer, file_paths: list = None):
    '''
    :param sim_docs_document_term_matrix: document term matrix, entries are tf-idf values of tokens per document
    :param file_paths: list of file paths; default: None; if set, the file path of the document with all zero tf-idf values is printed
    :return: None
    
    This method prints the number of documents, which are represented as all zero tf-idf values.
    This may be due to the fact, that there are no tokens in the document, which are in the vocabulary.
    The vocabulary has to be limited in order to reduce the dimensionality of the vector space.
    '''
    count = 0
    for i in range(len(sim_docs_document_term_matrix)):
        if np.array([entry  == 0 for entry in sim_docs_document_term_matrix[i]]).all():
            if file_paths is not None:
                print(f'{file_paths[i]} is all zero')
            count += 1
    print(f'number of documents with all zero tf-idf values: {count} from {len(sim_docs_document_term_matrix)}')

def show_preprocessing_steps(sample_text='The résultat: 123 people, 123456 CATS and 123.45 pizzas!\n'):
    sample = sample_text
    #preProc = TfidfTextPreprocessor()
    preProcSteps = TfidfPreprocessingSteps([sample])
    preProcSteps.strip_accents()
    print('accents: ', preProcSteps.X)
    preProcSteps.strip_newlines()
    print('new lines: ', preProcSteps.X)
    preProcSteps.lowercase()
    print('lowercase: ', preProcSteps.X)
    preProcSteps.discretize_numbers()
    print('discretize numbers: ', preProcSteps.X)
    preProcSteps.remove_punctuations()
    print('punctations: ', preProcSteps.X)
    preProcSteps.change_number_encoding()
    print('change encoding: ', preProcSteps.X)
    preProcSteps.remove_stopwords()
    print('stopwords: ', preProcSteps.X)
    preProcSteps.lemmatisation()
    print('lemmatisation: ', preProcSteps.X)
    print('result: ', preProcSteps.to_text())

def main(file_paths):

    docs = get_docs_from_file_paths(file_paths)

    # custom preprocessor
    # usage of uni-grams only, n_gram (n>1) increases vocabulary size (bad), but does not reduce number of zero tf-idf document embeddings (bad)
    '''tfidf = TfidfVectorizer(input='content', preprocessor=TfidfTextPreprocessor().fit_transform, min_df=3, max_df=int(len(docs)*0.07))
    sim_docs_document_term_matrix = tfidf.fit_transform(docs).todense()
    get_num_all_zero_tfidf_embeddings(sim_docs_document_term_matrix, file_paths)
    print('vocabulary: ', tfidf.get_feature_names_out(), '\nnumber of elements of vocabulary: ', len(tfidf.get_feature_names_out()))
    
    # add flag ('column') which indicates if document is all zero tf-idf vector
    print(sim_docs_document_term_matrix.shape)
    flags = np.array([1 if np.array([entry  == 0 for entry in sim_docs_document_term_matrix[i]]).all() else 0 for i in range(len(sim_docs_document_term_matrix))]).reshape(len(sim_docs_document_term_matrix),1)
    flag_matrix = np.append(sim_docs_document_term_matrix, flags, axis=1)
    print(flag_matrix.shape)
    get_num_all_zero_tfidf_embeddings(flag_matrix, file_paths)'''

    # preprocess example
    show_preprocessing_steps()
        
