from read_pdf import *
from cli import *
from gensim.utils import simple_preprocess
import pandas as pd
import numpy as np
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
import itertools
from sklearn.metrics.pairwise import cosine_similarity

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
    tfidf_per_token = pd.DataFrame(tfidf_document_vector.T.todense(), index=tfidf.get_feature_names_out(), columns=["tfidf"]) 
    tfidf_per_token.sort_values(by=["tfidf"], ascending=False, inplace=True)
    return tfidf_per_token

def get_tfidf_matrix(file_paths: list, tfidf: TfidfVectorizer, document_term_matrix: np.ndarray) -> np.ndarray:
    '''
    DEPRECATED, use to dense instead: https://hackernoon.com/document-term-matrix-in-nlp-count-and-tf-idf-scores-explained
    :param file_paths: list of file paths
    :param tfidf: trained tf-idf model
    :param document_term_matrix: document term matrix, entries are tf-idf values of tokens per document
    :return: numpy array with tf-idf values for each token in every document. The row's identifier is the document and the columns' identifier is the token's encoding from the tfidf model.
    
    This function creates document vectorization using tf-idf model's token encoding.
    The return value is a tf-idf matrix with the following structure:
        - rows: documents (first index in access tuple))
        - columns: tokens (second index in access tuple))
        - entries: tf-idf value of tokens

    for more information cf. https://towardsdatascience.com/tf-idf-for-document-ranking-from-scratch-in-python-on-real-world-dataset-796d339a4089
    '''
    D = np.zeros((len(file_paths), len(list(tfidf.vocabulary_.values()))))  
    for comb in itertools.product(list(range(len(file_paths))), list(tfidf.vocabulary_.values())):
        D[comb] = document_term_matrix[comb]
    return D

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
    :param document_term_matrix: document term matrix, entries are tf-idf values of tokens per document
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

if __name__ == '__main__':
    args = arguments()
    file_paths = get_input_filepath(args)

    #df_clean_token = get_preprocessed_tokens_from_file_paths(file_paths)
    docs = get_docs_from_file_paths(file_paths)

    #vocabulary, words_per_doc = get_vocab_word_per_doc(df_clean_token)
    #print('vocab', vocabulary)
    #print('words_per_doc ', words_per_doc)

    # tfIdf model
    # use min/ max_df to filter out tokens that appear in too many/ too few documents -> reduce vector dimensionality
    '''tfidf = TfidfVectorizer(input='content', max_df=int(len(docs)*0.25), min_df=7, lowercase=True, analyzer='word', stop_words='english', token_pattern="\w+")
    #tfidf = tfidf.fit(docs)
    print(tfidf.get_feature_names_out())

    document_term_matrix = tfidf.fit_transform(docs)    # format: (document, token encoding) tf-idf score -> use D (below) to access tf-idf values
    # print_info_abt_doc_term_mat_model(document_term_matrix, tfidf)

    # returns tf-idf values for the first document with token human readable, but SORTED (≠ document vectorization)
    print(get_tfidf_per_doc(tfidf, 0, document_term_matrix))
    #print(get_tfidf_per_doc(tfidf, 1, document_term_matrix))

    D = get_tfidf_matrix(file_paths, tfidf, document_term_matrix)
    print(f'tfidf of {list(tfidf.vocabulary_.keys())[10]} in the first Document is {D[0,list(tfidf.vocabulary_.values())[10]]}')
    # returns tf-idf values for the first document with token NOT human readable
    print('tf-idf of first document: ', D[0])

    # document search engine using TF-IDF and cosine similarity
    #transformed_query = print_tfidf_transformation_example(tfidf=tfidf, query='human readable Bahamas credit system')
 
    #print_cosine_similarity_examples(transformed_query=transformed_query, document_term_matrix=document_term_matrix)'''

    
    # max_features: top frequent words -> not suitable for our use case, cf. https://stackoverflow.com/questions/46118910/scikit-learn-vectorizer-max-features
    # no more numbers in vocabulary, only words, cf. https://stackoverflow.com/questions/51643427/how-to-make-tfidfvectorizer-only-learn-alphabetical-characters-as-part-of-the-vo
    # usage of uni-grams only
    sim_docs_tfidf = TfidfVectorizer(input='content', lowercase=True, min_df=3, max_df=int(len(docs)*0.07), analyzer='word', stop_words='english', token_pattern=r'(?u)\b[A-Za-z]+\b')
    # usage of n_gram increases vocabulary size (bad), but does not reduce number of zero tf-idf document embeddings (bad)
    #sim_docs_tfidf = TfidfVectorizer(input='content', lowercase=True, ngram_range=(1,3), min_df=3, max_df=int(len(docs)*0.07), analyzer='word', stop_words='english', token_pattern=r'(?u)\b[A-Za-z]+\b')
    sim_docs_tfidf = sim_docs_tfidf.fit(docs)
    print('max df of vocabulary: ', int(len(docs)*0.04))  # == 7
    print('vocabulary: ', sim_docs_tfidf.get_feature_names_out(), '\nnumber of elements of vocabulary: ', len(sim_docs_tfidf.get_feature_names_out()))

    # to dense: https://hackernoon.com/document-term-matrix-in-nlp-count-and-tf-idf-scores-explained
    sim_docs_document_term_matrix = sim_docs_tfidf.fit_transform(docs).todense()

    # all zero tf-idf document embeddings
    get_num_all_zero_tfidf_embeddings(sim_docs_document_term_matrix, file_paths)