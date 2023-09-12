from text_embeddings.preprocessing.read_pdf import *
from user_interface.cli import *
import gensim
from gensim.models import Word2Vec

'''------Different word embeddings-------'''

def main(file_paths: list):
    # https://www.geeksforgeeks.org/python-word-embedding-using-word2vec/
    
    lower_case_tokens = []
    for path in file_paths:
        text = pdf_to_str(path)

        tokens = tokenize(text)
        lower_case_tokens.append(list(map(lambda x: x.lower(), tokens)))

    # Create CBOW model of both files
    model1 = gensim.models.Word2Vec(lower_case_tokens, min_count = 1, vector_size = 100, window = 5)     

    # Print results
    print("Cosine similarity between 'bahamas' " + "and 'wealth' - CBOW : ", model1.wv.similarity('bahamas', 'wealth'))
        
    print("Cosine similarity between 'bahamas' " + "and 'credit' - CBOW : ", model1.wv.similarity('bahamas', 'credit'))
    

    # Create Skip Gram model of both files
    model2 = gensim.models.Word2Vec(lower_case_tokens, min_count = 1, vector_size = 100, window = 5, sg = 1)
    
    # Print results
    print("Cosine similarity between 'bahamas' " + "and 'wealth' - Skip Gram : ", model2.wv.similarity('bahamas', 'wealth'))
        
    print("Cosine similarity between 'bahamas' " + "and 'credit' - Skip Gram : ", model2.wv.similarity('bahamas', 'credit'))