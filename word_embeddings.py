from read_pdf import *
import gensim
from gensim.models import Word2Vec

if __name__ == '__main__':
    # https://www.geeksforgeeks.org/python-word-embedding-using-word2vec/
    lower_case_tokens = []
    for path in glob.glob('/Users/klara/Downloads/*.pdf'):
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