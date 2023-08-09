from cli import *
from PyPDF2 import PdfReader
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import matplotlib.pyplot as plt

'''------Code to extract text from PDF files and preprocess it-------
run this code by typing and altering the path:
    python3 code_file.py -i '/Users/klara/Downloads/SAC2-12.pdf'
    python3 code_file.py -i '/Users/klara/Downloads/SAC2-12.pdf' '/Users/klara/Downloads/SAC1-6.pdf'
    python3 code_file.py -d '/Users/klara/Downloads/*.pdf'
'''

def pdf_to_str(path: str) -> str:
    '''
    :param path: path to pdf file
    :return: text from pdf file

    This function extracts the text from a pdf file.
    cf. https://pypi.org/project/PyPDF2/
    '''
    reader = PdfReader(path)

    text = ''
    for page in reader.pages:
        text += page.extract_text()

    return text

def tokenize(text: str) -> list:
    '''
    :param text: text to be tokenized
    :return: list of tokens

    This function tokenizes the text.
    cf. https://www.pythonpool.com/tokenize-string-in-python/
    '''
    tokens = word_tokenize(text)
    return tokens

def remove_stop_words(tokens: list) -> list:
    '''
    :param tokens: list of tokens
    :return: list of lower case tokens without stop words

    This function first converts the words to lower case and then removes stop words from the list of tokens.
    cf. https://www.geeksforgeeks.org/removing-stop-words-nltk-python/
    '''
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [w for w in tokens if not w.lower() in stop_words]
    return filtered_tokens

def stemming(tokens: list) -> list:
    '''
    :param tokens: list of tokens
    :return: list of stemmed tokens

    This function stems the tokens.
    cf. https://www.datacamp.com/tutorial/stemming-lemmatization-python
    cf. https://www.geeksforgeeks.org/python-stemming-words-with-nltk/
    '''
    ps = PorterStemmer()
    stemmed_filtered_tokens = []
    #print("{0:20}{1:20}".format("--Word--","--Stem--"))
    for word in tokens:
        stemmed_filtered_tokens.append(ps.stem(word))
        #print ("{0:20}{1:20}".format(word, stemmed_filtered_tokens[-1]))

    return stemmed_filtered_tokens


if __name__ == '__main__':
    args = arguments()
    file_paths = get_filepath(args)

    for path in file_paths:
        text = pdf_to_str(path)

        tokens = tokenize(text)
        filtered_tokens = remove_stop_words(tokens)
        stemmed_filtered_tokens = stemming(filtered_tokens)
