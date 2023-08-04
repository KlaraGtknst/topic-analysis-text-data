from PyPDF2 import PdfReader
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import glob

def term_frequency(tokens, file_name):
    '''
    :param tokens: list of tokens
    :return: None

    This function plots the term frequency of the tokens.
    '''
    plt.figure(figsize=(15, 10))
    plt.title('Term frequency in ' + file_name)
    plt.hist(tokens, bins=100, orientation='vertical', color='green')
    plt.xticks(rotation=90, fontsize=5)
    plt.show()

def pdf_to_str(path):
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

def tokenize(text):
    '''
    :param text: text to be tokenized
    :return: list of tokens

    This function tokenizes the text.
    cf. https://www.pythonpool.com/tokenize-string-in-python/
    '''
    tokens = word_tokenize(text)
    return tokens

def remove_stop_words(tokens):
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

def stemming(tokens):
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
    for path in glob.glob('/Users/klara/Downloads/*.pdf'):
        text = pdf_to_str(path)

        tokens = tokenize(text)
        filtered_tokens = remove_stop_words(tokens)
        stemmed_filtered_tokens = stemming(filtered_tokens)
        term_frequency(stemmed_filtered_tokens, file_name=path.split('/')[-1])
