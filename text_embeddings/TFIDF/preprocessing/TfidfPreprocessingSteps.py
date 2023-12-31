import re
import string
import nltk
from nltk.corpus import wordnet as wn
import unidecode
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

'''
code snippet from https://towardsdatascience.com/elegant-text-pre-processing-with-nltk-in-sklearn-pipeline-d6fe18b91eb8
'''

def download_if_non_existent(res_path, res_name):
        try:
            nltk.data.find(res_path)
        except LookupError:
            nltk.download(res_name)

class TfidfPreprocessingSteps():
    def __init__(self, X):
        self.X = X
        download_if_non_existent('corpora/stopwords', 'stopwords')
        download_if_non_existent('tokenizers/punkt', 'punkt')
        download_if_non_existent('corpora/wordnet', 'wordnet')
        self.sw_nltk = set(stopwords.words('english'))

    def strip_accents(self):
        # more information about unidecode: https://www.geeksforgeeks.org/how-to-remove-string-accents-using-python-3/
        self.X = [unidecode.unidecode(text) for text in self.X]
        return self
    
    def strip_newlines(self):
        self.X = [text.replace('\n', ' ') for text in self.X]
        return self
    
    def lowercase(self):
        self.X = [text.lower() for text in self.X]
        return self
    
    def discretize_numbers(self):
        # replace numbers with SMALLNUMBER, BIGNUMBER, FLOAT
        # more information about regex: https://www.bogotobogo.com/python/python_regularExpressions.php
        self.X = [re.sub(r'\d{1,5}', 'SMALLNUMBER', re.sub(r'\d{6,}', 'BIGNUMBER', (re.sub(r'\d+\.\d+', 'FLOAT', text)))) for text in self.X]
        return self
    
    def remove_punctuations(self):
        # '!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~' 32 punctuations in python string module
        self.X = [re.sub('[%s]' % re.escape(string.punctuation), ' ', text) for text in self.X]
        return self
    
    def remove_double_spaces(self):
        self.X = [re.sub(r'\W+', ' ', text) for text in self.X] # matches any character which is not a word character
        return self

    def change_number_encoding(self):
        self.X = [re.sub('FLOAT', '<FLOAT>', re.sub('SMALLNUMBER', '<SMALLNUMBER>', (re.sub('BIGNUMBER', '<BIGNUMBER>', text)))) for text in self.X]
        return self
    
    def remove_stopwords(self):  
        stop_words = set(stopwords.words('english'))
        self.X = [[w for w in text.split(' ') if w not in stop_words] for text in self.X]
        return self
    
    def lemmatisation(self):
        lemmatizer = WordNetLemmatizer()
        self.X = [[lemmatizer.lemmatize(w) for w in text] for text in self.X]
        return self
    
    def to_text(self) -> list:
        self.X = [' '.join(text) for text in self.X]
        return self.X


    