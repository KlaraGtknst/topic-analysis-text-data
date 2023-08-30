from TfidfPreprocessingSteps import *

'''
code snippet from https://towardsdatascience.com/elegant-text-pre-processing-with-nltk-in-sklearn-pipeline-d6fe18b91eb8
'''

class TfidfTextPreprocessor():
  def __init__(self):
    pass

  def fit(self, X, y=None):
    return self
  
  def fit_transform(self, X, y=None):
    return self.transform(X)

  def transform(self, X, y=None):
    data = [X] if type(X) == str else X
    txt_preproc = TfidfPreprocessingSteps(data)
    processed_text = \
            txt_preproc.strip_accents().strip_newlines().lowercase().\
            discretize_numbers().remove_punctuations().\
            change_number_encoding().remove_stopwords().\
            lemmatisation().to_text()
    
    return processed_text[0] if type(X) == str else processed_text
  
  