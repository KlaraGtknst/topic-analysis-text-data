import sys
from constants import MODEL_NAMES
from text_embeddings import save_models
import numpy as np


def get_models(src_path: str, model_names: list = MODEL_NAMES):
    '''
    src_path: path to the directory of all documents to be inserted into the database
    model_names: names of the models to be used for embedding
    return: dictionary with model names as keys and the models as values
    '''
    print('started with get_models() ')
    #sys.stdout.flush()
    models = {}
    if 'infer' in model_names and (not 'ae' in model_names):    # needs AE for embedding
        model_names = model_names + ['ae']
    if 'tfidf' in model_names and (not 'tfidf_ae' in model_names):  # needs AE for embedding on large corpus
        model_names = model_names + ['tfidf_ae']
    for model_name in model_names:
        try: # model exists
            print('try to load model_name: ', model_name)
            model = save_models.load_model(model_name)
            models[model_name] = model
            print('model loaded: get_models() ')
            #sys.stdout.flush()
        except: # model does not exist, create and save it
            print('model not saved, train model: get_models() ', model_name)
            #sys.stdout.flush()
            model = save_models.train_model(model_name, src_path)
            models[model_name] = model
            save_models.save_model(model, model_name)
    return models

def get_tfidf_emb(tfidf_model, text:list):
    tfidf_emb = tfidf_model.transform(text).todense()
    flag = 1 if np.array([entry  == 0 for entry in tfidf_emb]).all() else 0
    if tfidf_emb.shape[0] > 1:
        tfidf_emb = tfidf_emb.reshape(1, -1)
    flag_matrix = np.append(tfidf_emb, np.array(flag).reshape(1,1), axis=1)
    embedding = np.ravel(np.array(flag_matrix))
    return embedding