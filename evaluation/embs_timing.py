import hashlib
from multiprocessing import Pool
from elasticsearch import ApiError, ConflictError, Elasticsearch
import base64
from gensim.utils import simple_preprocess
# own modules
from text_embeddings.preprocessing.read_pdf import *
from user_interface.cli import *
from doc_images.pdf_matrix import *
from elasticSearch.queries.query_documents_tfidf import *
from text_embeddings.universal_sent_encoder_tensorFlow import *
from text_embeddings.hugging_face_sentence_transformer import *
from elasticSearch.queries.query_database import *
from elasticsearch.helpers import bulk
from doc_images.PCA.PCA_image_clustering import *
from text_embeddings.TFIDF.preprocessing.TfidfTextPreprocessor import *
from text_embeddings.InferSent.infer_pretrained import *
from text_embeddings import save_models 
from constants import *
from elasticSearch.models_aux import *
from elasticSearch.create_documents import *
from doc_images.convert_pdf2image import *
import sys
from elasticSearch.recursive_search import *

def generate_models_embedding(src_path: str, models : dict, model_name: str = 'no_model'):  
    print('started with generate_models_embedding(), model_name: ', model_name)
    for path in list(scanRecurse(src_path)):
        text = pdf_to_str(path)

        if (model_name in models.keys()) and (model_name != 'ae'):
            try:
                embedding = get_embedding(models=models, model_name=model_name, text=text)
            except Exception as e:
                print('error in embedding: ', path, e)
                continue
        


def get_embedding(models: dict, model_name: str, text: str):
    '''
    :param models: dictionary with model names as keys and the models as values
    :param model_name: name of the model to be used for embedding
    :param text: text to be embedded
    :return: embedding of the text
    '''
    if model_name == "doc2vec": 
        return models['doc2vec'].infer_vector(simple_preprocess(text))
    
    elif model_name in ['google_univ_sent_encoding', 'universal']:
        return embed([text], models['universal'])[0].numpy()
    
    elif model_name in ['huggingface_sent_transformer', 'hugging']:
        if type(text) == str:
            text = [text]
        return models['hugging'].encode(sentences=text)[0]
    
    elif model_name in ['inferSent_AE', 'infer']:
        inferSent_embedding = models['infer'].encode([text], tokenize=True)
        return models['ae'].predict(x=inferSent_embedding)[0]
                    
    elif model_name in ['sim_docs_tfidf', 'tfidf']:
        tfidf_embedding = get_tfidf_emb(models['tfidf'], [text])
        print('tfidf_embedding before ae: ', tfidf_embedding, tfidf_embedding.shape)

        if len(tfidf_embedding) > 2048: # AE to compress data
            tfidf_ae_model = models['tfidf_ae']
            tfidf_embedding = tfidf_embedding.reshape(1, tfidf_embedding.shape[0])
            print('tfidf_embedding after ae: ', tfidf_embedding, tfidf_embedding.shape)
            return tfidf_ae_model.predict(x=tfidf_embedding)[0]
        return tfidf_embedding


def main(baseDir: str, model_names: list = MODEL_NAMES):
    for model_name in model_names:
        print('started with model: ', model_name)
        models = get_models(baseDir, [model_name] if model_name else None)
        generate_models_embedding(src_path=baseDir, models=models, model_name=model_name)
        print('finished model: ', model_name)