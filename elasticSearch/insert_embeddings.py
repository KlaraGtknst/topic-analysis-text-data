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

class wrapper:
    def __init__(self, model_name: str, baseDir: str):
        self.model_name = model_name
        self.baseDir = baseDir

    def __call__(self, src_paths):
        insert_embedding(src_path=self.baseDir, src_paths=src_paths, model_name=self.model_name)

def get_src_paths(src_path: str, src_paths: list):
    if src_paths == []:
        return scanRecurse(src_path)    # local
    else:
        return src_paths    # parallel using Pool server

def generate_models_embedding(src_path: str, src_paths: list, models : dict, client: Elasticsearch, model_name: str = 'no_model'):  
    print('started with generate_models_embedding(), model_name: ', model_name)
    sys.stdout.flush()
   
    for path in get_src_paths(src_path=src_path, src_paths=src_paths):
    
        text = pdf_to_str(path)
        id = get_hash_file(path)

        if (model_name in models.keys()) and (model_name != 'ae'):
            try:
                embedding = get_embedding(models=models, model_name=model_name, text=text)
            except Exception as e:
                print('error in embedding: ', path, e)
                continue
        
            client.update(index='bahamas', id=id, body={'doc': {MODELS2EMB[model_name]: embedding}})


def insert_embedding(src_path: str, src_paths: list, models: dict={}, client_addr=CLIENT_ADDR, client: Elasticsearch=None, model_name: str = 'no_model'):
        '''
        :param src_path: path to the directory of the documents to be inserted into the database
        :param client: Elasticsearch client
        :param models: dictionary with model names as keys and the models as values
        :param model_names: names of the models to be used for embedding

        inserts specific embedding of all documents into the database 'bahamas'.
        '''
        print('started with insert_embedding() ')
        sys.stdout.flush()

        if (model_name not in MODEL_NAMES) or (model_name == 'ae') or (model_name == 'none'):
            return
        
        client = client if client else Elasticsearch(client_addr, timeout=1000)
        models = models if models else get_models(src_path, [model_name] if model_name else None)
        print('got models: ', models.values())

        try:
            generate_models_embedding(src_paths=src_paths, src_path=src_path, models=models, model_name=model_name, client=client)
        except (ConflictError, ApiError,EOFError) as err:
            print('error', err)
            return

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


def main(src_path: str, client_addr=CLIENT_ADDR, model_names: list = MODEL_NAMES, num_cpus:int=1):
    print('start inserting documents embeddings')

    if num_cpus == 1:   # FIXME: Pool does not work locally, even if num_cpus = 1
        for model_name in model_names:
            print('started with model: ', model_name)
            insert_embedding(src_path = src_path, src_paths=[], client_addr=client_addr, model_name = model_name)
            print('finished model: ', model_name)

    else:   # server uses parallelization (Pool)
        document_paths = list(scanRecurse(src_path))
        print('number of docs: ', len(document_paths))
        sub_lists = list(chunks(document_paths, int(len(document_paths)/num_cpus)))
        print('obtained sublists')

        # process n_cpus sublists
        with Pool(processes=num_cpus) as pool:
            for model_name in model_names:  # function und diese parallisieren: run_process(doc_paths)
                print('started with model: ', model_name)

                proc_wrap = wrapper(model_name=model_name, baseDir=src_path)
                print('initialized wrapper')
                sys.stdout.flush()

                pool.map(proc_wrap, sub_lists)
                print('finished model: ', model_name)

        print('finished inserting documents embeddings')