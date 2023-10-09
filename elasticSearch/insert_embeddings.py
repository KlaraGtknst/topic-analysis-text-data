import hashlib
from multiprocess import Pool
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


def generate_models_embedding(src_paths: list, models : dict, model_name: str = 'no_model'):  
    for path in src_paths:
        text = pdf_to_str(path)
        id = get_hash_file(path)

        if (model_name in models.keys()) and (model_name != 'ae'):
            embedding = get_embedding(models=models, model_name=model_name, text=text)

            yield {
                '_op_type': 'update',
                '_index': 'bahamas',
                '_id': id,
                'doc': {MODELS2EMB[model_name]: embedding}
            }

def insert_embedding(src_paths: list, models: dict=None, client_addr=CLIENT_ADDR, client: Elasticsearch=None, model_name: str = 'no_model'):
        '''
        :param src_path: list of paths to the documents to be inserted into the database
        :param client: Elasticsearch client
        :param models: dictionary with model names as keys and the models as values
        :param model_names: names of the models to be used for embedding

        inserts specific embedding of all documents into the database 'bahamas'.
        '''
        if (model_name not in MODEL_NAMES) or (model_name == 'ae'):
            return
        
        client = client if client else Elasticsearch(client_addr, timeout=1000)
        models = models if models else get_models(src_paths, [model_name] if model_name else None)

        try:
            bulk(client, generate_models_embedding(src_paths, models, model_name), stats_only= True)
        except (ConflictError, ApiError,EOFError) as err:
            print('error')
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
        return models['hugging'].encode(text)
    
    elif model_name in ['inferSent_AE', 'infer']:
        inferSent_embedding = models['infer'].encode([text], tokenize=True)
        return models['ae'].predict(x=inferSent_embedding)[0]
                    
    elif model_name in ['sim_docs_tfidf', 'tfidf']:
        tfidf_embedding = get_tfidf_emb(models['tfidf'], [text])
        if len(tfidf_embedding) > 2048:
            tfidf_ae_model = models['tfidf_ae']
            return tfidf_ae_model.predict(x=tfidf_embedding)[0]
        return tfidf_embedding


# PCA & OPTICS

def pca_optics_aux(src_paths: list, pca_dict: dict, img_path:str):  
    for path in src_paths:
        id = get_hash_file(path)
        if img_path.endswith('/'):
            img_path = img_path + '*.png'
        img_id = '/'.join(img_path.split('/')[:-1]) + '/' + path.split('/')[-1].split('.')[0] + '.png'

        yield {
            '_op_type': 'update',
            '_index': 'bahamas',
            '_id': id,
            'doc': {
                "pca_image": pca_dict['pca_weights'][img_id] if img_id in pca_dict['pca_weights'].keys() else None,
                "pca_optics_cluster": pca_dict['cluster'][img_id] if img_id in pca_dict['pca_weights'].keys() else None,
                "argmax_pca_cluster": np.argmax(pca_dict['pca_weights'][img_id]) if img_id in pca_dict['pca_weights'].keys() else None
                }
        }

def insert_pca_optics(src_paths: list, pca_dict: dict, img_path:str, client_addr=CLIENT_ADDR, client: Elasticsearch=None):
        '''
        :param src_path: list of paths to the documents to be inserted into the database
        :param client: Elasticsearch client
        :param pca_dict: dictionary with weights and clusters

        inserts pca weights and OPTICS cluster of all documents into the database 'bahamas'.
        '''
        client = client if client else Elasticsearch(client_addr)
      
        try:
            bulk(client, pca_optics_aux(src_paths, pca_dict, img_path=img_path), stats_only= True)
        except (ConflictError, ApiError,EOFError) as err:
            print('error')
            return

def main(src_paths: list, image_src_path: str, num_components=13, client_addr=CLIENT_ADDR, model_names: list = MODEL_NAMES):
    print('start inserting documents embeddings using bulk')
    for model_name in model_names:
        print('started with model: ', model_name)
        insert_embedding(src_paths = src_paths, client_addr=client_addr, model_name = model_name)
        print('finished model: ', model_name)
    pca_optics_dict = get_eigendocs_OPTICS_df(image_src_path, n_components=NUM_PCA_COMPONENTS).to_dict()
    print('finished getting pca-OPTICS cluster df')
    insert_pca_optics(src_paths=src_paths, pca_dict=pca_optics_dict, client_addr=client_addr, img_path=image_src_path)
    print('finished inserting pca-OPTICS cluster df')

    print('finished inserting documents embeddings using bulk')
    

if __name__ == '__main__':
    args = arguments()

    file_paths = get_input_filepath(args)
    model_names = get_model_names(args)

    main(src_paths=file_paths, client_addr=CLIENT_ADDR, model_names=model_names)