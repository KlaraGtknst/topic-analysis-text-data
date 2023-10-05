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
            if len(embedding) > 2048:   # tfidf
                return
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
        
        client = client if client else Elasticsearch(client_addr)
        models = models if models else get_models(src_paths, [model_name] if model_name else None)

        try:
            bulk(client, generate_models_embedding(src_paths, models, model_name), stats_only= True)
        except (ConflictError, ApiError,EOFError) as err:
            print('error')
            return

def get_embedding(models, model_name: str, text: str):
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
        tfidf_emb = models['tfidf'].transform([text]).todense()
        flag = np.array(1 if np.array([entry  == 0 for entry in tfidf_emb]).all() else 0).reshape(len(tfidf_emb),1)
        flag_matrix = np.append(tfidf_emb, flag, axis=1)
        return np.ravel(np.array(flag_matrix))


def main(src_paths, client_addr=CLIENT_ADDR, model_names: list = MODEL_NAMES):
  
    print('start inserting documents embeddings using bulk')
    for model_name in model_names:
        print('started with model: ', model_name)
        insert_embedding(src_paths = src_paths, client_addr=client_addr, model_name = model_name)
        print('finished model: ', model_name)
    print('finished inserting documents embeddings using bulk')
    

if __name__ == '__main__':
    args = arguments()

    file_paths = get_input_filepath(args)
    model_names = get_model_names(args)

    main(src_paths=file_paths, client_addr=CLIENT_ADDR, model_names=model_names)