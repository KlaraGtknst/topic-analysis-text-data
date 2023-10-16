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
from elasticSearch import create_documents as document_creation
from elasticSearch import create_database as create_database
from elasticSearch import insert_embeddings as insert_embeddings

def main(src_path:str, image_src_path:str, client_addr=CLIENT_ADDR, n_pools=1, model_names: list = MODEL_NAMES):
    create_database.initialize_db(src_path, client_addr=client_addr)
    print('start creating documents using bulk')
    document_creation.create_documents(src_path = src_path, client_addr=client_addr) 
    print('finished creating documents using bulk')
    print('start inserting documents embeddings using bulk')
    insert_embeddings.main(src_path = src_path, image_src_path=image_src_path, client_addr=client_addr, model_names=model_names)
    

if __name__ == '__main__':
    args = arguments()

    #file_paths = get_input_filepath(args)
    file_path = args.directory
    out_file = get_filepath(args, option='output')
    image_src_path = get_filepath(args, option='image')
    model_names = get_model_names(args)

    main(src_path=file_path, image_src_path=image_src_path, client_addr=CLIENT_ADDR, model_names=model_names)