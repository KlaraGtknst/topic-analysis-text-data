import hashlib
from multiprocess import Pool
from elasticsearch import ApiError, ConflictError, Elasticsearch
import base64
from gensim.utils import simple_preprocess
from elasticSearch.models_aux import get_models
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
from elasticSearch.recursive_search import *

'''------initiate, fill and search in database-------
run this code by typing and altering the path:
    python3 db_elasticsearch.py -d '/Users/klara/Documents/Uni/bachelorarbeit/data/0/*.pdf' -D '/Users/klara/Documents/Uni/bachelorarbeit/images/images/'
'''

def init_db(client: Elasticsearch, sim_docs_vocab_size: int, n_components: int):
    '''
    :param client: Elasticsearch client
    :return: None

    This function initializes the database by creating an index (i.e. the structure for an entry of type 'bahamas' database).
    The index contains the following fields:
    - doc2vec: a dense vector of 100 (default) dimensions. This is the vector numerical representation of the document. Its similarity is measured by cosine similarity.
    - text: the text of the document. The text is not tokenized, stemmed etc.
    - path: the path to the document on the local maschine.
    - image: the image of the document (i.e. information about the document layout). The image is encoded in base64 and has 500 dpi.

    cf. https://www.elastic.co/guide/en/elasticsearch/reference/current/binary.html for information about binary for images
    cf. https://www.elastic.co/guide/en/elasticsearch/reference/current/dense-vector.html for information about dense vectors and similarity measurement types
    '''
    client.indices.create(index='bahamas', mappings={
        "properties": {
            "doc2vec": {
                "type": "dense_vector",
                "dims": 100,
                "index": True,
                "similarity": "cosine",
            },
            "google_univ_sent_encoding": {
                "type": "dense_vector",
                "dims": 512,
                "index": True,
                "similarity": "cosine",
            },
            "huggingface_sent_transformer": {
                "type": "dense_vector",
                "dims": 384,
                "index": True,
                "similarity": "cosine",
            },
            "sim_docs_tfidf": {
                "type": "dense_vector",
                "dims": min(sim_docs_vocab_size + 1, 2048), # last cell indicates all-zero-vector-representation
                "index": True,
                "similarity": "cosine",
            },
            "inferSent_AE": {
                "type": "dense_vector",
                "dims": 2048,
                "index": True,
                "similarity": "cosine",
            },
            "pca_image": {
                "type": "dense_vector",
                "dims": n_components,
            },
            "pca_optics_cluster": {
                "type": "byte",
            },
            "argmax_pca_cluster": {
                "type": "byte",
            },
            "text": {
                "type": "text",
            },
            "path": {
                "type": "keyword",
            },
            "image": {
                "type": "binary",
            },
        },
    })


def initialize_db(src_path, num_components: int=13, client_addr=CLIENT_ADDR):

    print('-' * 80)

    sim_docs_vocab_size = len(get_models(src_path = src_path, model_names = ["tfidf"])["tfidf"].vocabulary_.values())

    # Create the client instance
    client = Elasticsearch(client_addr)
    print('finished creating client')

    # delete old index and create new one
    client.options(ignore_status=[400,404]).indices.delete(index='bahamas')
    init_db(client, sim_docs_vocab_size=sim_docs_vocab_size, n_components=num_components)
    print('finished deleting old and creating new index')

    return client 


def main(src_path:str, client_addr=CLIENT_ADDR):
    initialize_db(src_path, client_addr=client_addr)
    

if __name__ == '__main__':
    args = arguments()
    file_path = args.directory

    initialize_db(file_path, client_addr=CLIENT_ADDR)