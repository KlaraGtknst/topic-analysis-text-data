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
from constants import *
from elasticSearch.recursive_search import *

def create_document_aux(src_paths: list, client: Elasticsearch):  
    for path in src_paths:
       
        try:           
            id = get_hash_file(path)

            if get_doc_meta_data(client, doc_id=id) is not None:    # document already in database
                continue
            
            text = pdf_to_str(path)
            
            yield { 
                    '_op_type': 'create',
                    '_index': 'bahamas',
                    '_id': id,
                    "text": text,
                    "path": path,
                }
          
        except (EOFError) as err:
            print('EOF error')
            return

def create_documents(src_paths: list, client_addr=CLIENT_ADDR, client: Elasticsearch=None):
        '''
        :param src_path: path to the document to be inserted into the database
        :param client: Elasticsearch client
        :param client_addr: address of the Elasticsearch client

        creates all document in the database 'bahamas'.
        '''
        client = client if client else Elasticsearch(client_addr)
        try:
            bulk(client, create_document_aux(src_paths, client), stats_only= True)
         
        except (ConflictError, ApiError,EOFError) as err:
            print(err)
            return

def get_b64_image(image: str):
    '''
    :param image: path to the image
    :return: base64 encoded image
    '''
    try:
        with open(image, "rb") as img_file:
            b64_image = base64.b64encode(img_file.read())
    except FileNotFoundError:
        # bc i did not copy all images from cluster to local machine
        # dummy (black) picture to avoid not inserting the document into the database
        b64_image = base64.b64encode(np.zeros([100,100,3],dtype=np.uint8)) 
    return b64_image
    
def get_hash_file(path: str):
    '''
    :param path: path to the file
    :return: hash of the file
    '''
    BLOCK_SIZE = 65536000 # The size of each read from the file
    file_hash = hashlib.sha256() # Create the hash object, can use something other than `.sha256()` if you wish
    with open(path, 'rb') as f: # Open the file to read it's bytes, automatically closes file at end
        fb = f.read(BLOCK_SIZE) # Read from the file. Take in the amount declared above
        while len(fb) > 0: # While there is still data being read from the file
            file_hash.update(fb) # Update the hash
            fb = f.read(BLOCK_SIZE)
    id = file_hash.hexdigest()
    return id

def main(src_path:str, client_addr=CLIENT_ADDR, num_cpus:int=1):
    print('start creating documents using bulk')

    # all paths
    document_paths = list(scanRecurse(src_path))
    sub_lists = list(chunks(document_paths, int(len(document_paths)/num_cpus)))
   
   # process n_cpus sublists
    with Pool(processes=num_cpus) as pool:
        pool.map(lambda x : create_documents(src_paths = x, client_addr=client_addr), sub_lists)

    print('finished creating documents using bulk')
        
    

if __name__ == '__main__':
    args = arguments()

    file_path = args.directory

    create_documents(src_path=file_path, client_addr=CLIENT_ADDR)