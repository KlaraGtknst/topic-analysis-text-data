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

def insert_documents(src_paths: list, pca_dict: dict, client: Elasticsearch, image_path: str = None, n_pools: int = 1, client_addr: str = CLIENT_ADDR, model_names: list = MODEL_NAMES):
    '''
    :param src_paths: path to the documents to be inserted into the database
    :param doc2vec_model: Doc2Vec model
    :param client: Elasticsearch client
    :param google_model: google universal sentence encoder model
    :param huggingface_model: huggingface sentence transformer model
    :param tfidf_model: tfidf model
    :param find_doc_tfidf_vectorization: document-term matrix of the tfidf embedding to find a document from the corpus
    :param pca_dict: dictionary with high level level keys: 'pca_weights' and 'cluster'. The second level keys are the indeces ('path').
    :param inferEncoder: encoder from trained autoencoder model to reduce dimension of InferSent embedding
    :param inferSent_model: trained InferSent model
    :param image_path: path to the images of the documents to be inserted into the database; if not set, assumes the images are in the same folder as the documents.
    :return: None

    This function inserts the documents into the database 'bahamas'. The documents are inserted as follows:
    - doc2vec: the embedding is inferred from the document text using the trained Doc2Vec model.
        The text is preprocessed using the gensim function 'simple_preprocess', which returns a list of tokens, i.e. unicode strings,
        which are lowercase and tokenized. (cf. https://tedboy.github.io/nlps/generated/generated/gensim.utils.simple_preprocess.html).
    - text: the text of the document. The text is not tokenized, stemmed etc.
    - path: the path to the document on the local machine.
    - image: the image of the document (i.e. information about the document layout). The image is encoded in base64 and has 500 dpi.
    
    The documents receive an id which is the name of the document (i.e. the name of the pdf file without the extension).

    cf. https://stackoverflow.com/questions/8908287/why-do-i-need-b-to-encode-a-string-with-base64 for information about base64 encoding
    cf. https://www.codespeedy.com/convert-image-to-base64-string-in-python/ for information about converting images to base64
    '''
    image_path = image_path if image_path else (src_paths.split('data/0/')[0] + 'images/images/')
    print('start multiprocessing'if n_pools > 1 else 'start single processing')
    models = get_models(src_paths, model_names=model_names) if n_pools == 1 else None
    #models = None
    if n_pools == 1:    # single processing
        for src_path in src_paths:
            insert_document(src_path, pca_dict, image_path, client_addr=client_addr, models=models, client=client, model_names=model_names)
    else: # multiprocessing, TODO: models are not pickable -> too much memory spent reloading
        with Pool(n_pools) as p: # number of cpus n_pools
            p.starmap(insert_document, list(map(lambda src_path:[src_path, pca_dict, image_path, client_addr, models], src_paths)))


def get_models(src_paths: list, model_names: list = MODEL_NAMES):
    '''
    src_paths: paths to the documents to be inserted into the database
    model_names: names of the models to be used for embedding
    return: dictionary with model names as keys and the models as values
    '''
    models = {}
    if 'infer' in model_names and (not 'ae' in model_names):    # needs AE for embedding
        model_names = model_names + ['ae']
    for model_name in model_names:
        try: # model exists
            model = save_models.load_model(model_name)
            models[model_name] = model
        except: # model does not exist, create and save it
            model = save_models.train_model(model_name, src_paths)
            models[model_name] = model
            save_models.save_model(model, model_name)
    return models

def insert_document(src_path, pca_dict: dict, image_path, models, client_addr=CLIENT_ADDR, client: Elasticsearch=None, model_names: list = MODEL_NAMES):
        '''
        :param src_path: path to the document to be inserted into the database
        :param pca_dict: dictionary with high level level keys: 'pca_weights' and 'cluster'. The second level keys are the indeces ('path').
        :param client: Elasticsearch client
        :param image_path: path to image
        :param models: dictionary with model names as keys and the models as values
        :param model_names: names of the models to be used for embedding

        inserts one document into the database 'bahamas'.
        '''
        client = client if client else Elasticsearch(client_addr)
        path = src_path
        models = models if models else get_models(src_path, model_names if model_names else None)

        try:
            text = pdf_to_str(path)
           
            id = get_hash_file(path)

            if get_doc_meta_data(client, doc_id=id) is not None:    # document already in database
                return
            
            # insert new document
            # TODO: batch aus nicht inserted documents?

            image = image_path + path.split('/')[-1].split('.')[0]  + '.png'
        
            b64_image = get_b64_image(image)
       
            doc = { 
                    "pca_image": pca_dict['pca_weights'][image] if image in pca_dict['pca_weights'].keys() else None,
                    "pca_optics_cluster": pca_dict['cluster'][image] if image in pca_dict['pca_weights'].keys() else None, 
                    "text": text,
                    "path": path,
                    "image": b64_image.decode('ASCII')#str(b64_image) # TODO: statt str... b64_image.decode('ASCII'),
                }
            for model_name in model_names:
                if (model_name in models.keys()) and (model_name != 'ae'):
                    embedding = get_embedding(models=models, model_name=model_name, text=text)
                    if len(embedding) > 2048:   # tfidf
                        continue
                    doc[MODELS2EMB[model_name]] = embedding

            client.create(index='bahamas', id=id, document=doc, timeout='50s')
         
        except (ConflictError, ApiError,EOFError) as err:
            print('error')
            return

def create_document_aux(src_paths: list, client: Elasticsearch):  
    for path in src_paths:
        try:           
            id = get_hash_file(path)

            if get_doc_meta_data(client, doc_id=id) is not None:    # document already in database
                return
            
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
        if model_name not in MODEL_NAMES:
            return
        
        client = client if client else Elasticsearch(client_addr)
        models = models if models else get_models(src_paths, [model_name] if model_name else None)

        try:
            bulk(client, generate_models_embedding(src_paths, models, model_name), stats_only= True)
        except (ConflictError, ApiError,EOFError) as err:
            print('error')
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


def initialize_db(src_paths, num_components: int=13, client_addr=CLIENT_ADDR):

    print('-' * 80)

    sim_docs_vocab_size = len(get_models(src_paths = src_paths, model_names = ["tfidf"])["tfidf"].vocabulary_.values())

    # Create the client instance
    client = Elasticsearch(client_addr)
    print('finished creating client')

    # delete old index and create new one
    client.options(ignore_status=[400,404]).indices.delete(index='bahamas')
    init_db(client, sim_docs_vocab_size=sim_docs_vocab_size, n_components=num_components)
    print('finished deleting old and creating new index')

    return client 

def documents_into_db(src_paths, image_src_path, client, num_components: int=13, client_addr=CLIENT_ADDR, n_pools: int = 1, model_names: list = MODEL_NAMES):
    client = client if client else Elasticsearch(client_addr)

    # Eigendocs (PCA) + OPTICS clustering
    pca_optics_dict = get_eigendocs_OPTICS_df(image_src_path, n_components=num_components).to_dict()
    print('finished getting pca-OPTICS cluster df')

    # insert documents into database
    print(f'start inserting {len(src_paths)} documents')
    insert_documents(src_paths, client=client, image_path=image_src_path, n_pools=n_pools, pca_dict=pca_optics_dict, model_names=model_names)
    print('finished inserting documents')

    # alternatively, use AsyncElasticsearch or time.sleep(1)
    client.indices.refresh(index="bahamas")

    # properties in db
    print(client.indices.get_mapping(index='bahamas'))

    # number of documents in database
    client.indices.refresh(index='bahamas')
    resp = client.count(index='bahamas')
    print('number of documents in database: ', resp['count'])

def init_db_aux(src_paths, image_src_path, client_addr=CLIENT_ADDR, n_pools=1, model_names: list = MODEL_NAMES):
    '''
    everything that happens in the main function to fill the database.
    '''
    NUM_COMPONENTS = 13

    client = initialize_db(src_paths, client_addr=client_addr, num_components=NUM_COMPONENTS)

    documents_into_db(src_paths, image_src_path, client, num_components= NUM_COMPONENTS, client_addr=client_addr, n_pools= n_pools, model_names= model_names)



def main(src_paths, image_src_path, client_addr=CLIENT_ADDR, n_pools=1, model_names: list = MODEL_NAMES):
    # everything in one function
    #init_db_aux(src_paths, image_src_path, client_addr=client_addr, n_pools=n_pools, model_names=model_names)

    # stepwise
    #initialize_db(src_paths, client_addr=client_addr) # WORKS
    #documents_into_db(src_paths, image_src_path, client=None, client_addr=client_addr, n_pools= n_pools, model_names= model_names)  # OUT OF MEMORY
    #print('start creating documents using bulk')
    #create_documents(src_paths = src_paths, client_addr=client_addr) # WORKS
    #print('finished creating documents using bulk')
    print('start inserting documents embeddings using bulk')
    for model_name in model_names:
        print('started with model: ', model_name)
        insert_embedding(src_paths = src_paths, client_addr=client_addr, model_name = model_name)
        print('finished model: ', model_name)
    print('finished inserting documents embeddings using bulk')
    

if __name__ == '__main__':
    args = arguments()

    file_paths = get_input_filepath(args)
    out_file = get_filepath(args, option='output')
    image_src_path = get_filepath(args, option='image')
    model_names = get_model_names(args)

    init_db_aux(src_paths=file_paths, image_src_path=image_src_path, client_addr=CLIENT_ADDR, model_names=model_names)