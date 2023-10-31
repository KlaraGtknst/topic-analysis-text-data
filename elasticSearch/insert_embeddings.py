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
        return scanRecurse(src_path)    # bulk, local
    else:
        return src_paths    # parallel, server

def generate_models_embedding(src_path: str, src_paths: list, models : dict, client: Elasticsearch, model_name: str = 'no_model'):  
    print('started with generate_models_embedding() ')
    #sys.stdout.flush()
    print('model_name: ', model_name, ' models.keys(): ', models.keys())
    for path in get_src_paths(src_path=src_path, src_paths=src_paths):
    
        text = pdf_to_str(path)
        id = get_hash_file(path)

        if (model_name in models.keys()) and (model_name != 'ae'):
            print('hi')
            try:
                embedding = get_embedding(models=models, model_name=model_name, text=text)
            except Exception as e:
                print('error in embedding: ', path, e)
                continue
            # print('hi2')
            # if src_paths == []: # bulk, local
            #     print(id)
            #     yield {     # vielleicht weg?
            #         '_op_type': 'update',
            #         '_index': 'bahamas',
            #         '_id': id,
            #         'doc': {MODELS2EMB[model_name]: embedding}
            #     }
            # else:   # parallel, server
        
            client.update(index='bahamas', id=id, body={'doc': {MODELS2EMB[model_name]: embedding}})
            print('hi3')


def insert_embedding(src_path: str, src_paths: list, models: dict={}, client_addr=CLIENT_ADDR, client: Elasticsearch=None, model_name: str = 'no_model'):
        '''
        :param src_path: path to the directory of the documents to be inserted into the database
        :param client: Elasticsearch client
        :param models: dictionary with model names as keys and the models as values
        :param model_names: names of the models to be used for embedding

        inserts specific embedding of all documents into the database 'bahamas'.
        '''
        print('started with insert_embedding() ')
        #sys.stdout.flush()
        if (model_name not in MODEL_NAMES) or (model_name == 'ae') or (model_name == 'none'):
            return
        
        client = client if client else Elasticsearch(client_addr, timeout=1000)
        models = models if models else get_models(src_path, [model_name] if model_name else None)
        print('got models: ', models.values())

        try:
            # if src_paths == []:
            #     print('entered bulk')
            #     #generate_models_embedding(src_paths=src_paths, src_path=src_path, models=models, model_name=model_name, client=client)
            #     bulk(client, generate_models_embedding(src_path=src_path, src_paths=src_paths, models=models, client=client, model_name=model_name), stats_only= True)
            # else:
            print('entered parallel')
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
    #print('started with get_embedding() ')
    if model_name == "doc2vec": 
        return models['doc2vec'].infer_vector(simple_preprocess(text))
    
    elif model_name in ['google_univ_sent_encoding', 'universal']:
        return embed([text], models['universal'])[0].numpy()
    
    elif model_name in ['huggingface_sent_transformer', 'hugging']:
        if type(text) == str:
            text = [text]

        return models['hugging'].encode(sentences=text)[0]
    
    elif model_name in ['inferSent_AE', 'infer']:
        #print('init start infer emb ', len(text), [text], type(text))
        # TODO
        inferSent_embedding = models['infer'].encode([text], tokenize=True)
    
        return models['ae'].predict(x=inferSent_embedding)[0]
                    
    elif model_name in ['sim_docs_tfidf', 'tfidf']:
        print('entered tfidf    ', [text])
        tfidf_embedding = get_tfidf_emb(models['tfidf'], [text])
        print('tfidf_embedding before ae: ', tfidf_embedding, tfidf_embedding.shape)
        if len(tfidf_embedding) > 2048:
            tfidf_ae_model = models['tfidf_ae']
            tfidf_embedding = tfidf_embedding.reshape(1, tfidf_embedding.shape[0])
            print('tfidf_embedding after ae: ', tfidf_embedding, tfidf_embedding.shape)
            return tfidf_ae_model.predict(x=tfidf_embedding)[0]
        return tfidf_embedding


# PCA & OPTICS

def pca_optics_aux(pca_dict: dict):  
    for id in pca_dict.index:
        yield {
            '_op_type': 'update',
            '_index': 'bahamas',
            '_id': id,
            'doc': {
                "pca_optics_cluster": pca_dict['cluster'][id],
                }
        }

def insert_pca_optics(pca_dict: dict, client_addr=CLIENT_ADDR, client: Elasticsearch=None):
        '''
        :param client: Elasticsearch client
        :param pca_dict: dictionary with weights and clusters

        inserts pca weights and OPTICS cluster of all documents into the database 'bahamas'.
        '''
        client = client if client else Elasticsearch(client_addr, timeout=1000)
      
        try:
            bulk(client, pca_optics_aux(pca_dict), stats_only= True)
        except (ConflictError, ApiError,EOFError) as err:
            print('error')
            return
        
def pca_weights_aux(src_path: str, image_root_path:str, max_w:int, max_h:int, pca_model: decomposition.PCA): 
    '''
    :param src_path: path to the directory of the documents to be inserted into the database
    :param image_root_path: path to the images
    :param max_w: max width of the images
    :param max_h: max height of the images
    :param pca_model: fitted PCA model

    inserts pca weights and argmax clusters of all documents into db
    ''' 
    for path in scanRecurse(src_path):
      
        id = get_hash_file(path)
        img_path = image_root_path + path.split('/')[-1].split('.')[0] + '.png'
        if not os.path.exists(img_path):
            print('not existent: ', path)
            pdf_to_png(file_path= [path], outpath=image_root_path, save= True)
        if not os.path.exists(img_path):    # some pdf cannot be read
            document = np.asarray([np.zeros((max_w, max_h)).ravel()])
        else:
            document_raw = plt.imread(img_path)
            new_w = np.minimum(document_raw.shape[0], max_w)
            new_h = np.minimum(document_raw.shape[1], max_h)
            document_raw = np.resize(document_raw, (new_w, new_h))
            
            document = eigendocs.proprocess_docs(raw_documents=[document_raw], max_w=max_w, max_h=max_h)

        reduced_img = pca_model.transform(document)
       
        yield {
            '_op_type': 'update',
            '_index': 'bahamas',
            '_id': id,
            'doc': {
                "pca_image": reduced_img[0],
                "argmax_pca_cluster": np.argmax(reduced_img),
                }
        }
        
def insert_pca_weights(src_path: str, pca_model: decomposition.PCA, img_path:str, max_w:int, max_h:int, client_addr=CLIENT_ADDR, client: Elasticsearch=None):
        '''
        :param src_path: paths to the directory of the documents to be inserted into the database
        :param client: Elasticsearch client
        :param pca_model: fitted PCA model

        inserts pca weights and OPTICS cluster of all documents into the database 'bahamas'.
        '''
        client = client if client else Elasticsearch(client_addr, timeout=1000)
      
        try:
            bulk(client, pca_weights_aux(src_path=src_path, pca_model=pca_model, image_root_path=img_path, max_w=max_w, max_h=max_h), stats_only= True)
        except (ConflictError, ApiError,EOFError) as err:
            print('error')
            return

def insert_precomputed_clusters(src_path: str, image_src_path:str, client_addr:str=CLIENT_ADDR):
    # get PCA model
    print('start getting pca model')
    pca_model, max_w, max_h = get_eigendocs_PCA(img_dir_src_path=image_src_path, n_components=NUM_PCA_COMPONENTS)
    print('finished getting pca model')

    # save weights and argmax cluster in db
    insert_pca_weights(src_path=src_path, pca_model=pca_model, img_path=image_src_path, client_addr=client_addr, max_w=max_w, max_h=max_h)
    print('finished inserting pca-argmax cluster df')

    # OPTICS clusters
    # get all pca weights and id 
    elastic_search_client = Elasticsearch(client_addr, timeout=1000)
    results = get_all_docs_in_db(elastic_search_client, src_includes = ['pca_image'])   # _id, pca_image
    result_df = pd.DataFrame.from_dict(results)
    result_df.set_index('_id', inplace=True)
    
    # clustering
    clt = OPTICS(cluster_method='dbscan', min_samples=2, max_eps=10, eps=0.5)
    result_df['cluster'] = clt.fit_predict(np.array(result_df['pca_image'].values.tolist()))

    print('finished getting pca-OPTICS cluster df')
    insert_pca_optics(pca_dict=result_df, client_addr=client_addr)
    print('finished inserting pca-OPTICS cluster df')

def main(src_path: str, client_addr=CLIENT_ADDR, model_names: list = MODEL_NAMES, num_cpus:int=1):
    src_path = '/Users/klara/Documents/uni/bachelorarbeit/data/*.pdf'
    print('start inserting documents embeddings')

    if num_cpus == 1:
        for model_name in model_names:  # function und diese parallisieren: run_process(doc_paths)
            print('started with model: ', model_name)
            insert_embedding(src_path = src_path, src_paths=[], client_addr=client_addr, model_name = model_name)
            print('finished model: ', model_name)
    else:
        # all paths
        document_paths = list(scanRecurse(src_path))
        print('number of docs: ', len(document_paths))
        sub_lists = list(chunks(document_paths, int(len(document_paths)/num_cpus)))

        print('obtained sublists')

        # process n_cpus sublists
        with Pool(processes=num_cpus) as pool:
            for model_name in model_names:  # function und diese parallisieren: run_process(doc_paths)
                proc_wrap = wrapper(model_name=model_name, baseDir=src_path)
                print('initialized wrapper')
                print('started with model: ', model_name)
                #sys.stdout.flush()
                pool.map(proc_wrap, sub_lists)
                print('finished model: ', model_name)

        print('finished inserting documents embeddings')