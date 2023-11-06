from elasticsearch import ApiError, ConflictError, Elasticsearch
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
from elasticSearch.models_aux import *
from elasticSearch.create_documents import *
from doc_images.convert_pdf2image import *
from elasticSearch.recursive_search import *

class wrapper:
    def __init__(self, imgBaseDir:str, pca_model, max_w:int, max_h:int, client_addr:str=CLIENT_ADDR):
        self.imgBaseDir = imgBaseDir
        self.client_addr = client_addr
        self.pca_model = pca_model
        self.max_w = max_w
        self.max_h = max_h

    def __call__(self, src_paths: list):
        insert_pca_weights_aux(src_paths, self.pca_model, self.max_w, self.max_h, image_src_path=self.imgBaseDir, client_addr=self.client_addr)
        #insert_precomputed_clusters(src_paths=src_paths, image_src_path=self.imgBaseDir, client_addr=self.client_addr)


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
        
def pca_weights_aux(src_paths: list, image_root_path:str, max_w:int, max_h:int, pca_model: decomposition.PCA): 
    '''
    :param src_path: path to the directory of the documents to be inserted into the database
    :param image_root_path: path to the images
    :param max_w: max width of the images
    :param max_h: max height of the images
    :param pca_model: fitted PCA model

    inserts pca weights and argmax clusters of all documents into db
    ''' 
    for path in src_paths:
        if not path.split('/')[-1].startswith('.'):
        
            id = get_hash_file(path)
            img_path = image_root_path + path.split('/')[-1].split('.')[0] + '.png'
            if not os.path.exists(img_path):
                pdf_to_png(file_path= [path], outpath=image_root_path, save= True)

            try:
                document_raw = plt.imread(img_path)
                new_w = np.minimum(document_raw.shape[0], max_w)
                new_h = np.minimum(document_raw.shape[1], max_h)
                document_raw = np.resize(document_raw, (new_w, new_h))
                document = eigendocs.proprocess_docs(raw_documents=[document_raw], max_w=max_w, max_h=max_h)
            except:
                document = np.asarray([np.zeros((max_w, max_h)).ravel()])
                
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
        
def insert_pca_weights(src_paths: list, pca_model: decomposition.PCA, img_path:str, max_w:int, max_h:int, client_addr=CLIENT_ADDR, client: Elasticsearch=None):
        '''
        :param src_path: paths to the directory of the documents to be inserted into the database
        :param client: Elasticsearch client
        :param pca_model: fitted PCA model

        inserts pca weights and OPTICS cluster of all documents into the database 'bahamas'.
        '''
        client = client if client else Elasticsearch(client_addr, timeout=1000)
      
        try:
            bulk(client, pca_weights_aux(src_paths=src_paths, pca_model=pca_model, image_root_path=img_path, max_w=max_w, max_h=max_h), stats_only= True)
        except (ConflictError, ApiError,EOFError) as err:
            print('error')
            return

def insert_pca_weights_aux(src_paths: list, pca_model, max_w, max_h, image_src_path:str, client_addr:str=CLIENT_ADDR):
    # save weights and argmax cluster in db
    insert_pca_weights(src_paths=src_paths, pca_model=pca_model, img_path=image_src_path, client_addr=client_addr, max_w=max_w, max_h=max_h)

def insert_precomputed_clusters(src_paths: list, image_src_path:str, client_addr:str=CLIENT_ADDR, from_n:int=0):
    # get PCA model
    print('start getting pca model')
    pca_model, max_w, max_h = get_eigendocs_PCA(img_dir_src_path=image_src_path, n_components=NUM_PCA_COMPONENTS)
    print('finished getting pca model')

    # save weights and argmax cluster in db
    for l in src_paths:
        insert_pca_weights(src_paths=l, pca_model=pca_model, img_path=image_src_path, client_addr=client_addr, max_w=max_w, max_h=max_h)

    print('finished inserting pca-argmax cluster df')
    elastic_search_client = Elasticsearch(client_addr, timeout=1000)
    for l in src_paths:
        # OPTICS clusters
        # get all pca weights and id 
        results = get_all_docs_in_db(elastic_search_client, src_includes = ['pca_image'], all=False, from_n=from_n)
        print('results: ', len(results))
        sys.stdout.flush()
        result_df = pd.DataFrame.from_dict(results)
        result_df.set_index('_id', inplace=True)
        result_df = result_df.dropna()
        print('result_df: ', result_df.shape)
        print(np.array(result_df['pca_image'].values).shape)
        sys.stdout.flush()

        
        # clustering
        clt = OPTICS(cluster_method='dbscan', min_samples=2, max_eps=10, eps=0.5)
        result_df['cluster'] = clt.fit_predict(np.array(result_df['pca_image'].values.tolist()))

        print('finished getting pca-OPTICS cluster df')
        insert_pca_optics(pca_dict=result_df, client_addr=client_addr)
        print('finished inserting pca-OPTICS cluster df')

def main(src_path: str, image_src_path: str, client_addr=CLIENT_ADDR, num_cpus:int=1):
    elastic_search_client = Elasticsearch(client_addr, timeout=1000)
    size= get_number_docs_in_db(elastic_search_client)
    print('number of documents in db: ', size)

    # all paths
    document_paths = list(scanRecurse(src_path))
    sub_lists = list(chunks(document_paths, 2000))

    print('start getting pca model')
    pca_model, max_w, max_h = get_eigendocs_PCA(img_dir_src_path=image_src_path, n_components=NUM_PCA_COMPONENTS)
    print('finished getting pca model')
   
    # process n_cpus sublists
    print('start inserting documents clusters and PCA weights')
    #with Pool(processes=num_cpus) as pool:
    for l in sub_lists:
        proc_wrap = wrapper(imgBaseDir=image_src_path, client_addr=client_addr,  pca_model=pca_model, max_w=max_w, max_h=max_h)
        print('initialized wrapper')
        sys.stdout.flush()
        proc_wrap(l)
        #pool.map(proc_wrap, sub_lists)
        print('finished inserting documents clusters and PCA weights')