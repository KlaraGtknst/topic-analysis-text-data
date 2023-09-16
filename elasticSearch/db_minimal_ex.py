import collections
from http.client import HTTPException
from elasticsearch import ApiError, ConflictError, Elasticsearch
import base64
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.utils import simple_preprocess
# own modules
from text_embeddings.preprocessing.read_pdf import *
from user_interface.cli import *
from doc_images.pdf_matrix import *
from elasticSearch.queries.query_documents_tfidf import *
from text_embeddings.universal_sent_encoder_tensorFlow import *
from text_embeddings.hugging_face_sentence_transformer import *
from elasticSearch.queries.query_database import *
from doc_images.PCA.PCA_image_clustering import *
from text_embeddings.TFIDF.preprocessing.TfidfTextPreprocessor import *
from text_embeddings.InferSent.infer_pretrained import *
from text_embeddings import save_models 
from constants import *
from db_elasticsearch import *

def main(src_paths, image_src_path):
    NUM_DIMENSIONS = 55
    NUM_COMPONENTS = 2

    print('-' * 80)

    model_names = ['doc2vec', 'universal', 'hugging', 'infer', 'ae', 'tfidf']
    models = {}
    for model_name in model_names:
        print(model_name)
        try: # model exists
            model = save_models.load_model(model_name)
            models[model_name] = model
        except: # model does not exist, create and save it
            model = save_models.train_model(model_name, src_paths)
            models[model_name] = model
            save_models.save_model(model, model_name)

    # TODO: commented to save memory
    #sim_docs_vocab_size = len(list(models['tfidf'].vocabulary_.values()))

    # tfidf embedding incl. all-zero-vector-flag
    src_paths = ['/mnt/datasets/Bahamas/SAC/0/SAC1-6.pdf', '/mnt/datasets/Bahamas/SAC/0/SAC32-4.pdf']
    docs = get_docs_from_file_paths(src_paths)  # FIXME: lists will be very long- memory problem?
    sim_docs_document_term_matrix = models['tfidf'].fit_transform(docs).todense()
    flags = np.array([1 if np.array([entry  == 0 for entry in sim_docs_document_term_matrix[i]]).all() else 0 for i in range(len(sim_docs_document_term_matrix))]).reshape(len(sim_docs_document_term_matrix),1)
    flag_matrix = np.append(sim_docs_document_term_matrix, flags, axis=1)

    # Create the client instance
    client = Elasticsearch(CLIENT_ADDR)

    # delete old index and create new one, TODO: commented to save memory
    #client.options(ignore_status=[400,404]).indices.delete(index='bahamas')
    #init_db(client, num_dimensions=NUM_DIMENSIONS, sim_docs_vocab_size=sim_docs_vocab_size, n_components=NUM_COMPONENTS)

    # PCA + KMeans clustering, FIXME: a lot of memory due to list actions
    pca_cluster_df = get_cluster_PCA_df(src_path= image_src_path, n_cluster= 4, n_components= NUM_COMPONENTS, preprocess_image_size=600)

    # insert documents into database
    insert_documents(src_paths, doc2vec_model=models['doc2vec'], client=client, image_path=image_src_path, google_model=models['universal'], 
                     huggingface_model=models['hugging'], sim_doc_tfidf_vectorization=flag_matrix, pca_df=pca_cluster_df, 
                     inferSent_model=models['infer'], inferEncoder=models['ae'])