import base64
import glob
import io
import pickle

from elasticsearch import Elasticsearch
from elasticSearch.db_elasticsearch import *
from elasticSearch import models_aux
from text_embeddings.InferSent.infer_pretrained import autoencoder_emb_model, init_infer
from text_embeddings.universal_sent_encoder_tensorFlow import *
from text_embeddings.hugging_face_sentence_transformer import *
from elasticSearch.queries.query_documents_tfidf import *
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from PIL import Image
from tensorflow.python.keras.models import model_from_json
from gensim.test.utils import get_tmpfile
from constants import CLIENT_ADDR
from elasticSearch.recursive_search import *
import sys

SRC_PATH = '/Users/klara/Documents/uni/bachelorarbeit/data/0/*.pdf'
DOC_PATH = '/Users/klara/Downloads/*.pdf'
NUM_COMPONENTS = 2
NUM_RESULTS = 4

def save_model(model, model_name:str):
    '''
    :param model: The model to be saved.
    :param model_name: The name/ type of the model.

    Saves the model to the models folder.
    If the folder models does not exist, it will be created.
    '''

    if not os.path.exists('models'):
        os.mkdir('models')

    if 'doc2vec' in model_name:
        model.save(f'models/{model_name}.pkl')

    elif 'tfidf_ae' == model_name:
        with open(f'models/{model_name}.pkl', 'wb') as file:  
            pickle.dump(model, file)
    
    elif 'tfidf' in model_name:
        with open(f'models/{model_name}_vectorizer.pk', 'wb') as fin:
            pickle.dump(model, fin)

    elif 'infer' in model_name:
        with open(f'models/{model_name}.pkl', 'wb') as file:  
            pickle.dump(model, file)

    elif 'ae' in model_name:
        with open(f'models/{model_name}.pkl', 'wb') as file:  
            pickle.dump(model, file)



def load_model(model_name:str):
    '''
    :param model_name: The name/ type of the model.
    :return: The model.
    
    Loads the model from the models folder.
    '''
    if 'doc2vec' in model_name:
        model_path = 'models/doc2vec_model.pkl' # local path
        if not os.path.exists(model_path):
            model_path = 'models/doc2vec.pkl'   # server path
        return Doc2Vec.load(model_path)
    
    elif 'universal' in model_name:
        return google_univ_sent_encoding_aux()
    
    elif 'hugging' in model_name:
        return init_hf_sentTrans_model()
    
    elif 'tfidf_ae' == model_name:
        with open(f'models/tfidf_ae.pkl', 'rb') as file: 
            return pickle.load(file)
    
    elif 'infer' in model_name:
        print('entered: load_model() ', model_name)
        sys.stdout.flush()
        with open(f'models/infer.pkl', 'rb') as file:  
            return pickle.load(file)    
        
    elif ('ae' in model_name) and not ('tfidf' in model_name):
        with open(f'models/ae.pkl', 'rb') as file: # inferSent_AE
            return pickle.load(file)
        
    elif 'tfidf' in model_name:
        return pickle.load(open('models/tfidf_vectorizer.pk', 'rb'))
    
    else:
        print(f'{model_name} not found')

def get_tagged_input_documents(src_path: str, tokens_only: bool = False):
    '''
    :param src_path: paths to directory of the documents to be inserted into the database
    :param tokens_only: if True, only the tokens of the document are returned, else tagged tokens are returned
    :return: tagged tokens or only the tokens (depending on tokens_only)

    The gensim function 'simple_preprocess' converts a document into a list of tokens (cf. https://tedboy.github.io/nlps/generated/generated/gensim.utils.simple_preprocess.html).
    The resulting list of unicode strings is lowercased and tokenized.

    cf. https://radimrehurek.com/gensim/auto_examples/tutorials/run_doc2vec_lee.html#sphx-glr-auto-examples-tutorials-run-doc2vec-lee-py
    for the original code
    '''
    src_paths = list(scanRecurse(src_path))
    for i in range(len(src_paths)):
        path = src_paths[i]
        tokens = simple_preprocess(pdf_to_str(path))
        if tokens_only:
            yield tokens
        else:
            yield TaggedDocument(tokens, [i])

def train_model(model_name:str, src_path:str, client:Elasticsearch=None):
    '''
    :param model_name: The name/ type of the model.
    :param src_path: The path to the directory of the documents to be used for training.
    :return: The trained model.
    '''

    if 'doc2vec' in model_name:
        train_corpus = list(get_tagged_input_documents(src_path=src_path))
        d2v_model = Doc2Vec(train_corpus)
        return d2v_model
    
    elif 'universal' in model_name:
        return google_univ_sent_encoding_aux()
    
    elif 'hugging' in model_name:
        return init_hf_sentTrans_model()
    
    elif 'infer' in model_name:
        print('train_model():  start model training', model_name)
        sys.stdout.flush()
        # model
        local_model_path = '/Users/klara/Developer/Uni/encoder/infersent1.pkl'
        server_model_path = '/mnt/stud/work/kgutekunst/encoder/infersent1.pkl'
        model_path = local_model_path if os.path.exists(local_model_path) else server_model_path

        print('train_model():  obtained model path', model_path)
        sys.stdout.flush()

        # word2vec embeddings
        # w2v_local_path = '/Users/klara/Developer/Uni/GloVe/glove.840B.300d.txt'
        w2v_local_path = '/Users/klara/Developer/Uni/bahamas_word2vec/bahamas_w2v.txt'
        w2v_server_path = '/mnt/stud/work/kgutekunst/bahamas_word2vec/bahamas_w2v.txt'
        custom_w2v_path = w2v_local_path if os.path.exists(w2v_local_path) else w2v_server_path

        print('train_model():  obtained w2v path', custom_w2v_path)
        sys.stdout.flush()

        inferSent_model, docs = init_infer(model_path=model_path, w2v_path=custom_w2v_path, file_path=src_path, version=1)

        print('train_model():  trained model', custom_w2v_path)
        sys.stdout.flush()
        return inferSent_model
    
    elif 'tfidf_ae' in model_name:
        try:    # existing model
            tfidf_model = load_model('tfidf')
        except: # new model
            tfidf_model = train_model('tfidf', src_path)
        docs = get_docs_from_file_paths(src_path)
        tfidf_embeddings = np.array([models_aux.get_tfidf_emb(tfidf_model, [doc]) for doc in docs])
        encoded_tfidf_embedding, ae_tfidf_encoder, ae_tfidf_decoder = autoencoder_emb_model(input_shape=tfidf_embeddings.shape[1], latent_dim=2048, data=tfidf_embeddings)
        return ae_tfidf_encoder
        
    elif 'ae' in model_name:
        # get inferSent model
        try:    # existing model
            inferSent_model = load_model('infersent_model')
            docs = get_docs_from_file_paths(src_path)
        except: # new model
            inferSent_model = train_model('infersent_model', src_path)

        infer_embeddings = inferSent_model.encode(docs, tokenize=True)
        encoded_infersent_embedding, ae_infer_encoder, ae_infer_decoder = autoencoder_emb_model(input_shape=infer_embeddings.shape[1], latent_dim=2048, data=infer_embeddings)
        return ae_infer_encoder
        
    elif 'tfidf' in model_name:
        if client is None:
            client = Elasticsearch(CLIENT_ADDR)
        docs = get_docs_from_file_paths(src_path)
        sim_docs_tfidf = TfidfVectorizer(input='content', preprocessor=TfidfTextPreprocessor().transform, min_df=3, max_df=int(len(docs)*0.07))
        sim_docs_tfidf.fit(docs)
        return sim_docs_tfidf
    
    else:
        print(f'{model_name} not found')

    
def get_model(model_name:str, src_path:str):
    '''
    :param model_name: The name/ type of the model.
    :param src_path: The path to the directory of the documents to be used for training.
    :return: The model.
    '''
    try: # model exists
        model = load_model(model_name)
    except: # model does not exist, create and save it
        model = save_models.train_model(model_name, src_path = src_path)
        save_models.save_model(model, model_name)
    return model

def main(path=None):
    # path = glob.glob('/Users/klara/Downloads/*.pdf')[0]
    src_path='/Users/klara/Documents/Uni/bachelorarbeit/data/0/*.pdf'
    #src_paths = glob.glob(src_path)
    text = pdf_to_str(path)

    # TF-IDF + AE
    tfidf_model = train_model('tfidf ', src_path)
    #tfidf_ae_model = train_model('tfidf_ae', src_paths)
    tfidf_embedding = models_aux.get_tfidf_emb(tfidf_model, [text])
    # (1, 4096)
    tfidf_embedding = tfidf_embedding.reshape(1, tfidf_embedding.shape[0])
    tfidf_embedding = np.array(tfidf_embedding)
    encoded_tfidf_embedding, tfidf_ae_model, ae_tfidf_decoder = autoencoder_emb_model(input_shape=tfidf_embedding.shape[1], latent_dim=2048, data=tfidf_embedding)
    tfidf_ae_embedding = tfidf_ae_model.predict(x=tfidf_embedding)[0]   # TODO: Problem

    # Universal 
    # model = google_univ_sent_encoding_aux()
    # print(model)
    # embedding = embed([text], model)[0].numpy()
    # print(embedding)

    # Doc2Vec
    # save and load
    # train_corpus = list(get_tagged_input_documents(src_paths=glob.glob(SRC_PATH)))
    # d2v_model = Doc2Vec(train_corpus)
    # model_name = 'doc2vec_model'
    
    # save_model(d2v_model, model_name)

    # d2v_model = Doc2Vec.load(f'models/{model_name}.pkl')
    # print(d2v_model.infer_vector(simple_preprocess(pdf_to_str(path))))
    '''# train
    d2v_model = train_model('doc2vec_model', src_paths)
    print('Doc2Vec: ',d2v_model.infer_vector(simple_preprocess(pdf_to_str(path))))'''

    # TF-IDF
    # save and load
    # client_addr=CLIENT_ADDR
    # client = Elasticsearch(client_addr)
    
    # docs = get_docs_from_file_paths(src_paths)
    # # sim_docs_tfidf = TfidfVectorizer(input='content', preprocessor=TfidfTextPreprocessor().transform, min_df=3, max_df=int(len(docs)*0.07))
    # # sim_docs_document_term_matrix = sim_docs_tfidf.fit(docs)
    # # save_model(sim_docs_tfidf, 'tfidf')
    # print('docs loaded')
    # model = load_model('models/tfidf_vectorizer.pk')
    # embedding = model.transform(docs)
    # embedding = np.ravel(embedding.todense())
    # embedding = np.append(embedding, 1 if np.array([entry  == 0 for entry in embedding]).all() else 0)
    # print(np.array([entry  == 0 for entry in embedding]).all())
    # print(embedding)
    
    # '''# train
    # model = train_model('tfidf', src_paths)
    # embedding = model.transform([pdf_to_str(path)])
    # embedding = np.ravel(embedding.todense())
    # embedding = np.append(embedding, 1 if np.array([entry  == 0 for entry in embedding]).all() else 0)
    # print('tfidf: ', embedding)'''
    
    # # InferSent + AE
    # # save and load
    # infer_model_name = 'infersent_model'
    # ae_model_name = 'ae'
    # # InferSent
    # if (not os.path.exists(f"models/{infer_model_name}.pkl")):
    #     MODEL_PATH = '/Users/klara/Developer/Uni/encoder/infersent1.pkl'
    #     #W2V_PATH = '/Users/klara/Developer/Uni/GloVe/glove.840B.300d.txt'
    #     CUSTOM_W2V_PATH = '/Users/klara/Developer/Uni/bahamas_word2vec/bahamas_w2v.txt'
    #     inferSent_model, docs = init_infer(model_path=MODEL_PATH, w2v_path=CUSTOM_W2V_PATH, file_paths=src_paths, version=1)
    #     save_model(inferSent_model, infer_model_name)
    # else:
    #     inferSent_model = load_model(infer_model_name)
        #docs = get_docs_from_file_paths(src_paths)

    # # AE
    # if (not os.path.exists(f"models/{ae_model_name}.pkl")):
    #     infer_embeddings = inferSent_model.encode(docs, tokenize=True)
    #     encoded_infersent_embedding, ae_infer_encoder = autoencoder_emb_model(input_shape=infer_embeddings.shape[1], latent_dim=2048, data=infer_embeddings)
    #     save_model(ae_infer_encoder, ae_model_name)
    # else:
    #     ae_infer_encoder = load_model(ae_model_name)

    # inferSent_embedding = inferSent_model.encode([text], tokenize=True)
    # compressed_infersent_embedding = ae_infer_encoder.predict(x=inferSent_embedding)
    # print(compressed_infersent_embedding[0])
    # print(np.array([entry  == 0 for entry in compressed_infersent_embedding]).all())
    # train
    # inferSent_model = train_model('infer', src_paths)
    # inferSent_embedding = inferSent_model.encode([text], tokenize=True)
    # print('infersent: ', inferSent_embedding, inferSent_embedding.shape)
    # ae_infer_encoder = train_model('ae_model', src_paths)
    # compressed_infersent_embedding = ae_infer_encoder.predict(x=inferSent_embedding)[0]
    # print('InferSent + AE: ', compressed_infersent_embedding)