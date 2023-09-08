import base64
import glob
import io
from tkinter import *

from elasticsearch import Elasticsearch
import torch
from text_visualizations import visualize_texts
from elasticSearch.queries import query_database
from text_embeddings.universal_sent_encoder_tensorFlow import *
from text_embeddings.hugging_face_sentence_transformer import *
from elasticSearch.queries.query_documents_tfidf import *
from elasticSearch import db_elasticsearch
from gensim.models.doc2vec import Doc2Vec
from doc_images import pdf_matrix, convert_pdf2image
from PIL import Image
from tensorflow.python.keras.models import model_from_json
from gensim.test.utils import get_tmpfile

SRC_PATH = '/Users/klara/Documents/uni/bachelorarbeit/data/0/*.pdf'
DOC_PATH = '/Users/klara/Downloads/*.pdf'
NUM_DIMENSIONS = 55
NUM_COMPONENTS = 2
NUM_RESULTS = 4

def save_model(model, model_name):
    try:
        if not os.path.exists('models'):
            os.mkdir('models')

        if 'doc2vec' in model_name:
            model.save(f'models/{model_name}.pkl')
            return
        
        # serialize model to JSON
        model_json = model.to_json()
        # TODO: fix
        
        with open(f"models/{model_name}.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights(f"models/{model_name}.h5")
    except Exception as e:
        print(e)
        model_path = f'models/{model_name}.pth'
        torch.save(model, model_path)
    print(" 4 Saved model to disk")


def load_model(model_name):
    if 'doc2vec' in model_name:
        return Doc2Vec.load(f'models/{model_name}')
    elif 'universal' in model_name:
        return google_univ_sent_encoding_aux()
    elif 'hugging' in model_name:
        return init_hf_sentTrans_model()
    elif 'infer' in model_name:
        print('help')
    elif 'tfidf' in model_name:
        return tfidf_aux()
    try:
        # load json and create model
        json_file = open(f'models/{model_name}.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights(f"models/{model_name}.h5")
    except:
        loaded_model = torch.load(f'models/{model_name}.pth')
    print("Loaded model from disk")
    return loaded_model

if __name__ == '__main__':
    train_corpus = list(db_elasticsearch.get_tagged_input_documents(src_paths=glob.glob(SRC_PATH)))
    d2v_model = Doc2Vec(train_corpus, vector_size=NUM_DIMENSIONS, window=2, min_count=2, workers=4, epochs=40)
    model_name = 'doc2vec_model'
    path = glob.glob('/Users/klara/Downloads/*.pdf')[0]
    save_model(d2v_model, model_name)

    d2v_model = Doc2Vec.load(f'models/{model_name}.pkl')
    print(d2v_model.infer_vector(simple_preprocess(pdf_to_str(path))))