import base64
import glob
import io
import pickle
from tkinter import *

from elasticsearch import Elasticsearch
from text_embeddings.universal_sent_encoder_tensorFlow import *
from text_embeddings.hugging_face_sentence_transformer import *
from elasticSearch.queries.query_documents_tfidf import *
from gensim.models.doc2vec import Doc2Vec
from PIL import Image
from tensorflow.python.keras.models import model_from_json
from gensim.test.utils import get_tmpfile

SRC_PATH = '/Users/klara/Documents/uni/bachelorarbeit/data/0/*.pdf'
DOC_PATH = '/Users/klara/Downloads/*.pdf'
NUM_DIMENSIONS = 55
NUM_COMPONENTS = 2
NUM_RESULTS = 4

def save_model(model, model_name):

    if not os.path.exists('models'):
        os.mkdir('models')

    if 'doc2vec' in model_name:
        model.save(f'models/{model_name}.pkl')
        return
    
    elif 'tfidf' in model_name:
        with open(f'models/{model_name}_vectorizer.pk', 'wb') as fin:
            pickle.dump(model, fin)
    print(" 4 Saved model to disk")


def load_model(model_name):
    if 'doc2vec' in model_name:
        return Doc2Vec.load('models/doc2vec_model.pkl')
    elif 'universal' in model_name:
        return google_univ_sent_encoding_aux()
    elif 'hugging' in model_name:
        return init_hf_sentTrans_model()
    elif 'infer' in model_name:
        print('help')
    elif 'tfidf' in model_name:
        return pickle.load(open('models/tfidf_vectorizer.pk', 'rb'))
    else:
        print(f'{model_name} not found')

def main(path=None):
    path = glob.glob('/Users/klara/Downloads/*.pdf')[0]
    # Doc2Vec
    '''train_corpus = list(db_elasticsearch.get_tagged_input_documents(src_paths=glob.glob(SRC_PATH)))
    d2v_model = Doc2Vec(train_corpus, vector_size=NUM_DIMENSIONS, window=2, min_count=2, workers=4, epochs=40)
    model_name = 'doc2vec_model'
    
    save_model(d2v_model, model_name)

    d2v_model = Doc2Vec.load(f'models/{model_name}.pkl')
    print(d2v_model.infer_vector(simple_preprocess(pdf_to_str(path))))'''

    # TF-IDF
    src_paths='/Users/klara/Documents/Uni/bachelorarbeit/data/0/*.pdf'
    client_addr="http://localhost:9200"
    client = Elasticsearch(client_addr)
    src_paths = glob.glob(src_paths)
    docs = get_docs_from_file_paths(src_paths)
    sim_docs_tfidf = TfidfVectorizer(input='content', preprocessor=TfidfTextPreprocessor().transform, min_df=3, max_df=int(len(docs)*0.07))
    sim_docs_document_term_matrix = sim_docs_tfidf.fit(docs)
    save_model(sim_docs_tfidf, 'tfidf')
    model = load_model('tfidf')
    embedding = model.transform([pdf_to_str(path)])
    embedding = np.ravel(embedding.todense())
    embedding = np.append(embedding, 1 if np.array([entry  == 0 for entry in embedding]).all() else 0)
    print(embedding)
   