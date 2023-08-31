from text_embeddings.preprocessing.read_pdf import *
from user_interface.cli import *
from elasticSearch.queries.query_documents_tfidf import get_docs_from_file_paths
import nltk
from text_embeddings.InferSent.models import InferSent
import torch



'''------Code to encode documents as sentence embeddings using pretrained models (InferSent)-------
Since the model is pretrained and the output embedding is a vector of size 4096, the model has to be trained again to fit the database's maximum dense vector size.

The code below is based on:
    https://github.com/facebookresearch/InferSent
    https://github.com/facebookresearch/InferSent/blob/main/models.py
    https://www.kaggle.com/code/jacksoncrow/infersent-demo
    https://morioh.com/a/95a832e85c0d/infersent-sentence-embeddings

run this code by typing and altering the path:
    python3 infer_pretrained.py -d '/Users/klara/Downloads/*.pdf' -o '/Users/klara/Downloads/'
    python3 infer_pretrained.py -d '/Users/klara/Documents/Uni/bachelorarbeit/data/0/*.pdf' -o '/Users/klara/Downloads/'
'''

def init_infer(model_path: str, w2v_path: str, file_paths: list, version: int = 1) -> tuple:
    params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048, # value bigger than maximum database vector size, change non-trivial since pre-trained model  has to be trained again
                    'pool_type': 'max', 'dpout_model': 0.0, 'version': version}
    infersent = InferSent(params_model)
    infersent.load_state_dict(torch.load(model_path))
    infersent.set_w2v_path(w2v_path)
    docs = get_docs_from_file_paths(file_paths)
    infersent.build_vocab(docs, tokenize=True)

    return infersent, docs


if __name__ == '__main__':
    args = arguments()
    file_paths = get_input_filepath(args)
    outpath = get_filepath(args, option='output')

    nltk.download('punkt')
    V = 1   # trained with GloVe
    MODEL_PATH = '/Users/klara/Developer/Uni/encoder/infersent%s.pkl' % V
    W2V_PATH = '/Users/klara/Developer/Uni/GloVe/glove.840B.300d.txt'

    infersent, docs = init_infer(model_path=MODEL_PATH, w2v_path=W2V_PATH, file_paths=file_paths, version=V)
    
    embeddings = infersent.encode(docs, tokenize=True)

    print(embeddings.shape)
    infersent.visualize('A man plays an instrument.', tokenize=True)

    # retrain model (cf. https://github.com/parasdahal/infersent/blob/master/train.py)
    # project does not support training: https://github.com/facebookresearch/InferSent/issues/82