from read_pdf import *
from cli import *
from query_documents_tfidf import get_docs_from_file_paths
import seaborn as sns
import nltk
from models import InferSent
import torch



'''------Code to encode documents as sentence embeddings using pretrained models (InferSent)-------
The code below is based on:
    https://github.com/facebookresearch/InferSent
    https://github.com/facebookresearch/InferSent/blob/main/models.py

run this code by typing and altering the path:
    python3 infer_pretrained.py -d '/Users/klara/Downloads/*.pdf' -o '/Users/klara/Downloads/'
    python3 infer_pretrained.py -d '/Users/klara/Documents/Uni/bachelorarbeit/data/0/*.pdf' -o '/Users/klara/Downloads/'
'''


if __name__ == '__main__':
    args = arguments()
    file_paths = get_input_filepath(args)
    outpath = get_filepath(args, option='output')

    nltk.download('punkt')
    
    V = 1   # trained with GloVe
    MODEL_PATH = '/Users/klara/Developer/Uni/encoder/infersent%s.pkl' % V
    params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                    'pool_type': 'max', 'dpout_model': 0.0, 'version': V}
    infersent = InferSent(params_model)
    infersent.load_state_dict(torch.load(MODEL_PATH))

    W2V_PATH = '/Users/klara/Developer/Uni/GloVe/glove.840B.300d.txt'#'fastText/crawl-300d-2M.vec'
    infersent.set_w2v_path(W2V_PATH)

    docs = get_docs_from_file_paths(file_paths)
    infersent.build_vocab(docs, tokenize=True)
    embeddings = infersent.encode(docs, tokenize=True)
    infersent.visualize('A man plays an instrument.', tokenize=True)