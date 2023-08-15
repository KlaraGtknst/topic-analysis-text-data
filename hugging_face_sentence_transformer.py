from read_pdf import *
from cli import *
from query_documents_tfidf import get_docs_from_file_paths
import numpy as np
from sentence_transformers import SentenceTransformer



'''------Code to encode documents as sentence embeddings using pretrained models (from Hugging Face)-------
Since the model is pretrained and the output embedding is a vector of size 4096, the model has to be trained again to fit the database's maximum dense vector size.

The code below is based on:
    https://huggingface.co/sentence-transformers

run this code by typing and altering the path:
    ### CREATE/ SAVE MODEL ###
    python3 hugging_face_sentence_transformer.py -d '/Users/klara/Downloads/*.pdf' -o '/Users/klara/Developer/Uni/hugging_face_sentence_transformer'
    python3 hugging_face_sentence_transformer.py -d '/Users/klara/Documents/Uni/bachelorarbeit/data/0/*.pdf' -o '/Users/klara/Developer/Uni/hugging_face_sentence_transformer'

    ### LOAD MODEL ###
    python3 hugging_face_sentence_transformer.py -d '/Users/klara/Developer/Uni/hugging_face_sentence_transformer'
'''

def init_model() -> SentenceTransformer:
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    return model

def save_model_to_disk(model: SentenceTransformer, outpath: str) -> None:
    model.save(outpath)

def load_model(model_path: str) -> SentenceTransformer:
    model = SentenceTransformer(model_path)
    return model


if __name__ == '__main__':
    args = arguments()
    file_paths = get_input_filepath(args)
    outpath = get_filepath(args, option='output')

    docs = get_docs_from_file_paths(file_paths)
    # Choose wether to train a new model or load an existing one
    #model = init_model()   # new model
    model = load_model(model_path=file_paths[0])      # existing model

    # Comment if you don't want to save the model
    #save_model_to_disk(model=model, outpath=outpath)
    
    #Sentences we want to encode. Example:
    sentence = docs[0]

    #Sentences are encoded by calling model.encode()
    embedding = model.encode(sentence)

    print(embedding.shape)
    print(embedding)