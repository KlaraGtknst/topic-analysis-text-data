from gensim.models.doc2vec import Doc2Vec
import glob
from text_embeddings.save_models import get_tagged_input_documents


def main(file_paths):
    NUM_DIMENSIONS = 55

    train_corpus = list(get_tagged_input_documents(src_paths=file_paths))
    #d2v_model = Doc2Vec(train_corpus, vector_size=NUM_DIMENSIONS, window=2, min_count=2, workers=4, epochs=40)

    d2v_model2 = Doc2Vec(train_corpus)   # default dim 100! 
    print(d2v_model2)

    #print(len(d2v_model.infer_vector(train_corpus[0].words)))
    print(len(d2v_model2.infer_vector(train_corpus[0].words)))
