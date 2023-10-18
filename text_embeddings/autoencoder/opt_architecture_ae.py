import os
import statistics
import tensorflow as tf
import numpy as np
# from tensorflow.python.keras.models import Sequential, Model
# from tensorflow.python.keras.layers import Dense
from tensorflow import keras
from sklearn.model_selection import train_test_split
from itertools import product
from elasticSearch.models_aux import get_models, get_tfidf_emb
from elasticSearch.recursive_search import scanRecurse
from text_embeddings.InferSent.models import InferSent
import pandas as pd

from text_embeddings.preprocessing.read_pdf import pdf_to_str

def train_and_evaluate_autoencoder(neurons_per_layer: list, X_train, X_test, model_name:str, input_dim:tuple=(4096,), latent_dim:tuple=(2048,)):
    num_layers = len(neurons_per_layer)

    print(neurons_per_layer, num_layers)

    autoencoder_model = keras.models.Sequential()
    
    # Encoder
    autoencoder_model.add(keras.layers.Dense(input_dim[0], input_dim=input_dim[0], activation='relu'))
    for layer_num in range(num_layers):
        print(neurons_per_layer[layer_num])
        autoencoder_model.add(keras.layers.Dense(neurons_per_layer[layer_num], activation='relu'))
    autoencoder_model.add(keras.layers.Dense(latent_dim[0], input_dim=neurons_per_layer[-1], activation='relu'))

    # Decoder
    autoencoder_model.add(keras.layers.Dense(latent_dim[0], input_dim=latent_dim[0], activation='relu'))
    for layer_num in range(1, num_layers + 1):
        print(neurons_per_layer[-layer_num])
        autoencoder_model.add(keras.layers.Dense(neurons_per_layer[-layer_num], activation='relu'))
    autoencoder_model.add(keras.layers.Dense(latent_dim[0], input_dim=neurons_per_layer[0], activation='relu'))

    
    autoencoder_model.compile(optimizer='adam', loss='mean_squared_error')
    print(autoencoder_model.layers)
    print(X_train.shape)
    autoencoder_model.fit(X_train, X_train, epochs=20, batch_size=256, shuffle=True, validation_data=(X_train, X_train))


    # Get the encoder layers & model
    # encoder_layers = autoencoder_model.layers[:num_layers-1]
    # encoder_model = Model(inputs=autoencoder_model.input, outputs=encoder_layers[-1].output)
    
    # Evaluate the model
    loss = autoencoder_model.evaluate(X_test, X_test)

    # RSME
    inv_embs = autoencoder_model.predict(X_test)
    rsme = np.linalg.norm(inv_embs - X_test) / np.sqrt(X_test.shape[0])
    print('RMSE: ', rsme)
    # cosine similarity
    cos_sim = statistics.mean([np.dot(inverse_emb,embedding)/(np.linalg.norm(inverse_emb)*np.linalg.norm(embedding)) for inverse_emb, embedding in zip(inv_embs, X_test)])
    print('cosine similarity: ', cos_sim)
    
    return {'model name': model_name, 'architectur': neurons_per_layer, 'loss': loss, 'rsme':rsme, 'cosine similarity': cos_sim}





def main(src_path):
    paths = list(scanRecurse(src_path))
    docs = [pdf_to_str(path) for path in paths]
    print('got paths')

    # infersent & tfidf
    models = get_models(src_path, model_names = ['infer'])#, 'tfidf'])   
    print('got models')
    infer_embeddings = models['infer'].encode(docs, tokenize=True)
    #tfidf_embeddings = get_tfidf_emb(models['tfidf'], docs)
    print('got embeddings')

    # infersent
    X_train, X_test = train_test_split(infer_embeddings, test_size=0.2, random_state=42)

    if os.path.exists('results/score_per_ae_architecture.json'):
        scores = pd.read_json('results/score_per_ae_architecture.json')
        scores = pd.DataFrame(scores)
    else:
        scores = pd.DataFrame(columns=['model name', 'architectur', 'loss', 'rsme', 'cosine similarity'])

    # test architectures
    input_shape = 4096
    latent_shape = 2048

    infer_enc_arch = [[3500]]

    for architecture in infer_enc_arch:
        infer_score = train_and_evaluate_autoencoder(neurons_per_layer=architecture, X_train=X_train, X_test=X_test, model_name='infer')
        scores.update(pd.DataFrame(infer_score))

    # if tfidf_embeddings.shape[0] > 2048:
    #     for architecture in infer_enc_arch:
    #         tfidf_score = train_and_evaluate_autoencoder(neurons_per_layer=architecture, X_train=X_train, X_test=X_test, model_name='tfidf')
    #         scores.update(pd.DataFrame(tfidf_score))

    print(scores)

