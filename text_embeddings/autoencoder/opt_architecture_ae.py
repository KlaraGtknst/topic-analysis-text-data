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
import torch
from sklearn.model_selection import GridSearchCV

from text_embeddings.preprocessing.read_pdf import pdf_to_str
import torch
import torch.nn as nn
import optuna

LATENT_SHAPE = 2048
INPUT_SHAPE = 4096

class AE(nn.Module):
    def __init__(self, layer_sizes):
        super(AE, self).__init__()
        self.loss_function = nn.MSELoss()
        
        self.encoder = nn.ModuleList([
            nn.Sequential(
                nn.Linear(layer_sizes[i], layer_sizes[i+1]),
                nn.ReLU()
            ) for i in range(len(layer_sizes)-1)
        ])

        self.decoder  = nn.ModuleList([
            nn.Sequential(
                nn.Linear(layer_sizes[-i], layer_sizes[-(i+1)]),
                nn.ReLU()
            ) for i in range(1, len(layer_sizes))
        ])
        
        self.optimizer = torch.optim.Adam(params=self.parameters(), lr=0.005)


    def forward(self, x):
        # run through encoder
        for layer in self.encoder:
            x = layer(x)

        # run though decoder
        for layer in self.decoder:
            x = layer(x)
        
        return x

    
    def train(self, x):
        # forward pass
        y = self.forward(x)

        # compute error (RMSE)
        loss = self.loss_function(y, x)

        # zero gradients, backward pass, update weights
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()



def eval(autoencoder_model, X_test, model_name:str, neurons_per_layer:list):
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


 # Objective function to optimize by OPTUNA
def objective(trial):
    n_layer = trial.suggest_int("layers_num", np.arange(2,9))
    step_size = int((LATENT_SHAPE-INPUT_SHAPE)/n_layer)
    layer_size = list(np.arange(INPUT_SHAPE, LATENT_SHAPE, step=step_size, dytpe=int))

    model = AE(layer_size)
    model.compile(optimizer='adam', loss='mse')
    # Implement early stopping criterion. 
    # Training process stops when there is no improvement during 50 iterations
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=50)
    X_train = np.random(4096)
    history = model.train(X_train)
    return history.history["loss"][-1]


def main(src_path):
    # paths = list(scanRecurse(src_path))
    # docs = [pdf_to_str(path) for path in paths]
    # print('got paths')

    # # infersent & tfidf
    # models = get_models(src_path, model_names = ['infer'])#, 'tfidf'])   
    # print('got models')
    # infer_embeddings = models['infer'].encode(docs, tokenize=True)
    # #tfidf_embeddings = get_tfidf_emb(models['tfidf'], docs)
    # print('got embeddings')

    # # infersent
    # X_train, X_test = train_test_split(infer_embeddings, test_size=0.2, random_state=42)

    # if os.path.exists('results/score_per_ae_architecture.json'):
    #     scores = pd.read_json('results/score_per_ae_architecture.json')
    #     scores = pd.DataFrame(scores)
    # else:
    #     scores = pd.DataFrame(columns=['model name', 'architectur', 'loss', 'rsme', 'cosine similarity'])


    # test architectures using grid search
   
    
    params = [{'n_layers': [3000]}]


    # init model
    #grid = GridSearchCV(estimator=AE, param_grid=params)
    model = AE([4096, 3000, 2048])

    # train model
    X_train = torch.rand(4096)
    X_test = torch.rand(4096)
    print(len(X_train))
    model.train(X_train)

    # forward pass to get decompressed input
    print(len(model(X_test)))



    # for architecture in infer_enc_arch:
    #     infer_score = train_and_evaluate_autoencoder(neurons_per_layer=architecture, X_train=X_train, X_test=X_test, model_name='infer')
    #     scores.update(pd.DataFrame(infer_score))

    # if tfidf_embeddings.shape[0] > 2048:
    #     for architecture in infer_enc_arch:
    #         tfidf_score = train_and_evaluate_autoencoder(neurons_per_layer=architecture, X_train=X_train, X_test=X_test, model_name='tfidf')
    #         scores.update(pd.DataFrame(tfidf_score))



