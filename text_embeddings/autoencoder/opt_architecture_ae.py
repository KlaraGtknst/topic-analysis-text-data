import json
from multiprocessing import Pool
import os
import statistics
import sys
from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from elasticSearch.models_aux import get_models, get_tfidf_emb 
from elasticSearch.recursive_search import chunks, scanRecurse
from elasticSearch.selected_docs import select_rep_path
from text_embeddings.InferSent.models import InferSent
import torch
from text_embeddings.preprocessing.read_pdf import pdf_to_str
import torch
import torch.nn as nn
import optuna

LATENT_SHAPE = 2048
INPUT_SHAPE = 4096
class wrapper:
    def __init__(self, study_name):
        self.study_name = study_name

    def __call__(self, layers_lst: list):
        # optuna
        search_space = {
        'layers_num': layers_lst
        }
        study = optuna.create_study(sampler=optuna.samplers.GridSampler(search_space), direction='minimize', study_name=self.study_name)
        study.optimize(objective, n_trials=3*3)
        print('Best hyperparams found by Optuna: \n', study.best_params)

class AE(nn.Module):
    def __init__(self, layer_sizes):
        super(AE, self).__init__()
        self.loss_function = nn.MSELoss()
        
        self.encoder = nn.ModuleList([
            nn.Sequential(
                nn.Linear(layer_sizes[i], layer_sizes[i+1]),
                nn.LeakyReLU()
            ) for i in range(len(layer_sizes)-1)
        ])

        self.decoder  = nn.ModuleList([
            nn.Sequential(
                nn.Linear(layer_sizes[-i], layer_sizes[-(i+1)]),
                nn.LeakyReLU() # beim letzten kein LeakyRelu
            ) for i in range(1, len(layer_sizes)-1)
        ])

        self.decoder.append(nn.Linear(layer_sizes[1], layer_sizes[0]))

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
        return loss



def eval(X_test, inv_embs):
    X_test = X_test.detach().numpy()
    inv_embs = inv_embs.detach().numpy()

    rsme = np.linalg.norm(inv_embs - X_test) / np.sqrt(X_test.shape[0])

    # cosine similarity
    cos_sim = statistics.mean([np.dot(inverse_emb,embedding)/(np.linalg.norm(inverse_emb)*np.linalg.norm(embedding)) for inverse_emb, embedding in zip(inv_embs, X_test)])
    
    return {'rsme': rsme, 'cosine_similarity': cos_sim}


 # Objective function to optimize by OPTUNA
def objective(trial):
    n_layer = trial.suggest_int("layers_num", 2,20)
    layer_size = get_layer_config(n_layer)

    model = AE(layer_size)

    # data
    # infersent
    baseDir, resDir = get_directories()

    # result dict
    ae_config_scores = json.load(open(resDir + 'ae_configs.json')) if os.path.exists(resDir + 'ae_configs.json') else {}

    infer_embeddings = get_infer_emb(baseDir)
    #tfidf_embeddings = get_tfidf_emb(models['tfidf'], docs)
    print('got embeddings')

    # infersent
    X_train, X_test = train_test_split(infer_embeddings, test_size=0.2, random_state=42)
    X_train = torch.from_numpy(X_train)
    X_test = torch.from_numpy(X_test)


    # train
    N_EPOCHS = 20
    losses = []
    for _ in range(N_EPOCHS):
        for x in X_train:
            losses.append(model.train(x).detach().numpy())

    # plot_losses(losses)

    # eval 
    scores = eval(X_test=X_test, inv_embs=model.forward(X_test))
    print('layer config: ', layer_size, 'scores: ', scores)

    # save results
    save_results('infer', layer_size, resDir, ae_config_scores, scores)

    return scores['rsme']

def save_results(model: str, layer_sizes:list, resDir:str, ae_config_scores:dict, scores:dict):
    ae_config_scores[model + '_' + str(layer_sizes)] = {'rsme': str(scores['rsme']), 'cosine_similarity': str(scores['cosine_similarity'])}

    out_file = open(resDir + 'ae_configs.json','w+')
    json.dump(ae_config_scores, out_file)

def plot_losses(losses):
    losses = losses
    plt.style.use('fivethirtyeight')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    
    # Plotting the last 100 values
    plt.plot(losses[-min(100, len(losses)):])
    plt.show()

def get_layer_config(n_layer):
    step_size = int((LATENT_SHAPE-INPUT_SHAPE)/(min(1,n_layer-1)))
    layer_size = list(np.arange(INPUT_SHAPE, LATENT_SHAPE+1, step=step_size))
    if len(layer_size) > n_layer-1:
        layer_size = layer_size[:-1]
    layer_size.append(LATENT_SHAPE)
    return layer_size

def get_infer_emb(baseDir:str):
    paths = select_rep_path(baseDir, 10) if baseDir.startswith('/mnt/') else list(scanRecurse(baseDir))

    docs = [pdf_to_str(path) for path in paths]
    
    models = get_models(baseDir, model_names = ['infer'])#, 'tfidf'])   
    print('got models')
    infer_embeddings = models['infer'].encode(docs, tokenize=True)
    return infer_embeddings

def get_directories():
    baseDir = '/mnt/datasets/Bahamas/'
    resDir = '/mnt/stud/home/kgutekunst/logs/'
    if os.path.exists('/Users/klara/Documents/uni/bachelorarbeit/data/0/'):
        baseDir = '/Users/klara/Documents/uni/bachelorarbeit/data/0/'
        resDir = '/Users/klara/Developer/Uni/topic-analysis-text-data/results/'
    return baseDir,resDir




def main(src_path, num_cpus:int):
    print('ae config on ', num_cpus, ' layers & cpus')
    max_layer = num_cpus
    n_layers = list(range(2,max_layer+2))
    sub_lists = list(chunks(n_layers, len(n_layers)//num_cpus))
    print(sub_lists)
    sys.stdout.flush()

    with Pool(processes=num_cpus) as pool:
        proc_wrap = wrapper('ae-opt-infer')
        print('initialized wrapper')
        sys.stdout.flush()
        pool.map(proc_wrap, sub_lists)

        



if(__name__ == "__main__"):
    # layersizes=[10,5]
    # ae = AE(layersizes)
    # x = torch.randn(32,10)
    # res = ae(x)
    # print(res)
    # print(res.shape)
    main('', 1)

