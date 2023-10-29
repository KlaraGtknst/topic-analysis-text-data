

import ast
import random
import re

from elasticsearch import Elasticsearch
import numpy as np
import pandas as pd
from constants import *
from elasticSearch.create_documents import get_hash_file
from elasticSearch.queries.query_database import get_knn_res
from elasticSearch.recursive_search import scanRecurse
import venn
from matplotlib import pyplot as plt
import seaborn as sns
#from matplotlib_venn import venn2


def create_sim_docs_log(baseDir:str, client_addr:str=CLIENT_ADDR, num_rep_docs:int=100, n_res:int=10):
    '''
    :param baseDir: directory where documents are stored
    :param client_addr: address of elastic search client
    :param num_rep_docs: number of documents to query for
    :param n_res: number of results to return
    :return: df which stores most similar documents for each model for each query

    logs query results for similar documents
    '''
    model_names = get_model_names()
    docs_to_search_for = random.sample(list(scanRecurse(baseDir=baseDir)), num_rep_docs)
    ids = [get_hash_file(path) for path in docs_to_search_for]
    client = Elasticsearch(client_addr)
    res_dict = pd.DataFrame(columns=['search_id', *model_names])
    res_dict.set_index('search_id', inplace=True)
    for id in ids:
        row = {}
        for model in model_names: 
            res = get_knn_res(doc_to_search_for=id, elastic_search_client=client, query_type=MODELS2EMB[model], n_results=n_res)
            if type(res)==dict:
                continue
            row[model] = [cell['_id'] for cell in res]
        res_dict.loc[id] = row
    return res_dict

def save_to_dir(df:pd.DataFrame, resDir:str, file_name:str):
    '''
    :param df: df which stores most similar documents for each model for each query
    :param resDir: directory where results are saved
    :param file_name: name of file where results are saved
    :return: -
    '''
    df.to_csv(resDir + file_name, index_label='search_id', columns=df.columns)


def read_from_dir(path_to_file:str):
    '''
    :param path_to_file: path to csv file which stores results from most similar documents
    :return: df
    '''
    return pd.read_csv(filepath_or_buffer=path_to_file, sep=',', index_col='search_id')


def visualize_using_venn(df:pd.DataFrame, save:bool=False, resDir:str='results/'):
    '''
    :param df: df which stores most similar documents for each model for each query
    :param save: if True, venn diagram is saved to resDir
    :param resDir: directory where venn diagram is saved
    :return: -
    
    creates venn diagram for each query
    '''
    model_names = get_model_names()
    dataset = {}
    # for i, search_id in enumerate(df.index):
    #     if i == 6:
    #         break

    #     print(df.loc[search_id].values.flatten())
    #     dataset[search_id] = set(df.loc[search_id].values.flatten())
    for model in model_names:
        dataset[model] = set(df.loc[df.index[0],model])

    print(dataset)
    labels = venn.get_labels(dataset.values(), fill=['number'])
    print(labels)
    fig, ax = venn.venn6(labels, names=dataset.keys())
    title = 'Venn diagram of most similar documents'
    plt.title(title)
    if save:
        plt.savefig(resDir + re.sub(' ', '_', string=title), bbox_inches = 'tight')
    #plt.legend(dataset.keys())
    plt.show()
    # venn2(subsets = dataset.values(), set_labels = dataset.keys())
    # plt.show()

def df_str2list(df:pd.DataFrame):
    for row_id, row in df.iterrows():
        for col_id, cell in row.items():
            df.loc[row_id, col_id] = [img for img in ast.literal_eval(cell)]
    return df


def encode_lists(df:pd.DataFrame):
    '''
    :param df: df which stores most similar documents for each model for each query
    :return: df with encoded lists
    '''
    df = df_str2list(df)
    ids = set()
    for row in list(df.values):
        for cell in row:
            for item in cell:
                ids.add(item)
    enc = {y: x+1 for x,y in enumerate(sorted(ids))}

    for row_id, row in df.iterrows():
        for col_id, cell in row.items():
            df.loc[row_id, col_id] = [enc[img] for img in cell]
    idx = df.index
    df.index = idx.map(lambda id: enc[id])
    return df

def get_model_names():
    '''
    :return: list of model names without ae
    '''
    model_names = MODEL_NAMES
    if 'ae' in model_names:
        model_names.remove('ae')
    return model_names

def similarity_matrix(df:pd.DataFrame):
    '''
    :param df: df which stores most similar documents for each model for each query
    :return: matrix

    returns a matrix which saves the total number of equal entries for two models over mutliple queries, 
    i.e. how many 'most similar' documents were identified by both models.
    '''
    model_names = get_model_names()
    sim_matr = np.matrix(np.zeros((len(model_names), len(model_names))))
    for id in df.index:
        for i, model in enumerate(model_names):
            for j in range(i, len(model_names)):
                sim_matr[i, j] += np.sum([df.loc[id, model_names[j]].count(item) for item in df.loc[id, model]])
                sim_matr[j, i] = sim_matr[i, j]
    return sim_matr

def create_sim_heatmap(res_df:pd.DataFrame, save:bool=False, resDir:str='results/'):
    '''
    :param res_df: df which stores most similar documents for each model for each query
    :param save: if True, heatmap is saved to resDir
    :param resDir: directory where heatmap is saved
    :return: -

    creates and possibly saves heatmap of similarity matrix.
    '''
    sim_matr = similarity_matrix(res_df)
    model_names = get_model_names()
    plt.figure(figsize=(8,5))
    ax = sns.heatmap(sim_matr, annot=True, yticklabels=model_names, xticklabels=model_names)
    ax.set(xlabel="model", ylabel="model")
    title = 'Number of equal query results'
    plt.title(title)
    plt.yticks(rotation=0) 
    if save:
        plt.savefig(resDir + re.sub(' ', '_', string=title), bbox_inches = 'tight')
    plt.show()

def main(baseDir:str):
    
    # res_df = create_sim_docs_log(baseDir=baseDir, num_rep_docs=10, n_res=3)
    # save_to_dir(resDir='results/', df=res_df, file_name='query_res.csv')

    res_df = read_from_dir('results/' + 'query_res.csv')
    #print(res_df)
    
    res_df = encode_lists(res_df)
    #visualize_using_venn(res_df)

    create_sim_heatmap(res_df, save=True)

