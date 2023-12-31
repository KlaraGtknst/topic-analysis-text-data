

import ast
import operator
import os
import random
import re
import cv2

from elasticsearch import Elasticsearch
import numpy as np
import pandas as pd
from constants import *
from elasticSearch.create_documents import get_hash_file
from elasticSearch.queries.query_database import get_knn_res, get_doc_meta_data
from elasticSearch.recursive_search import scanRecurse
from doc_images.pdf_matrix import alter_axes
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
    model_names = ["pca_optics_cluster", "argmax_pca_cluster"]# get_model_names()   #TODO
    all_paths = list(scanRecurse(baseDir=baseDir))
    docs_to_search_for = random.sample(all_paths, num_rep_docs) if num_rep_docs < len(all_paths) else all_paths
    ids = [get_hash_file(path) for path in docs_to_search_for]
    client = Elasticsearch(client_addr)
    res_dict = pd.DataFrame(columns=['search_id', *model_names])
    res_dict.set_index('search_id', inplace=True)
    for id in ids:
        row = {}
        for model in model_names: 
            res = get_knn_res(doc_to_search_for=id, elastic_search_client=client, query_type=MODELS2EMB[model], n_results=n_res + 1)
            if type(res)==dict:
                continue
            for cell in res:
                if id == cell['_id']: # remove query document from results
                    res.remove(cell)
            res = res[:n_res]   # remove last element if more than n_res results, i.e. query id is not included
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


def visualize_using_venn(df:pd.DataFrame, save:bool=False, resDir:str='results/', title:str='Venn diagram of most similar documents'):
    '''
    :param df: df which stores most similar documents for each model for each query
    :param save: if True, venn diagram is saved to resDir
    :param resDir: directory where venn diagram is saved
    :return: -
    
    creates venn diagram for each query
    '''
    model_names = get_model_names()
    dataset = {}
    for model in model_names:
        dataset[model] = set()
        for cell in df.loc[:,model].values: # iterate over all queries 
            dataset[model].update(set(cell))    # set of all response irrespective of query
    labels = venn.get_labels(dataset.values(), fill=['number'])
    fig, ax = venn.venn6(labels, names=dataset.keys())
    plt.title(title)
    if save:
        plt.savefig(resDir + re.sub(' ', '_', string=title) + '.pdf', format='pdf', bbox_inches = 'tight')
    plt.show()

def df_str2list(df:pd.DataFrame):
    for row_id, row in df.iterrows():
        for col_id, cell in row.items():
            if type(cell) == list:
                return df
            else:
                df.loc[row_id, col_id] = [img for img in ast.literal_eval(cell)]
    return df


def encode_lists(df:pd.DataFrame):
    '''
    :param df: df which stores most similar documents for each model for each query
    :return: df with encoded lists
    '''
    df = df_str2list(df)
    ids = set(df.index)
    for row in list(df.values):
        for cell in row:
            for item in cell:
                ids.add(item)
    enc = {y: x+1 for x,y in enumerate(sorted(ids))}

    enc_df = pd.DataFrame(columns=df.columns)
    for row_id, row in df.iterrows():
        for col_id, cell in row.items():
            enc_df.loc[row_id, col_id] = [enc[img] for img in cell]
    idx = df.index
    enc_df.index = idx.map(lambda id: enc[id])
    return enc_df

def get_model_names():
    '''
    :return: list of model names without ae
    '''
    model_names = MODEL_NAMES
    if 'ae' in model_names:
        model_names.remove('ae')
    return model_names

def similarity_matrix(df:pd.DataFrame, normalize:bool=True):
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
    if normalize:
        sim_matr /= np.array(len(df.index)* len(df.iloc[0,0]))
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
    ax = sns.heatmap(sim_matr, annot=True, yticklabels=model_names, xticklabels=model_names, fmt='g')
    ax.set(xlabel="model", ylabel="model")
    title = 'Portion of equal query results ({} queries, {} responses/query, 2048 document corpus)'.format(len(res_df.index), len(res_df.iloc[0,0]))
    plt.title(title)
    plt.yticks(rotation=0) 
    if save:
        plt.savefig(resDir + re.sub(' ', '_', string=title) + '.pdf', format='pdf', bbox_inches = 'tight')
    plt.show()

def get_img_path(doc_path:str, image_baseDir:str):
    image_baseDir = image_baseDir if image_baseDir.endswith('/') else image_baseDir + '/'
    file_name = doc_path.split('/')[-1].split('.')[0]
    image_path = image_baseDir + file_name + '.png'
    return image_path

def add_border_around_subfig(ax):
    ax.patch.set_linewidth(10)
    ax.patch.set_edgecolor('pink')

def create_query_img(query: pd.DataFrame, image_baseDir:str, outpath:str='', client_addr:str=CLIENT_ADDR):
    '''
    :param query: a df which contains the id as a column, and columns for the most similar documents per embedding model (column name)
    '''
    # get paths to documents
    client = Elasticsearch(client_addr)
    for doc_idx in range(len(query.index)):
        meta_data = get_doc_meta_data(client, query.iloc[doc_idx].name)
        path = meta_data['path']

        # get path to query image
        image_path = get_img_path(doc_path=path, image_baseDir=image_baseDir)

        # get path to response document images
        for model in ["pca_optics_cluster", "argmax_pca_cluster"]:# get_model_names():   #TODO
            paths =[image_path]
            response_docs = query.iloc[doc_idx][model]
            for resp_doc in (response_docs if type(response_docs) == list else ast.literal_eval(response_docs)):
                doc_path = get_doc_meta_data(client, resp_doc)['path']
                paths.append(get_img_path(doc_path=doc_path, image_baseDir=image_baseDir))

            # display images
            fig, axs = plt.subplots(nrows=1, ncols=len(paths), figsize=(len(paths)*5,5))
            fig.subplots_adjust(hspace = .00, wspace= .00)
            for i, img in enumerate(paths):
                ax = fig.add_subplot(1, len(paths), i+1)
                if i == 0:
                    add_border_around_subfig(ax)
                axs[i].axis('off')
                alter_axes(ax)
                ax.set_xlabel('query document' if i==0 else '{}. response document'.format(i))
                image = cv2.imread(img, cv2.IMREAD_COLOR)
                plt.imshow(image)

            title = 'Most similar images found by {}'.format(model)
            plt.title(title)
            if outpath != '':
                # if not os.path.exists(outpath + 'png/'):
                #     os.mkdir(outpath + 'png/')
                if not os.path.exists(outpath + query.iloc[doc_idx].name):
                    os.mkdir(outpath + query.iloc[doc_idx].name)
                plt.savefig(outpath + query.iloc[doc_idx].name + '/' + re.sub(' ', '_', title) + '.pdf', format="pdf", bbox_inches="tight", dpi=1200)
            #plt.show()

def add_broder_around_figure(fig):
    fig.patch.set_linewidth(10)
    fig.patch.set_edgecolor('blue')



def same_query_stats(baseDir:str, client_addr:str=CLIENT_ADDR, num_rep_docs:int=10, n_res:int=10):
    '''
    :param df: df which stores a set of the most similar documents for each model for each query
    '''
    shared_docs = {}
    for trial in range(31):
        res_df = create_sim_docs_log(baseDir=baseDir, client_addr=client_addr, num_rep_docs=num_rep_docs, n_res=n_res)
        similarity_matr = similarity_matrix(res_df, normalize=True)
        for i, model in enumerate(get_model_names()):
            model_names = get_model_names()
            for j in range(i, len(model_names)):
                shared_docs.setdefault((model, model_names[j]), []).append(similarity_matr[i,j])
    mu_shared_docs = {}
    std_shared_docs = {}
    for key, value in shared_docs.items():
        mu_shared_docs[key] = np.mean(value).round(2)
        std_shared_docs[key] = np.std(value).round(2)
    
    shared_doc_stats = pd.DataFrame({'model 1': list(map(operator.itemgetter(0), mu_shared_docs)), 'model 2': list(map(operator.itemgetter(1), mu_shared_docs)), 'mean': list(mu_shared_docs.values()), 'std': list(std_shared_docs.values())})
    shared_doc_stats.set_index(['model 1', 'model 2'], inplace=True)
    shared_doc_stats.to_csv('results/shared_doc_statistics.csv', index=True)
    return mu_shared_docs, std_shared_docs


def main(baseDir:str):
    # same_query_stats(baseDir=baseDir, num_rep_docs=10, n_res=10)

    num_queries = 10
    for num_resp in [5]:#[3, 5, 10]:
        res_df = create_sim_docs_log(baseDir=baseDir, num_rep_docs=num_queries, n_res=num_resp)
        # save_to_dir(resDir='results/', df=res_df, file_name='{}_query_res_{}_resp.csv'.format(num_queries, num_resp))
        # res_df = read_from_dir('results/' + '{}_query_res_{}_resp.csv'.format(num_queries, num_resp))
        # res_df = encode_lists(res_df)
        # visualize_using_venn(res_df, save=True, title='Venn diagram ({} queries à {} responses in a 2048 document corpus)'.format(num_queries, num_resp))
        create_query_img(res_df, image_baseDir='/Users/klara/Documents/uni/bachelorarbeit/images', outpath='results/')

    # res_df = read_from_dir('results/' + 'query_res.csv')
    # create_sim_heatmap(res_df, save=True)

