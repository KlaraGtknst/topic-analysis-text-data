from read_pdf import *
from cli import *
import seaborn as sns
#from absl import logging

import tensorflow as tf

import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import seaborn as sns

'''------Code to compare documents in terms of similiarity-------
The code below is based on:
    https://github.com/tensorflow/docs/blob/master/site/en/hub/tutorials/semantic_similarity_with_tf_hub_universal_encoder.ipynb
    https://www.tensorflow.org/hub/tutorials/semantic_similarity_with_tf_hub_universal_encoder

run this code by typing and altering the path:
    python3 universal_sent_encoder_tensorFlow.py -d '/Users/klara/Downloads/*.pdf' -o '/Users/klara/Downloads/'
    python3 universal_sent_encoder_tensorFlow.py -d '/Users/klara/Documents/Uni/bachelorarbeit/data/0/*.pdf' -o '/Users/klara/Downloads/'
'''


def embed(input: list, model) -> np.ndarray:
    '''
    :param input: list of strings to be embedded
    :param model: trained model
    :return: numpy array of embeddings
    '''
    return model(input)


def plot_similarity(labels: list, features: np.ndarray, outpath: str = None) -> None:
    '''
    :param labels: list of labels; a label is the (beginning of the) text of a document
    :param features: numpy array of embeddings of the documents
    :param outpath: path to save plot; if not set the plot is not saved
    :return: None
    '''
    plt.figure(figsize=(8,8))
    corr = np.inner(features, features)
    sns.set(font_scale=(0.5 if len(labels) < 25 else 0.25))
    g = sns.heatmap(
        corr,
        xticklabels=labels,
        yticklabels=labels,
        vmin=0,
        vmax=1,
        cmap="YlOrRd")
    g.set_xticklabels(labels, rotation=(0 if len(labels) < 25 else 90))
    g.set_yticklabels(labels, rotation=(90 if len(labels) < 25 else 0))
    title = 'Semantic Textual Similarity'
    plt.title(title)
    if outpath:
        plt.savefig(outpath + '/' + title + '.pdf', format="pdf", bbox_inches="tight")
    plt.show()

def run_and_plot(messages: list, model, num_chars: int = 300, outpath: str = None) -> None:
    '''
    :param messages: list of strings to be embedded
    :param model: trained model
    :param num_chars: number of characters to be displayed as label; if not set 300 characters are displayed
    :param outpath: path to save plot; if not set the plot is not saved
    :return: None
    '''
    message_embeddings_ = embed(messages, model)
    labels = [msg[0:num_chars] + '...' for msg in messages]
    plot_similarity(labels, message_embeddings_, outpath=outpath)
        
def print_info_abt_embeddings(message_embeddings: list, messages: list) -> None:
    '''
    :param message_embeddings: list of embeddings
    :param messages: list of strings to be embedded
    :return: None
    '''
    print('num of messages: ', len(messages))
    for i, message_embedding in enumerate(np.array(message_embeddings).tolist()):
        print("Message: {}".format(messages[i]))
        print("Embedding size: {}".format(len(message_embedding)))
        message_embedding_snippet = ", ".join((str(x) for x in message_embedding[:3]))
        print("Embedding: [{}, ...]\n".format(message_embedding_snippet))

if __name__ == '__main__':
    args = arguments()
    file_paths = get_input_filepath(args)
    outpath = get_filepath(args, option='output')

    module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
    model = hub.load(module_url)

    messages = []
    for path in file_paths:
        text = pdf_to_str(path)
        messages.append(text)

    run_and_plot(messages, model, 10, outpath)

    # get embedding for single document
    embedding = embed([messages[0]], model)
    #print((embedding).numpy().tolist()[0])

    # get embedding for all documents in a folder, find out how to access the embeddings of the single documents
    embeddings = embed(messages, model)
    #print('difference between embedding of single document and the embedding in the matrix containing all embeddings:\n', embeddings[0] - embedding)

    #print_info_abt_embeddings(embeddings, messages)