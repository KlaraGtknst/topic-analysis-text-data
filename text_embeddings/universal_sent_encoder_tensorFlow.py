from text_embeddings.preprocessing.read_pdf import *
from user_interface.cli import *
import seaborn as sns
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
tf.random.set_seed(123)
np.random.seed(123)

'''------Code to compare documents in terms of similiarity-------
The code below is based on:
    https://github.com/tensorflow/docs/blob/master/site/en/hub/tutorials/semantic_similarity_with_tf_hub_universal_encoder.ipynb
    https://www.tensorflow.org/hub/tutorials/semantic_similarity_with_tf_hub_universal_encoder

run this code by typing and altering the path:
    python3 universal_sent_encoder_tensorFlow.py -d '/Users/klara/Downloads/*.pdf' -o '/Users/klara/Downloads/'
    python3 universal_sent_encoder_tensorFlow.py -d '/Users/klara/Documents/Uni/bachelorarbeit/data/0/*.pdf' -o '/Users/klara/Downloads/'
'''


def embed(input: list, model):
    '''
    :param input: list of strings to be embedded
    :param model: trained model
    :return: EagerTensor of embeddings
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

def run_and_plot(messages: list, model, num_chars: int = None, outpath: str = None) -> None:
    '''
    :param messages: list of strings to be embedded
    :param model: trained model
    :param num_chars: number of characters to be displayed as label; if not set 300 characters are displayed if the number of messages is less than 25, otherwise 10 characters are displayed.
    :param outpath: path to save plot; if not set the plot is not saved
    :return: None
    '''
    message_embeddings_ = embed(messages, model)
    num_chars = num_chars if num_chars else (200 if len(messages) < 25 else 10)
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

def google_univ_sent_encoding_aux():
    '''
    :param src_paths: paths to the documents to be inserted into the database
    :return: document-term matrix and the trained tfidf vectorizer model
    '''
    try:
        local_url = "/Users/klara/Developer/Uni/topic-analysis-text-data/models/universal-sentence-encoder_4"
        server_url = "/mnt/stud/work/kgutekunst/topic-analysis-text-data/models/universal-sentence-encoder_4"
        module_url = local_url if os.path.exists(local_url) else server_url
        #"https://tfhub.dev/google/universal-sentence-encoder/4"
        model = hub.load(module_url)
    except:
        module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/5"
        model = hub.load(module_url)
    return model

def main(file_paths, outpath):
    # if load of URL does not work, use: "https://tfhub.dev/google/universal-sentence-encoder/4", cf. https://www.kaggle.com/code/nicapotato/universal-sentence-encoder-semantic-similarity
    module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/5"
    model = hub.load(module_url)
    print ("module %s loaded" % module_url)

    messages = []
    for path in file_paths:
        # input is allowed to be not only lower case and not yet tokenized (cf. https://www.tensorflow.org/hub/tutorials/semantic_similarity_with_tf_hub_universal_encoder)
        text = pdf_to_str(path)
        messages.append(text)

    run_and_plot(messages, model, outpath=outpath)

    # get embedding for single document
    embedding = embed([messages[1]], model) # paper: Universal Sentence Encoder page 2
    embedding2 = embed([messages[1]], model)
    #print((embedding).numpy().tolist()[0])

    # get embedding for all documents in a folder, find out how to access the embeddings of the single documents
    embeddings = embed(messages, model) # https://www.tensorflow.org/hub/tutorials/semantic_similarity_with_tf_hub_universal_encoder
    print('\nmatrix of differences between embedding of single document and the embedding in the matrix containing all embeddings:\n', 
          embeddings[1].numpy() - embedding.numpy()[0])
    print('\nshape of single embedding and the one from a matrix: ',
          embedding.numpy()[0].shape, embeddings[1].numpy().shape)
    # TODO: hypothesis: 
    # (1) embedding is influenced by other documents in the input (context) 
    # or (2) model adapts to the input
    # or !(3) the embedding uses n-grams of documents close to current doc (like a window) to embed it, cf. DAN in https://amitness.com/2020/06/universal-sentence-encoder/
    print('\nsquared difference between single embedding and the one from a matrix:\n',
           sum((embedding.numpy()[0] - embeddings[1].numpy())**2))
    plt.figure(figsize=(12,7))
    plt.title('Difference between embedding of single document and the embedding in the matrix containing all embeddings')
    plt.plot(embedding.numpy()[0] - embeddings[1].numpy(), color='green')
    plt.yscale('symlog')
    plt.show()
    print('\nsquared difference between single embeddings of the same document:\n',
          sum((embedding.numpy()[0] - embedding2.numpy()[0])**2))
    #print_info_abt_embeddings(embeddings, messages)

    # https://github.com/tensorflow/hub/issues/658
    # TODO: in BA schreiben: bei diesen Modell gibt es Schwankungen mit Batch Übergabe -> liegt es am Code oder Modell (Modell wäre dann bei allen Anbietern falsch)
    # TODO: bei init/insert db einzeln embeddings berechnen und speichern?
    # TODO: huggingface Variante testen