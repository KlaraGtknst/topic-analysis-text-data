import statistics

import pandas as pd
from text_embeddings.preprocessing.read_pdf import *
from user_interface.cli import *
from elasticSearch.queries.query_documents_tfidf import get_docs_from_file_paths
import nltk
from text_embeddings.InferSent.models import InferSent
import torch
import tensorflow.python.keras.layers
import tensorflow.python.keras.models
import tensorflow.python.keras.optimizers
import numpy as np



'''------Code to encode documents as sentence embeddings using pretrained models (InferSent)-------
Since the model is pretrained and the output embedding is a vector of size 4096, the model has to be trained again to fit the database's maximum dense vector size.

The code below is based on:
    https://github.com/facebookresearch/InferSent
    https://github.com/facebookresearch/InferSent/blob/main/models.py
    https://www.kaggle.com/code/jacksoncrow/infersent-demo
    https://morioh.com/a/95a832e85c0d/infersent-sentence-embeddings

run this code by typing and altering the path:
    python3 infer_pretrained.py -d '/Users/klara/Downloads/*.pdf' -o '/Users/klara/Downloads/'
    python3 infer_pretrained.py -d '/Users/klara/Documents/Uni/bachelorarbeit/data/0/*.pdf' -o '/Users/klara/Downloads/'
'''

def init_infer(model_path: str, w2v_path: str, file_paths: list, version: int = 1) -> tuple:
    params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048, # value bigger than maximum database vector size, change non-trivial since pre-trained model  has to be trained again
                    'pool_type': 'max', 'dpout_model': 0.0, 'version': version}
    infersent = InferSent(params_model)
    infersent.load_state_dict(torch.load(model_path))   # params of model in state dict
    infersent.set_w2v_path(w2v_path)
    docs = get_docs_from_file_paths(file_paths)
    infersent.build_vocab(docs, tokenize=True)  # keep only those word vectors in vocab needed

    return infersent, docs

# RMSE
def rmse(y_true, y_predict):
    '''
    :param y_true: true values
    :param y_predict: predicted values
    :return: root mean squared error
    
    for more information see:
    https://blog.paperspace.com/autoencoder-image-compression-keras/
    '''
    return tensorflow.keras.backend.mean(tensorflow.keras.backend.square(y_true-y_predict))
    
def autoencoder_emb_model(input_shape : int, data : list, latent_dim : int = 2048):
    '''
    :param input_shape: dimension of the input data (1d array)
    :param data: data to be encoded
    :param latent_dim: dimension of the latent space; dimension of the output/ compressed data
    :return: encoded data and trained encoder of autoencoder

    for more information see:
    https://blog.paperspace.com/autoencoder-image-compression-keras/
    '''
    dim1 = 3500
    dim2 = 3000
    dim3 = 2500
    # Encoder
    x = tensorflow.keras.layers.Input(shape=(input_shape), name="encoder_input")

    encoder_dense_layer1 = tensorflow.keras.layers.Dense(units=dim1, name="encoder_dense_1")(x)
    encoder_activ_layer1 = tensorflow.keras.layers.LeakyReLU(name="encoder_leakyrelu_1")(encoder_dense_layer1)

    encoder_dense_layer2 = tensorflow.keras.layers.Dense(units=dim2, name="encoder_dense_2")(encoder_activ_layer1)
    encoder_activ_layer2 = tensorflow.keras.layers.LeakyReLU(name="encoder_leakyrelu_2")(encoder_dense_layer2)

    encoder_dense_layer3 = tensorflow.keras.layers.Dense(units=dim3, name="encoder_dense_3")(encoder_activ_layer2)
    encoder_activ_layer3 = tensorflow.keras.layers.LeakyReLU(name="encoder_leakyrelu_3")(encoder_dense_layer3)

    encoder_dense_layer4 = tensorflow.keras.layers.Dense(units=latent_dim, name="encoder_dense_4")(encoder_activ_layer3)
    encoder_output = tensorflow.keras.layers.LeakyReLU(name="encoder_output")(encoder_dense_layer4)

    encoder = tensorflow.keras.models.Model(x, encoder_output, name="encoder_model")
    # encoder.summary()

    # Decoder
    decoder_input = tensorflow.keras.layers.Input(shape=(latent_dim), name="decoder_input")

    decoder_dense_layer1 = tensorflow.keras.layers.Dense(units=dim3, name="decoder_dense_1")(decoder_input)
    decoder_activ_layer1 = tensorflow.keras.layers.LeakyReLU(name="decoder_leakyrelu_1")(decoder_dense_layer1)

    decoder_dense_layer2 = tensorflow.keras.layers.Dense(units=dim2, name="decoder_dense_2")(decoder_activ_layer1)
    decoder_activ_layer2 = tensorflow.keras.layers.LeakyReLU(name="decoder_leakyrelu_2")(decoder_dense_layer2)

    decoder_dense_layer3 = tensorflow.keras.layers.Dense(units=dim1, name="decoder_dense_3")(decoder_activ_layer2)
    decoder_activ_layer3 = tensorflow.keras.layers.LeakyReLU(name="decoder_leakyrelu_3")(decoder_dense_layer3)

    decoder_dense_layer4 = tensorflow.keras.layers.Dense(units=input_shape, name="decoder_dense_4")(decoder_activ_layer3)
    decoder_output = tensorflow.keras.layers.LeakyReLU(name="decoder_output")(decoder_dense_layer4)

    decoder = tensorflow.keras.models.Model(decoder_input, decoder_output, name="decoder_model")
    # decoder.summary()

    # Autoencoder
    ae_input = tensorflow.keras.layers.Input(shape=(input_shape), name="AE_input")
    ae_encoder_output = encoder(ae_input)
    ae_decoder_output = decoder(ae_encoder_output)

    ae = tensorflow.keras.models.Model(ae_input, ae_decoder_output, name="AE")
    # ae.summary()

    # AE Compilation
    ae.compile(loss="mse", optimizer=tensorflow.keras.optimizers.legacy.Adam(learning_rate=0.0005))

    # data set, TODO: split into train and test
    x_train = data
    x_test = data

    # Training AE
    ae.fit(x_train, x_train, epochs=20, batch_size=256, shuffle=True, validation_data=(x_test, x_test))

    encoded_images = encoder.predict(x_train)
    
    return encoded_images, encoder, decoder

def create_ae_score_plot():
    scores = pd.read_json('results/score_per_ae_config.json')
    scores = pd.DataFrame(scores)
    
    x = np.arange(len(scores.index))  # the label locations
    width = 0.25  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(layout='constrained')

    for cat in ['rsme', 'cosine_similarity']:
        offset = width * multiplier
        colours = ['green' if i == (np.argmin(scores[cat]) if cat == 'rsme' else np.argmax(scores[cat])) else ('blue' if cat == 'rsme' else 'orange') for i in scores.index ]
        rects = ax.bar(x + offset, scores[cat], width, label=cat, color=colours)
        ax.bar_label(rects, padding=3)
        multiplier += 1


    # Add some text for labels, title and custom x-axis tick labels, etc.
    labels = [str(sorted(score)) for score in scores['layers']]
    ax.set_ylabel('Quality of reconstruction')
    ax.set_xlabel('Network architecture (dimensions)')
    ax.set_title('Architecture comparison of autoencoders')
    ax.set_xticks(x + width, labels, rotation=45)
    ax.legend(loc='upper left', ncols=3)
    ax.set_ylim(0, max(max(scores['rsme']), max(scores['cosine_similarity'])) + 0.7)
    plt.savefig('results/ae_score_plot.pdf', format='pdf')
    plt.show()


def main(file_paths, outpath):

    # nltk.download('punkt')
    # V = 1   # trained with GloVe
    # MODEL_PATH = '/Users/klara/Developer/Uni/encoder/infersent%s.pkl' % V
    # W2V_PATH = '/Users/klara/Developer/Uni/bahamas_word2vec/bahamas_w2v.txt'
    # #'/Users/klara/Developer/Uni/GloVe/glove.840B.300d.txt'

    # # infersent
    # infersent, docs = init_infer(model_path=MODEL_PATH, w2v_path=W2V_PATH, file_paths=file_paths, version=V)
    
    # embeddings = infersent.encode(docs, tokenize=True)
    # doc = docs[0]
    # # embdding does not work on singular input
    # embedding = infersent.encode([doc], tokenize=True)
    # print(embedding)
    # # the difference is non zero!
    # #print('difference of embeddings: ', sum((embedding[0] - embeddings[0])**2))
    
    # # infersent.visualize('A man plays an instrument.', tokenize=True)

    # # AE
    # encoded_embedding, ae_encoder, ae_decoder = autoencoder_emb_model(input_shape=embeddings.shape[1], latent_dim=2048, data=embeddings)
    # # Encoder does not work on singular input
    # #test = np.array([embeddings[0], embeddings[0]])
    # #embedding = ae_encoder.predict(x= test)[0]
    # #print('difference of embeddings: ', sum((embedding - encoded_embedding[0])**2))

    # # reconstruction error
    # inverse_embedding = ae_decoder.predict(x= encoded_embedding)

    # # RMSE
    # rsme = np.linalg.norm(inverse_embedding - embeddings) / np.sqrt(embeddings.shape[0])
    # print('RMSE: ', rsme)
    # # cosine similarity
    # cos_sim = statistics.mean([np.dot(inverse_emb,embedding)/(np.linalg.norm(inverse_emb)*np.linalg.norm(embedding)) for inverse_emb, embedding in zip(inverse_embedding, embeddings)])
    # print('cosine similarity: ', cos_sim)

    # # save results
    # dim1 = 3500
    # dim2 = 3000
    # dim3 = 2500
    # dims_used = [embeddings.shape[1], dim1, dim2, dim3, 2048]
    # scores = pd.read_json('results/score_per_ae_config.json') if os.path.exists('results/score_per_ae_config.json') else pd.DataFrame(columns=['layers', 'rsme', 'cosine_similarity'])
    # scores = pd.concat([scores, pd.DataFrame({'layers': [dims_used], 'rsme': [rsme], 'cosine_similarity': [cos_sim]})])
    # scores.reset_index(inplace=True)
    # scores.to_json('results/score_per_ae_config.json')

    create_ae_score_plot()

    # layer 1/dim1, 2/dim2, 3/dim3 & 4/latent dim AE encoder:
    ## RMSE:  2.4383033437850337
    ## cosine similarity:  0.9231057

    # layer 1/dim1, 2/dim2 & 4/latent dim AE encoder:
    ## RMSE:  1.9848899488217058
    ## cosine similarity:  0.93076587

    # layer 1/dim1 & 4/latent dim AE encoder:   * best cosine similarity *
    # RMSE:  1.9654447244549933
    # cosine similarity:  0.9327269

    # layer 2/dim2 & 4/latent dim AE encoder: 
    # RMSE:  1.8729751456614294                 * best RMSE *
    # cosine similarity: 0.9323499

    # layer 3/dim3 & 4/latent dim AE encoder:   
    # RMSE:  1.9394561118826126
    # cosine similarity: 0.9317809

    # 4/latent dim AE encoder:                  
    ## RMSE:  1.9499668372448338
    ## cosine similarity:  0.9272389
