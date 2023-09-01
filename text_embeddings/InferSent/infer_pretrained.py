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
    infersent.load_state_dict(torch.load(model_path))
    infersent.set_w2v_path(w2v_path)
    docs = get_docs_from_file_paths(file_paths)
    infersent.build_vocab(docs, tokenize=True)

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
    # Encoder
    x = tensorflow.keras.layers.Input(shape=(input_shape), name="encoder_input")

    encoder_dense_layer1 = tensorflow.keras.layers.Dense(units=300, name="encoder_dense_1")(x)
    encoder_activ_layer1 = tensorflow.keras.layers.LeakyReLU(name="encoder_leakyrelu_1")(encoder_dense_layer1)

    encoder_dense_layer2 = tensorflow.keras.layers.Dense(units=latent_dim, name="encoder_dense_2")(encoder_activ_layer1)
    encoder_output = tensorflow.keras.layers.LeakyReLU(name="encoder_output")(encoder_dense_layer2)

    encoder = tensorflow.keras.models.Model(x, encoder_output, name="encoder_model")
    # encoder.summary()

    # Decoder
    decoder_input = tensorflow.keras.layers.Input(shape=(latent_dim), name="decoder_input")

    decoder_dense_layer1 = tensorflow.keras.layers.Dense(units=300, name="decoder_dense_1")(decoder_input)
    decoder_activ_layer1 = tensorflow.keras.layers.LeakyReLU(name="decoder_leakyrelu_1")(decoder_dense_layer1)

    decoder_dense_layer2 = tensorflow.keras.layers.Dense(units=input_shape, name="decoder_dense_2")(decoder_activ_layer1)
    decoder_output = tensorflow.keras.layers.LeakyReLU(name="decoder_output")(decoder_dense_layer2)

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
    
    return encoded_images, encoder


def main(file_paths, outpath):

    nltk.download('punkt')
    V = 1   # trained with GloVe
    MODEL_PATH = '/Users/klara/Developer/Uni/encoder/infersent%s.pkl' % V
    W2V_PATH = '/Users/klara/Developer/Uni/GloVe/glove.840B.300d.txt'

    infersent, docs = init_infer(model_path=MODEL_PATH, w2v_path=W2V_PATH, file_paths=file_paths, version=V)
    
    embeddings = infersent.encode(docs, tokenize=True)
    doc = docs[0]
    #print([doc], [doc])
    # embdding does not work on singular input
    embedding = infersent.encode([doc, doc], tokenize=True)
    print('shape of embeddings: ', embeddings[0].shape, embedding[0].shape)
    # the difference is non zero!
    print('difference of embeddings: ', sum((embedding[0] - embeddings[0])**2))
    
    # infersent.visualize('A man plays an instrument.', tokenize=True)

    # use AE to reduce dimensionality
    # TODO: split into train and test
    # TODO: normalize data?
    encoded_embedding, ae_encoder = autoencoder_emb_model(input_shape=embeddings.shape[1], latent_dim=2048, data=embeddings)
    # Encoder does not work on singular input
    test = np.array([embeddings[0], embeddings[0]])
    embedding = ae_encoder.predict(x= test)[0]
    print('difference of embeddings: ', sum((embedding - encoded_embedding[0])**2))