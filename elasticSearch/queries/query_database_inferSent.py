from elasticsearch import ConflictError, Elasticsearch
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.utils import simple_preprocess
from pyspark.mllib.linalg import Vectors
import pdb # use breakpoint() for debugging when running the code from the command line
# own modules
from text_embeddings.preprocessing.read_pdf import *
from user_interface.cli import *
from doc_images.pdf_matrix import *
from elasticSearch.queries.query_documents_tfidf import *
from text_embeddings.universal_sent_encoder_tensorFlow import *
from text_embeddings.hugging_face_sentence_transformer import *
from text_embeddings.TFIDF.preprocessing.TfidfTextPreprocessor import *
from text_embeddings.InferSent.infer_pretrained import *

'''------search in existing database-------
run this code by typing and altering the path:
    python3 query_database.py -d '/Users/klara/Documents/Uni/bachelorarbeit/data/0/*.pdf' -D '/Users/klara/Documents/Uni/bachelorarbeit/images/images/'
'''
CLIENT_ADDR = "http://localhost:9200"

def infer_embedding_vector(model: Elasticsearch, path: str):
    '''
    :param model: trained Doc2Vec model
    :param path: path to the document to be searched for
    :return: the embedding vector of the document to be searched for

    This function infers the embedding vector of the document to be searched for.
    The document is preprocessed in the same way as the documents stored in the database.
    The gensim function 'simple_preprocess' converts a document into a list of tokens (cf. https://tedboy.github.io/nlps/generated/generated/gensim.utils.simple_preprocess.html).
    The resulting list of unicode strings is lowercased and tokenized.
    '''
    return model.infer_vector(simple_preprocess(pdf_to_str(path)))

def search_inferSent_emb_in_db(client: Elasticsearch, infer_model, ae_infer_encoder, path: str):
    '''
    :param client: Elasticsearch client
    :param model: InferSent model
    :param ae_infer_encoder: trained encoder of autoencoder to reduce dimensionality of the InferSent embedding vector
    :param path: path to the document to be searched for
    :return: None

    The field of interest in the database, i.e. the one to be searched for, is the embedding. 
    The embedding is inferred from the document text using the trained InferSent model and the encoder.
    Since the base64 encoding of the image is very long, it is excluded from the search result.
    knn is used for the search. The search returns the k=10 most similar documents.
    The parameter num_candidates is set to 100. This means that the search is performed on 100 documents (per shard, i.e. computer to perform [part of] the computation).

    The search result is printed to the console.
    The score is the similarity between the document and the query document (i.e. the document to be searched for).
    The source are the fields of document itself, which are not specifically excluded.

    cf. https://www.elastic.co/guide/en/elasticsearch/reference/current/search-search.html#search-api-knn for information about knn in elasticsearch.
    '''
    doc = pdf_to_str(path)
    infer_emb = infer_model.encode([doc, doc], tokenize=True)
    embedding = ae_infer_encoder.predict(x= infer_emb)[0]

    result = client.search(index='bahamas', knn={
            "field": "inferSent_AE",
            "query_vector": embedding,
            "k": 10,
            "num_candidates": 100
        }, source_excludes=['image'])
    
    scores = {}
    for hit in result['hits']['hits']:
        scores[hit['_score']] = hit['_source']['path'].split('/')[-1]
    return scores


def main(file_paths, outpath):
     # Create the client instance
    client = Elasticsearch(CLIENT_ADDR)

    nltk.download('punkt')
    V = 1   # trained with GloVe
    MODEL_PATH = '/Users/klara/Developer/Uni/encoder/infersent%s.pkl' % V
    W2V_PATH = '/Users/klara/Developer/Uni/GloVe/glove.840B.300d.txt'

    infersent, docs = init_infer(model_path=MODEL_PATH, w2v_path=W2V_PATH, file_paths=file_paths, version=V)

    embeddings = infersent.encode(docs, tokenize=True)
    
    encoded_embedding, ae_encoder = autoencoder_emb_model(input_shape=embeddings.shape[1], latent_dim=2048, data=embeddings)

    doc_to_search = file_paths[0]
    results = search_inferSent_emb_in_db(client, infer_model=infersent, ae_infer_encoder=ae_encoder, path= doc_to_search)
    # bad results
    print('results for document: ', doc_to_search) # /Users/klara/Documents/Uni/bachelorarbeit/data/0/SAC53-21.pdf
    print('results:\n', results) #  {0.5750969: 'SAC86-17.pdf', 0.56992793: 'SAC20-15.pdf', 0.5692922: 'SAC42-1.pdf', 0.5691847: 'SAC16-72.pdf', 0.56910694: 'SAC27-25.pdf', 0.56881595: 'SAC1-11.pdf', 0.5685914: 'SAC92-8.pdf', 0.56838506: 'SAC85-13.pdf', 0.5683696: 'SAC27-22.pdf'}