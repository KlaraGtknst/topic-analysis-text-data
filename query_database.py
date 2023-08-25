from elasticsearch import ConflictError, Elasticsearch
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.utils import simple_preprocess
from pyspark.mllib.linalg import Vectors
# own modules
from read_pdf import *
from cli import *
from pdf_matrix import *
from query_documents_tfidf import *
from universal_sent_encoder_tensorFlow import *
from hugging_face_sentence_transformer import *

'''------search in existing database-------
run this code by typing and altering the path:
    python3 query_database.py -d '/Users/klara/Documents/Uni/bachelorarbeit/data/0/*.pdf' -D '/Users/klara/Documents/Uni/bachelorarbeit/images/images/'
'''

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

def search_in_db(client: Elasticsearch, model: Doc2Vec, path: str):
    '''
    :param client: Elasticsearch client
    :param model: Doc2Vec model
    :param path: path to the document to be searched for
    :return: None

    The field of interest in the database, i.e. the one to be searched for, is the embedding. 
    The embedding is inferred from the document text using the trained Doc2Vec model.
    The document text is preprocessed in the same way as the documents stored in the database.
    Since the base64 encoding of the image is very long, it is excluded from the search result.
    knn is used for the search. The search returns the k=10 most similar documents.
    The parameter num_candidates is set to 100. This means that the search is performed on 100 documents (per shard, i.e. computer to perform [part of] the computation).

    The search result is printed to the console.
    The score is the similarity between the document and the query document (i.e. the document to be searched for).
    The source are the fields of document itself, which are not specifically excluded.

    cf. https://www.elastic.co/guide/en/elasticsearch/reference/current/search-search.html#search-api-knn for information about knn in elasticsearch.
    '''
    print(type(infer_embedding_vector(model, path)))
    result = client.search(index='bahamas', knn={
            "field": "embedding",
            "query_vector": infer_embedding_vector(model, path),
            "k": 10,
            "num_candidates": 100
        }, source_excludes=['image'])
    
    scores = {}
    for hit in result['hits']['hits']:
        scores[hit['_score']] = hit['_source']['path'].split('/')[-1]
    return scores


def find_document_tfidf(client: Elasticsearch, model: TfidfVectorizer, path: str):
    '''
    :param client: Elasticsearch client
    :param model: TfidfVectorizer model
    :param path: path to the document to be searched for
    :return: None

    The field of interest in the database, i.e. the one to be searched for, is the tfidf embedding. 
    The embedding is inferred from the document text using the fitted tfidf model.
    The document text is preprocessed in the same way as the documents stored in the database.
    Since the base64 encoding of the image is very long, it is excluded from the search result.
    knn is used for the search. The search returns the k=10 most similar documents.
    The parameter num_candidates is set to 100. This means that the search is performed on 100 documents (per shard, i.e. computer to perform [part of] the computation).

    The search result is printed to the console.
    The score is the similarity between the document and the query document (i.e. the document to be searched for).
    The source are the fields of document itself, which are not specifically excluded.

    cf. https://www.elastic.co/guide/en/elasticsearch/reference/current/search-search.html#search-api-knn for information about knn in elasticsearch.
    '''
    print(type(Vectors.dense(np.array(model.transform([pdf_to_str(path)]).tocoo().data))))
    # FIXME: TypeError(f"Unable to serialize {data!r} (type: {type(data)})") TypeError: Unable to serialize DenseVector([...]) (type: <class 'pyspark.ml.linalg.DenseVector'>)
    result = client.search(index='bahamas', knn={
            "field": "find_doc_tfidf",
            "query_vector": Vectors.dense(np.array(model.transform([pdf_to_str(path)]).tocoo().data)),
            "k": 10,
            "num_candidates": 100
        }, source_excludes=['image'])
    
    scores = {}
    for hit in result['hits']['hits']:
        scores[hit['_score']] = hit['_source']['path'].split('/')[-1]
    return scores

if __name__ == '__main__':
    args = arguments()
    src_paths = get_input_filepath(args)
    image_src_path = get_filepath(args, option='image')
    
    # Create the client instance
    client = Elasticsearch("http://localhost:9200")

    print('-' * 40, 'hello', '-' * 40)