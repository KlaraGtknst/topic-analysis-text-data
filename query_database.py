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
    #print(type(infer_embedding_vector(model, path)))
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

def get_docs_from_same_cluster(elastic_search_client: Elasticsearch, path_to_doc: str, n_results: int = 5) -> list:
    '''
    :param elastic_search_client: Elasticsearch client
    :param path_to_doc: path to the document to be searched for; acts as the index in the database
    :param n_results: number of results to be returned
    :return: list of paths to documents in the same cluster as the document to be searched for
    '''
    # get cluster
    doc_id = path_to_doc.split('/')[-1].split('.')[0]
    elastic_search_client.indices.refresh(index='bahamas')
    resp = elastic_search_client.get(index='bahamas', id=doc_id,  source_includes=['pca_kmeans_cluster'])
    cluster = resp['_source']['pca_kmeans_cluster'][0]

    # query
    query = {   
        "query_string": {
            "fields" : ["pca_kmeans_cluster"],
            "query": str(cluster),
        }
    }

    # results
    search_results = elastic_search_client.search(index='bahamas', query=query, source_includes=['path'], size=n_results)

    # Extract and process the search results
    print('Query Document: ', doc_id, ' of cluster ', cluster, '\n')
    for hit in search_results['hits']['hits']:
        # Access the relevant fields from the hit
        source = hit['_source']
        document_id = hit['_id']
        score = hit['_score']
        
        # Process the data as needed
        print(f"Document ID: {document_id}, Score: {score}")
        print("Source Data:", source)
        print("-" * 20)

    return search_results


if __name__ == '__main__':
    args = arguments()
    src_paths = get_input_filepath(args)
    image_src_path = get_filepath(args, option='image')
    
    # Create the client instance
    client = Elasticsearch("http://localhost:9200")

    print('-' * 40, 'Query for same cluster in database', '-' * 40)
    NUM_RESULTS = 5

    get_docs_from_same_cluster(elastic_search_client = client, path_to_doc = src_paths[13], n_results=NUM_RESULTS)