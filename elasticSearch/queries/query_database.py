from elasticsearch import ConflictError, Elasticsearch
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.utils import simple_preprocess
from pyspark.mllib.linalg import Vectors
from gensim.test.utils import get_tmpfile
from tkinter import *
from tensorflow.python.keras.models import model_from_json
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
from text_embeddings import save_models


SRC_INCLUDES = ['path', 'text']

'''------search in existing database-------
run this code by typing and altering the path:
    python3 query_database.py -d '/Users/klara/Documents/Uni/bachelorarbeit/data/0/*.pdf' -D '/Users/klara/Documents/Uni/bachelorarbeit/images/images/'
'''

def infer_doc2vec_embedding(model: Doc2Vec, path: str):
    '''
    :param model: trained Doc2Vec model
    :param path: path to the document to be searched for
    :return: the doc2vec embedding vector of the document to be searched for

    This function infers the embedding vector of the document to be searched for.
    The document is preprocessed in the same way as the documents stored in the database.
    The gensim function 'simple_preprocess' converts a document into a list of tokens (cf. https://tedboy.github.io/nlps/generated/generated/gensim.utils.simple_preprocess.html).
    The resulting list of unicode strings is lowercased and tokenized.
    '''
    return model.infer_vector(simple_preprocess(pdf_to_str(path)))

def search_sim_doc2vec_docs_in_db(client: Elasticsearch, path: str, src_paths='/Users/klara/Documents/Uni/bachelorarbeit/data/0/*.pdf', doc2vec_model: Doc2Vec=None):
    '''
    :param client: Elasticsearch client
    :param doc2vec_model: Doc2Vec model
    :param path: path to the document to be searched for
    :return: None

    The field of interest in the database, i.e. the one to be searched for, is the doc2vec embedding. 
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
    if doc2vec_model is None:
        doc2vec_model = save_models.get_model('doc2vec', src_paths)
    return get_db_search_results(client, infer_doc2vec_embedding(doc2vec_model, path), 'doc2vec')


def find_document_tfidf(client: Elasticsearch, model: TfidfVectorizer, path: str):
    '''
    :param client: Elasticsearch client
    :param model: TfidfVectorizer model
    :param path: path to the document to be searched for
    :return: None

    The field of interest in the database, i.e. the one to be searched for, is the tfidf embedding. 
    The embedding is inferred from the document text using the fitted tfidf model.
    Since there is a extra flag in the tfidf embedding in the database, which indicates whether the representation of the document is an all-zero vector, 
    this flag is also added to the query vector.
    The document text is preprocessed in the same way as the documents stored in the database.
    Since the base64 encoding of the image is very long, it is excluded from the search result.
    knn is used for the search. The search returns the k=10 most similar documents.
    The parameter num_candidates is set to 100. This means that the search is performed on 100 documents (per shard, i.e. computer to perform [part of] the computation).

    The search result is printed to the console.
    The score is the similarity between the document and the query document (i.e. the document to be searched for).
    The source are the fields of document itself, which are not specifically excluded.

    cf. https://www.elastic.co/guide/en/elasticsearch/reference/current/search-search.html#search-api-knn for information about knn in elasticsearch.
    '''
    embedding = model.transform([pdf_to_str(path)])
    embedding = np.ravel(embedding.todense())
    embedding = np.append(embedding, 1 if np.array([entry  == 0 for entry in embedding]).all() else 0)
   
    return get_db_search_results(client, embedding, 'sim_docs_tfidf')

def text_search_db(elastic_search_client: Elasticsearch, text:str, page:int=0, count:int=10):
    '''
    :param elastic_search_client: Elasticsearch client
    :param text: text to be searched for
    :param page: page of the results to be returned
    :param count: number of results to be returned
    :return: dictionary of paths and scores of best fitting 10 documents in database
    '''
    results = elastic_search_client.search(
        index='bahamas', 
        size=count,
        from_=(page*count),
        query= {
                'match' : {
                    'text': text
                }
            },
        source_includes=SRC_INCLUDES)['hits']['hits']
    return convert_hits(results)

def convert_hits(results):
    return [{'_id': result['_id'], '_score': result['_score'], **result['_source']} for result in results]

def get_all_docs_in_db(elastic_search_client: Elasticsearch) -> dict:
    '''
    :param elastic_search_client: Elasticsearch client
    :return: dictionary of paths and scores of all documents in database
    '''
    results = elastic_search_client.search(
        index='bahamas', 
        body={
            'size': get_number_docs_in_db(elastic_search_client),
            'query': {
                'match_all' : {}
                }
            },   
        source_includes=SRC_INCLUDES)['hits']['hits']
    return convert_hits(results)


def get_knn_res(doc_to_search_for:str, query_type:str, elastic_search_client:Elasticsearch, n_results:int):
    '''
    :param doc_to_search_for: id of the document to be searched for; acts as the index in the database
    :param query_type: type of the query to be searched for; must be one of:
        "doc2vec", "sim_docs_tfidf", "google_univ_sent_encoding", "huggingface_sent_transformer", "inferSent_AE", "pca_kmeans_cluster"
    :param elastic_search_client: Elasticsearch client
    :param n_results: number of results to be returned
    :return: dictionary of paths, scores, ids and scores of best fitting 10 documents in database
    '''
    # get embediding/ search query data
    elastic_search_client.indices.refresh(index='bahamas')
    if query_type != 'pca_kmeans_cluster':
        try:
            resp = elastic_search_client.get(index='bahamas', id=doc_to_search_for,  source_includes=[query_type])
            embedding = resp['_source'][query_type]
        except:
            # TODO: create embedding even though everything is offline?
            return {'error': 'document not found in database'}

        # get similar documents
        results = elastic_search_client.search(index='bahamas', knn={
                "field": query_type,
                "query_vector": embedding,
                "k": n_results,
                "num_candidates": 100
            }, source_includes=SRC_INCLUDES)

        return convert_hits(results['hits']['hits'])
    else:
        return get_docs_from_same_cluster(elastic_search_client, doc_to_search_for, n_results)


def get_doc_meta_data(elastic_search_client: Elasticsearch, doc_id: str):
    '''
    :param elastic_search_client: Elasticsearch client
    :param doc_id: document id to be searched for; acts as the index in the database
    :return: path and text of the document
    '''
    elastic_search_client.indices.refresh(index='bahamas')
    resp = elastic_search_client.get(index='bahamas', id=doc_id,  source_includes=SRC_INCLUDES).body
    return {'_id': resp['_id'], **resp['_source']}

def get_docs_in_db(elastic_search_client: Elasticsearch, start:int=0, n_docs:int=10) -> dict:
    '''
    :param elastic_search_client: Elasticsearch client
    :param start: start index of the documents to be returned
    :param n_docs: number of documents to be returned
    :return: dictionary of paths and texts to documents in database
    '''
    results = elastic_search_client.search(
        index='bahamas', 
        from_=start*n_docs,
        size=n_docs,
        query= {
                'match_all' : {}
                },  
        source_includes=SRC_INCLUDES)['hits']['hits']
    return convert_hits(results)

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
    resp = elastic_search_client.search(index='bahamas', query=query, source_includes=SRC_INCLUDES, size=n_results)['hits']['hits']
    return convert_hits(resp)


def get_number_docs_in_db(client: Elasticsearch) -> int:
    '''
    :param client: Elasticsearch client
    :return: number of documents in database
    
    for more information about the number of documents in database see: 
        https://stackoverflow.com/questions/49691574/counting-number-of-documents-in-an-index-in-elasticsearch
    '''
    client.indices.refresh(index='bahamas')
    resp = client.cat.count(index='bahamas', params={"format": "json"})
    return resp[0]['count']


def get_sim_docs_tfidf(doc_to_search_for, src_paths='/Users/klara/Documents/Uni/bachelorarbeit/data/0/*.pdf', client_addr="http://localhost:9200"):
    '''
    :param doc_to_search_for: path to the document to be searched for
    :param src_paths: path to the document corpus to be searched in
    :param client_addr: address of the Elasticsearch client
    :return: list of paths to documents in the same cluster as the document to be searched for
    '''
    client = Elasticsearch(client_addr)
    sim_docs_tfidf = save_models.get_model('tfidf', src_paths)
    
    return find_document_tfidf(client, sim_docs_tfidf, path=doc_to_search_for)
        

def find_sim_docs_google_univSentEnc(path: str, client: Elasticsearch=None):
    '''
    :param client: Elasticsearch client
    :param path: path to document to be searched for
    :return list of ten most similar scores and documents in database in terms of google universal sentence encoding
    '''    
    if client is None:
        client = Elasticsearch("http://localhost:9200")
    google_model = google_univ_sent_encoding_aux()
    embedding = embed([pdf_to_str(path)], google_model).numpy().flatten().tolist()
    return get_db_search_results(client, embedding, 'google_univ_sent_encoding')

def find_sim_docs_hugging_face_sentTrans(path: str, client: Elasticsearch=None):
    '''
    :param client: Elasticsearch client
    :param path: path to document to be searched for
    :return list of ten most similar scores and documents in database in terms of huggingface sentence transformers embedding
    '''
    if client is None:
        client = Elasticsearch("http://localhost:9200")
    huggingface_model = init_hf_sentTrans_model()
    embedding = huggingface_model.encode(pdf_to_str(path))
    return get_db_search_results(client, embedding, 'huggingface_sent_transformer')


def find_sim_docs_inferSent(src_paths:list, path: str, client: Elasticsearch=None):
    if client is None:
        client = Elasticsearch("http://localhost:9200")
    infer_model_name = 'infersent_model'
    ae_model_name = 'ae_model'
    text = pdf_to_str(path)
    # InferSent
    inferSent_model = save_models.get_model(infer_model_name, src_paths)
    

    # AE
    ae_infer_encoder = save_models.get_model(ae_model_name, src_paths)

    inferSent_embedding = inferSent_model.encode([text, text], tokenize=True)
    compressed_infersent_embedding = ae_infer_encoder.predict(x=inferSent_embedding)[0]

    return get_db_search_results(client, compressed_infersent_embedding, 'inferSent_AE')


def get_db_search_results(client: Elasticsearch, embedding: np.array, field: str, num_res:int=10):
    results = client.search(index='bahamas', knn={
            "field": field,
            "query_vector": embedding,
            "k": num_res,
            "num_candidates": 100
        }, source_includes=SRC_INCLUDES)
    
    return results


def main(src_paths, image_src_path):
    
    # Create the client instance
    client = Elasticsearch("http://localhost:9200")

    # number of documents in database
    print('number of documents in database: ', get_number_docs_in_db(client))

    # create json object to save results to disk and use them later for topic modelling
    results = {}

    # Cluster query
    doc_to_search_for = src_paths[0]
    print('-' * 40, f'Query for same cluster as {doc_to_search_for} in database', '-' * 40)
    NUM_RESULTS = 5
    cluster_results = get_docs_from_same_cluster(elastic_search_client = client, path_to_doc = doc_to_search_for, n_results=NUM_RESULTS)
    print('Cluster results: ',  [hit['_source']['path'] for hit in cluster_results['hits']['hits']])
    results['cluster'] = {doc_to_search_for: [hit['_source']['path'] for hit in cluster_results['hits']['hits']]}


    # query database for a document using tfidf
    docs = get_docs_from_file_paths(src_paths)
    sim_docs_tfidf = TfidfVectorizer(input='content', preprocessor=TfidfTextPreprocessor().fit_transform, min_df=3, max_df=int(len(docs)*0.07))
    sim_docs_document_term_matrix = sim_docs_tfidf.fit(docs)
    tfidf_results = find_document_tfidf(client, sim_docs_tfidf, path=doc_to_search_for)
    #image_paths = [image_src_path + file_name.split('.')[0] + '.png' for file_name in list(tfidf_results.values())]

    #create_image_matrix(image_paths, 2)
    print('-' * 40, f'TFIDF results for {doc_to_search_for} in database', '-' * 40)
    print(doc_to_search_for.split('/')[:-1])
    print()
    results['tfidf'] = {doc_to_search_for: ['/'.join(doc_to_search_for.split('/')[:-1]) + '/' + doc for doc in tfidf_results.values()]}
    
    # universal sentence encoder
    univSentEncRes = find_sim_docs_google_univSentEnc(doc_to_search_for)
    results['universal_sent_encoder'] = {doc_to_search_for: list(univSentEncRes.values())}

    # hugging face sentence transformer 
    hugFaceSentTransRes = find_sim_docs_hugging_face_sentTrans(path=doc_to_search_for)
    results['hugging_face_sentence_transformer'] = {doc_to_search_for: list(hugFaceSentTransRes.values())}
    
    # inferSent
    inferSentRes = find_sim_docs_inferSent(src_paths=src_paths, path=doc_to_search_for, client=client)
    results['inferSent'] = {doc_to_search_for: list(inferSentRes.values())}

    print(results)
    with open("results/results.json", "w") as outfile:
        json.dump(results, outfile)
