from elasticsearch import Elasticsearch
import base64
from read_pdf import *
import gensim
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.utils import simple_preprocess


def init_db(client: Elasticsearch):
    '''
    :param client: Elasticsearch client
    :return: None

    This function initializes the database by creating an index (i.e. the structure for an entry of type 'bahamas' database).
    The index contains the following fields:
    - embedding: a dense vector of 5 dimensions. This is the vector numerical representation of the document. Its similarity is measured by cosine similarity.
    - text: the text of the document. The text is not tokenized, stemmed etc.
    - path: the path to the document on the local maschine.
    - image: the image of the document (i.e. information about the document layout). The image is encoded in base64 and has 500 dpi.

    cf. https://www.elastic.co/guide/en/elasticsearch/reference/current/binary.html for information about binary for images
    '''
    client.indices.create(index='bahamas', mappings={
        "properties": {
            "embedding": {
                "type": "dense_vector",
                "dims": 5,
                "index": True,
                "similarity": "cosine",
            },
            "text": {
                "type": "text",
            },
            "path": {
                "type": "keyword",
            },
            "image": {
                "type": "binary",
            },
        },
    })

def insert_documents(src_path: str, model: Doc2Vec, client: Elasticsearch):
    '''
    :param src_path: path to the documents to be inserted into the database
    :param model: Doc2Vec model
    :param client: Elasticsearch client
    :return: None

    This function inserts the documents into the database 'bahamas'. The documents are inserted as follows:
    - embedding: the embedding is inferred from the document text using the trained Doc2Vec model.
    - text: the text of the document. The text is not tokenized, stemmed etc.
    - path: the path to the document on the local maschine.
    - image: the image of the document (i.e. information about the document layout). The image is encoded in base64 and has 500 dpi.
    
    The documents receive an id which is the name of the document (i.e. the name of the pdf file without the extension).

    cf. https://stackoverflow.com/questions/8908287/why-do-i-need-b-to-encode-a-string-with-base64 for information about base64 encoding
    cf. https://www.codespeedy.com/convert-image-to-base64-string-in-python/ for information about converting images to base64
    '''
    
    for path in glob.glob(src_path):
        image_path = path.split('.')[0] + '0001-1.png'
        with open(image_path, "rb") as img_file:
            b64_image = base64.b64encode(img_file.read())

        text = pdf_to_str(path)

        id = path.split('/')[-1].split('.')[0]
        client.create(index='bahamas', id=id, document={
            "embedding": model.infer_vector(tokenize(pdf_to_str(path))),
            "text": text,
            "path": path,
            "image": str(b64_image),
        })

def search_in_db(client: Elasticsearch, model: Doc2Vec, path: str):
    '''
    :param client: Elasticsearch client
    :param model: Doc2Vec model
    :param path: path to the document to be searched for
    :return: None

    The field of interest in the database, i.e. the one to be searched for, is the embedding. 
    The embedding is inferred from the document text using the trained Doc2Vec model.
    Since the base64 encoding of the image is very long, it is excluded from the search result.
    knn is used for the search. The search returns the k=10 most similar documents.
    The parameter num_candidates is set to 100. This means that the search is performed on 100 documents (per shard, i.e. computer to perform [part of] the computation).

    The search result is printed to the console.
    The score is the similarity between the document and the query document (i.e. the document to be searched for).
    The source are the fields of document itself, which are not specifically excluded.

    cf. https://www.elastic.co/guide/en/elasticsearch/reference/current/search-search.html#search-api-knn for information about knn in elasticsearch.
    '''
    result = client.search(index='bahamas', knn={
            "field": "embedding",
            "query_vector": infer_embedding_vector(model, path),
            "k": 10,
            "num_candidates": 100
        }, source_excludes=['image'])
    for hit in result['hits']['hits']:
        print(hit['_score'], hit['_source'])

def infer_embedding_vector(model: Elasticsearch, path: str):
    '''
    :param model: trained Doc2Vec model
    :param path: path to the document to be searched for
    :return: the embedding vector of the document to be searched for

    This function infers the embedding vector of the tokenized document to be searched for.
    '''
    # TODO: tokenize ok? how is doc2vec trained? Tokenizing/lower casing?
    return model.infer_vector(tokenize(pdf_to_str(path)))

def get_tagged_input_documents(src_path: str, tokens_only: bool = False):
    '''
    :param src_path: path to the documents to be inserted into the database
    :param tokens_only: if True, only the tokens of the document are returned, else tagged tokens are returned
    :return: tagged tokens or only the tokens (depending on tokens_only)

    The gensim function converts a document into a list of tokens (cf. https://tedboy.github.io/nlps/generated/generated/gensim.utils.simple_preprocess.html).
    The resulting list of unicode strings is lowercased and tokenized.
    
    cf. https://radimrehurek.com/gensim/auto_examples/tutorials/run_doc2vec_lee.html#sphx-glr-auto-examples-tutorials-run-doc2vec-lee-py
    for the original code
    '''
    for i, path in enumerate(glob.glob(src_path)):
        tokens = simple_preprocess(pdf_to_str(path))
        if tokens_only:
            yield tokens
        else:
            yield TaggedDocument(tokens, [i])


if __name__ == '__main__':
    print('-' * 80)
    src_path = '/Users/klara/Downloads/*.pdf'

    # Create the client instance
    client = Elasticsearch("http://localhost:9200")

    # init_db(client)
    documents = list(get_tagged_input_documents(src_path))
    print(documents)
    #model = Doc2Vec(documents, vector_size=5, window=2, min_count=1, workers=4)
    # insert_documents(src_path, model, client)    

    #for path in glob.glob(src_path):
     #   print('\n' + '-' * 40, path, '-' * 40)
     #   search_in_db(client, model, path)