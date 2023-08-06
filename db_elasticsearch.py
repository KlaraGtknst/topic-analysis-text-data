from elasticsearch import Elasticsearch
import base64
from read_pdf import *
import gensim
from gensim.models.doc2vec import Doc2Vec, TaggedDocument


def init_db(client: Elasticsearch):
    # https://www.elastic.co/guide/en/elasticsearch/reference/current/binary.html binary for images
    print(client.indices.create(index='bahamas', mappings={
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
    }))

def insert_documents(src_path: str):
    # https://stackoverflow.com/questions/8908287/why-do-i-need-b-to-encode-a-string-with-base64
    for path in glob.glob(src_path):
        # image: https://www.codespeedy.com/convert-image-to-base64-string-in-python/
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

def search_in_db(client, model, path):
    result = client.search(index='bahamas', knn={
            "field": "embedding",
            "query_vector": infer_embedding_vector(model, path),
            "k": 10,
            "num_candidates": 100
        }, source_excludes=['image'])
    for hit in result['hits']['hits']:
        print(hit['_score'], hit['_source'])

def infer_embedding_vector(model, path):
    return model.infer_vector(tokenize(pdf_to_str(path)))



def get_tagged_input_documents(src_path):
    texts = [pdf_to_str(path) for path in glob.glob(src_path)]
    documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(texts)]
    return documents

if __name__ == '__main__':
    print('-' * 80)
    src_path = '/Users/klara/Downloads/*.pdf'

    # Create the client instance
    client = Elasticsearch("http://localhost:9200")

    # init_db(client)

    # prepare documents
    documents = get_tagged_input_documents(src_path)
    model = Doc2Vec(documents, vector_size=5, window=2, min_count=1, workers=4)
 
    # insert_documents(src_path)    

    for path in glob.glob(src_path):
        print('-' * 40, path, '-' * 40)
        search_in_db(client, model, path)
