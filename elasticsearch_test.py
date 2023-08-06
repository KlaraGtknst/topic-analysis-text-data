from elasticsearch import Elasticsearch
import base64
import json

print('-' * 80)

# Create the client instance
client = Elasticsearch("http://localhost:9200")

# Successful response!
# print(client.info())

# https://www.elastic.co/guide/en/elasticsearch/reference/current/binary.html binary for images
'''
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

# https://stackoverflow.com/questions/8908287/why-do-i-need-b-to-encode-a-string-with-base64
print(str(base64.b64encode(b"base64-encoded-image")))
print(client.create(index='bahamas', id='1', document={
    "embedding": [1, 2, 3, 4, 5],
    "text": "Hello World!",
    "path": "/path/to/file",
    "image": str(base64.b64encode(b"base64-encoded-image")),
}))
'''
# print(client.get(index='bahamas', id='1'))
result = client.search(index='bahamas', knn={
    "field": "embedding",
    "query_vector": [1, 2, 3, 4, 6],
    "k": 10,
    "num_candidates": 100
})
for hit in result['hits']['hits']:
    print(hit['_score'], hit['_source'])
