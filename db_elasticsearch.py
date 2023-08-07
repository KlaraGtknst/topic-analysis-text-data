import collections
from elasticsearch import ConflictError, Elasticsearch
import base64
from read_pdf import *
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.utils import simple_preprocess


def init_db(client: Elasticsearch, num_dimensions: int):
    '''
    :param client: Elasticsearch client
    :param num_dimensions: number of dimensions of the embedding
    :return: None

    This function initializes the database by creating an index (i.e. the structure for an entry of type 'bahamas' database).
    The index contains the following fields:
    - embedding: a dense vector of num_dimensions dimensions. This is the vector numerical representation of the document. Its similarity is measured by cosine similarity.
    - text: the text of the document. The text is not tokenized, stemmed etc.
    - path: the path to the document on the local maschine.
    - image: the image of the document (i.e. information about the document layout). The image is encoded in base64 and has 500 dpi.

    cf. https://www.elastic.co/guide/en/elasticsearch/reference/current/binary.html for information about binary for images
    '''
    client.indices.create(index='bahamas', mappings={
        "properties": {
            "embedding": {
                "type": "dense_vector",
                "dims": num_dimensions,
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
        The text is preprocessed using the gensim function 'simple_preprocess', which returns a list of tokens, i.e. unicode strings,
        which are lowercased and tokenized. (cf. https://tedboy.github.io/nlps/generated/generated/gensim.utils.simple_preprocess.html).
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
            "embedding": model.infer_vector(simple_preprocess(pdf_to_str(path))),
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
    The document text is preprocessed in the same way as the documents stored in the database.
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
        print(hit['_score'], hit['_source']['path'].split('/')[-1])

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

def get_tagged_input_documents(src_path: str, tokens_only: bool = False):
    '''
    :param src_path: path to the documents to be inserted into the database
    :param tokens_only: if True, only the tokens of the document are returned, else tagged tokens are returned
    :return: tagged tokens or only the tokens (depending on tokens_only)

    The gensim function 'simple_preprocess' converts a document into a list of tokens (cf. https://tedboy.github.io/nlps/generated/generated/gensim.utils.simple_preprocess.html).
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



def assess_model(model: Doc2Vec, train_corpus: list):
    '''
    :param model: trained Doc2Vec model
    :param train_corpus: list of tagged documents
    :return: None

    This function assesses by:
    (1) infer a vector of a document from the training corpus
    (2) compare inferred vector with the vector of the document in the training corpus
    (3) return rank of the document based on self-similarity (i.e. the document itself should be ranked first)
    '''
    print('-'*100)
    ranks = []  # list of ranks the documents got when compared to themselves
    second_ranks = []   # list of similarities of the second most similar document
    for doc_id in range(len(train_corpus)):
        inferred_vector = model.infer_vector(train_corpus[doc_id].words)
        sims = model.dv.most_similar([inferred_vector], topn=len(model.dv)) # topn: number of most similar documents to be returned
        rank = [docid for docid, sim in sims].index(doc_id) # rank of the document via id
        ranks.append(rank)  # saves the rank of the document in terms of self-similarity
        second_ranks.append(sims[1])    # saves the similarity of the second most similar document

        print('Document ({}): «{}»\n'.format(doc_id, ' '.join(train_corpus[doc_id].words[:10])))
        print(u'SIMILAR/DISSIMILAR DOCS PER MODEL %s:\n' % model)
        for label, index in [('MOST', 0), ('SECOND-MOST', 1), ('MEDIAN', len(sims)//2), ('LEAST', len(sims) - 1)]:
            print(u'%s %s: «%s»\n' % (label, sims[index], ' '.join(train_corpus[sims[index][0]].words[:10])))
        print('-'*80)
    counter = collections.Counter(ranks)
    print(f'{counter[0]} documents are most self-similar to themselves.\n{counter[1]} documents were not ranked first.')
    

def infer_embedding_for_single_document(model: Doc2Vec, text: str):
    '''
    :param model: trained Doc2Vec model
    :param text: text of the document to be embedded
    :return: embedding vector of the document

    This function infers the embedding vector of a document.
    '''
    vector = model.infer_vector(simple_preprocess(text))
    return vector

if __name__ == '__main__':
    NUM_DIMENSIONS = 50
    print('-' * 80)
    src_path = '/Users/klara/Downloads/*.pdf'

    # Create the client instance
    client = Elasticsearch("http://localhost:9200")

    # delete old index and create new one
    if client.indices.get_mapping(index='bahamas')['bahamas']['mappings']['properties']['embedding']['dims'] != NUM_DIMENSIONS:
        client.options(ignore_status=[400,404]).indices.delete(index='bahamas')
        init_db(client, num_dimensions=NUM_DIMENSIONS)
    
    # training corpus
    train_corpus = list(get_tagged_input_documents(src_path))

    # model 
    # infrequent words are discarded, since retaining them can make model worse
    # iteration for 10s-of-thousands of documents: 10-20; more for smaller datasets
    model = Doc2Vec(train_corpus, vector_size=NUM_DIMENSIONS, window=2, min_count=2, workers=4, epochs=40)

    # build a vocabulary (i.e. list of unique words); accessable via model.wv.index_to_key
    # model.build_vocab(train_corpus)

    # additonal information about each word
    # word = 'credit'
    # print(f"Word '{word}' appeared {model.wv.get_vecattr(word, 'count')} times in the training corpus.")

    # train the model
    model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)

    # infer a vector of a new document
    #print(f'This is the numerical representation of the document: \n{infer_embedding_for_single_document(model, text="This is a new document to be searched for.")}')
    
    # assess the model
    # here: using training corpus -> overfitting, not representative
    # assess_model(model, train_corpus)
    
    try:
        insert_documents(src_path, model, client)  
    except ConflictError as err:
        print(err)

    # alternatively, use AsyncElasticsearch or time.sleep(1)
    client.indices.refresh(index="bahamas")

    # for path in glob.glob(src_path):
    #   print('\n' + '-' * 40, path, '-' * 40)
    #   search_in_db(client, model, path)