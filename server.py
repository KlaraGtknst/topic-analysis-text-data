import os
from elasticsearch import Elasticsearch
from flask import Flask, make_response, request, send_file, send_from_directory
from flask_restx import Api, Resource # https://stackoverflow.com/questions/60156202/flask-app-wont-launch-importerror-cannot-import-name-cached-property-from-w
from text_visualizations import visualize_texts
from elasticSearch.queries import query_database
from flask_cors import CORS
# flask --app server run --debug --port 8000

CLIENT_ADDR = "http://localhost:9200"
app = Flask(__name__)
api = Api(app, version='1.0', title='Topic Analysis of large unstructured document data',
    description='API of the project.')
cors = CORS(app)

search_doc = {'count': {'description':'Number of documents per page', 'type':'int','default':10}, 
                 'knn_type': {'description':'Type of knn search', 
                              'enum':[ "doc2vec","sim_docs_tfidf","google_univ_sent_encoding","huggingface_sent_transformer","inferSent_AE","pca_kmeans_cluster"]}}
knn_source = {'knn_source': 'Document to search for'}
page_doc = {'page': {'description':'Page number', 'type':'int','default':0}}
text_doc = {'text': {'description':'Text to search for', 'type':'string'}}
@api.doc(params=page_doc | text_doc | knn_source | search_doc)
                 
@api.route('/documents', endpoint='documents')
class Documents(Resource):
    # return all documents
    def get(self):
        # http://127.0.0.1:8000/documents?page=0&count=10
        # query parameters
        args = request.args
        page = args.get('page', default=0, type=int)
        count = args.get('count', default=10, type=int)
        text = args.get('text', default=None, type=str)
        knn_source = args.get('knn_source', default=None, type=str) # TODO maybe change type to int depending on final _id
        knn_type = args.get('knn_type', default=None, type=str)

        # client
        elastic_search_client = Elasticsearch(CLIENT_ADDR)

        if text:
            # text search
            # http://127.0.0.1:8000/documents?text=bahamas
            result = query_database.text_search_db(elastic_search_client, text=text, page=page, count=count)

        elif knn_type and knn_source:
            # http://127.0.0.1:8000/documents?knn_source=SAC1-6&knn_type=sim_docs_tfidf
            result = query_database.get_knn_res(doc_to_search_for=knn_source, query_type=knn_type, elastic_search_client=elastic_search_client, n_results=count)

        else:
            # regular list
            result = query_database.get_docs_in_db(elastic_search_client, start=page, n_docs=count)

        return result

id_doc = {'id': {'description':'identifier of document', 'type':'string','required':'true'}}
@api.doc(params=id_doc)
@api.route('/documents/<id>', endpoint='document')
class Document(Resource):
    # return one document as JSON
    def get(self, id):
        # http://127.0.0.1:8000/documents/SAC1-6
        elastic_search_client = Elasticsearch(CLIENT_ADDR)
        return query_database.get_doc_meta_data(elastic_search_client, doc_id=id)
        
@api.doc(params=id_doc)
@api.route('/documents/<id>/pdf', endpoint='pdf')
class PDF(Resource):
    # return one document as PDF
    def get(self, id):
        # http://127.0.0.1:8000/documents/SAC1-6.pdf
        elastic_search_client = Elasticsearch(CLIENT_ADDR)
        resp_path = query_database.get_doc_meta_data(elastic_search_client, doc_id=id)['path']
        print('*'*50)
        print(resp_path)    # FIXME: path is not relative under etc. current dir, cannot be displayed
        return send_file(resp_path)

@api.doc(params=search_doc)
@api.route('/documents/<id>/wordcloud', endpoint='wordcloud')
class WordCloud(Resource):
    # return wordcloud of one document as PNG
    def get(self, id):
        # http://127.0.0.1:8000/documents/SAC1-6/wordcloud

        elastic_search_client = Elasticsearch(CLIENT_ADDR)

        # query parameters
        args = request.args
        count = args.get('count', default=10, type=int)
        knn_type = args.get('knn_type', default=None, type=str)

        if knn_type: # multiple documents as input
            # http://127.0.0.1:8000/documents/SAC1-6/wordcloud?knn_source=SAC1-6&knn_type=sim_docs_tfidf
            sim_docs = query_database.get_knn_res(doc_to_search_for=id, query_type=knn_type, elastic_search_client=elastic_search_client, n_results=count)
            texts = [doc['text'] for doc in sim_docs]

        else:   # one document as input
            # get text from document
            texts = [query_database.get_doc_meta_data(elastic_search_client, id)['text']]
        img = visualize_texts.get_one_visualization_from_text(option='wordcloud', texts=texts)
        bytes = visualize_texts.image_to_byte_array(img)
        response = make_response(bytes)
        response.headers.set('Content-Type', 'image/png')
        return response

@api.doc(params=id_doc)
@api.route('/documents/<id>/term_frequency', endpoint='term_frequency')
class TermFrequency(Resource):
    # return term frequency of one document as PNG
    def get(self, id):
        # TODO: matplotlib nicht umgehbar, geht vielleicht nicht
        print(id) # http://127.0.0.1:8000/documents/SAC1-6/term_frequency
        
        if not os.path.exists('visualizations'):
            os.mkdir('visualizations')
        
        path = f'/Users/klara/Downloads/{id}.pdf'
        visualize_texts.get_one_visualization(option='term_frequency', paths=[path])#, outpath='visualizations')
        
        return id  # FIXME
        workingdir = os.path.abspath(os.getcwd())
        filepath = workingdir + '/visualizations/'
        return send_from_directory(filepath, f'{id}.pdf')

with app.test_request_context():
    #print(url_for('index'))
    print('hi')