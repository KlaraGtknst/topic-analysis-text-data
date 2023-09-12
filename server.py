import glob
import os
from elasticsearch import Elasticsearch
from flask import Flask, render_template, request, send_file, send_from_directory, url_for
from flask_restx import Api, Resource, fields # https://stackoverflow.com/questions/60156202/flask-app-wont-launch-importerror-cannot-import-name-cached-property-from-w
from text_visualizations import visualize_texts
from elasticSearch.queries import query_database
from flask_cors import CORS, cross_origin
# flask --app server run --debug --port 8000

app = Flask(__name__)
api = Api(app, version='1.0', title='Topic Analysis of large unstructured document data',
    description='API of the project.')
cors = CORS(app)

@api.doc(params={'page': {'description':'Page number', 'type':'int','default':0}, 
                 'count': {'description':'Number of documents per page', 'type':'int','default':10},
                 'text': 'Text to search for', 
                 'knn_source': 'Document to search for', 
                 'knn_type': {'description':'Type of knn search', 
                              'enum':[ "doc2vec","sim_docs_tfidf","google_univ_sent_encoding","huggingface_sent_transformer","inferSent_AE","pca_kmeans_cluster"]}})
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
        elastic_search_client = Elasticsearch("http://localhost:9200")

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
        elastic_search_client = Elasticsearch("http://localhost:9200")
        return query_database.get_doc_meta_data(elastic_search_client, doc_id=id)
        
@api.doc(params=id_doc)
@api.route('/documents/<id>/pdf', endpoint='pdf')
class PDF(Resource):
    # return one document as PDF
    def get(self, id):
        # http://127.0.0.1:8000/documents/SAC1-6.pdf
        elastic_search_client = Elasticsearch("http://localhost:9200")
        resp_path = query_database.get_doc_meta_data(elastic_search_client, doc_id=id)['path']
        print('*'*50)
        print(resp_path)    # FIXME: path is not relative under etc. current dir, cannot be displayed
        return send_file(resp_path)

@api.doc(params=id_doc)
@api.route('/documents/<id>/wordcloud', endpoint='wordcloud')
class WordCloud(Resource):
    # return wordcloud of one document as PNG
    def get(self, id):
        # http://127.0.0.1:8000/documents/SAC1-6/wordcloud

        if not os.path.exists('visualizations'):
            os.mkdir('visualizations')
        
        path = f'/Users/klara/Downloads/{id}.pdf'
        visualize_texts.get_one_visualization(option='wordcloud', paths=[path], outpath='visualizations')

        workingdir = os.path.abspath(os.getcwd())
        filepath = workingdir + '/visualizations/'

        return send_from_directory(filepath, f'{id}.pdf')

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