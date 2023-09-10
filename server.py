import glob
import os
from elasticsearch import Elasticsearch
from flask import Flask, render_template, request, send_file, send_from_directory, url_for
from text_visualizations import visualize_texts
from elasticSearch.queries import query_database
# flask --app server run --debug --port 8000

app = Flask(__name__)

@app.route('/')
def index():
    return 'index'

# return all documents
@app.route('/documents')
def get_all_docs_in_db():
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

# return one document as JSON
@app.route('/documents/<id>')
def get_one_doc(id):
    # http://127.0.0.1:8000/documents/SAC1-6
    elastic_search_client = Elasticsearch("http://localhost:9200")
    return query_database.get_doc_meta_data(elastic_search_client, doc_id=id)
    
# return one document as PDF
@app.route('/documents/<id>.pdf')
def get_one_doc_pdf(id):
    # http://127.0.0.1:8000/documents/SAC1-6.pdf
    elastic_search_client = Elasticsearch("http://localhost:9200")
    resp_path = query_database.get_doc_meta_data(elastic_search_client, doc_id=id)['path']
    print('*'*50)
    print(resp_path)    # FIXME: path is not relative under etc. current dir, cannot be displayed
    return send_from_directory(resp_path, f'{id}.pdf')


# return wordcloud of one document as PNG
@app.route('/documents/<id>/wordcloud')
def get_wordcloud(id):
    print(id) # http://127.0.0.1:8000/documents/SAC1-6/wordcloud

    if not os.path.exists('visualizations'):
        os.mkdir('visualizations')
    
    path = f'/Users/klara/Downloads/{id}.pdf'
    visualize_texts.get_one_visualization(option='wordcloud', paths=[path], outpath='visualizations')

    workingdir = os.path.abspath(os.getcwd())
    filepath = workingdir + '/visualizations/'

    return send_from_directory(filepath, f'{id}.pdf')

# return term frequency of one document as PNG
@app.route('/documents/<id>/term_frequency')
def get_termfreq(id):
    # TODO: matplotlib nicht umgehbar, gehtvielleicht nicht
    print(id) # http://127.0.0.1:8000/documents/SAC1-6/term_frequency
    
    if not os.path.exists('visualizations'):
        os.mkdir('visualizations')
    
    path = f'/Users/klara/Downloads/{id}.pdf'
    visualize_texts.get_one_visualization(option='term_frequency', paths=[path])#, outpath='visualizations')
    
    return id 
    workingdir = os.path.abspath(os.getcwd())
    filepath = workingdir + '/visualizations/'
    return send_from_directory(filepath, f'{id}.pdf')

with app.test_request_context():
    print(url_for('index'))