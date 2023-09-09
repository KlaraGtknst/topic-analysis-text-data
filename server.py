import glob
import os
from elasticsearch import Elasticsearch
from flask import Flask, render_template, send_file, send_from_directory, url_for
from text_visualizations import visualize_texts
from elasticSearch.queries import query_database
# flask --app server run --debug --port 8000

app = Flask(__name__)

@app.route('/')
def index():
    return 'index'

# return all documents
@app.route('/documents')
def login():
    return 'login'

# return x documents on page y
@app.route('/documents/<int:page>/<int:count>')
def get_docs_per_page(page:int=0, count:int=10):
    # http://127.0.0.1:8000/documents/0/10
    elastic_search_client = Elasticsearch("http://localhost:9200")
    # TODO: change to integer when index changed in db
    doc_ids = glob.glob('/Users/klara/Documents/Uni/bachelorarbeit/data/0/*.pdf')
    doc_ids= [doc.split('/')[-1].split('.')[0] for doc in  doc_ids[page*count:page*count+count]]
    return query_database.get_docs_in_db(elastic_search_client, indices=doc_ids, start=page, n_docs=count)

# return one document as JSON
@app.get('/documents/<id>')
def get_doc_meta_data(id):
    print(id) # http://127.0.0.1:8000/documents/SAC1-6
    elastic_search_client = Elasticsearch("http://localhost:9200")
    resp = query_database.get_doc_meta_data(elastic_search_client, doc_id=id)
    return resp

# return one document as PDF
@app.route('/documents/pdf/<id>')
def hello(id):
    # http://127.0.0.1:8000/documents/pdf/SAC1-6
    elastic_search_client = Elasticsearch("http://localhost:9200")
    resp_path = query_database.get_doc_meta_data(elastic_search_client, doc_id=id)['path']
    print(resp_path)
    # TODO: save the file in the POST request, and then from the socket event read the file that you saved, cf. https://github.com/miguelgrinberg/Flask-SocketIO/issues/1351
    with open(resp_path, 'rb') as static_file:
        return send_file(static_file, download_name=f'{id}.pdf')

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
    '''print(url_for('login'))
    print(url_for('login', next='/'))
    print(url_for('profile', username='John Doe'))
'''