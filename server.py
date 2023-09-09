from elasticsearch import Elasticsearch
from flask import Flask, render_template, url_for
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

# return one document as JSON
@app.get('/documents/<id>')
def get_doc_meta_data(id):
    print(id) # http://127.0.0.1:8000/documents/SAC1-6
    elastic_search_client = Elasticsearch("http://localhost:9200")
    resp = query_database.get_doc_meta_data(elastic_search_client, doc_id=id)
    return resp

# return one document as PDF
#@app.route('/documents/<id>.pdf')
def hello(id):
    print(id)
    return render_template('hello.html', name=id)

# return wordcloud of one document as PNG
@app.route('/documents/<id>/wordcloud')
def get_wordcloud(id):
    print(id) # http://127.0.0.1:8000/documents/SAC1-6/wordcloud
    path = f'/Users/klara/Downloads/{id}.pdf'
    return visualize_texts.get_one_visualization(option='wordcloud', paths=[path])

# return term frequency of one document as PNG
@app.route('/documents/<id>/term_frequency')
def get_termfreq(id):
    print(id) # http://127.0.0.1:8000/documents/SAC1-6/term_frequency
    path = f'/Users/klara/Downloads/{id}.pdf'
    return visualize_texts.get_one_visualization(option='term_frequency', paths=[path])

with app.test_request_context():
    print(url_for('index'))
    '''print(url_for('login'))
    print(url_for('login', next='/'))
    print(url_for('profile', username='John Doe'))
'''