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
        result = query_database.text_search_db(elastic_search_client, text=text, page=page, count=count)


    elif knn_source:
        # knn search
        print(knn_source, knn_type)
    else:
        # regular list
        result = query_database.get_docs_in_db(elastic_search_client, start=page, n_docs=count)
    # http://127.0.0.1:8000/documents
    return result

# return one document as JSON
@app.get('/documents/<id>')
def get_doc_meta_data(id):
    # http://127.0.0.1:8000/documents/SAC1-6
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
    
# return query for one document with query type x
@app.route('/documents/<id>/query/<query_type>')
def get_query(id, query_type): #TODO: async ?
    # http://127.0.0.1:8000/documents/SAC1-6/query/TF-IDF
    # get path of doc to search for
    elastic_search_client = Elasticsearch("http://localhost:9200")
    doc_to_search_for = query_database.get_doc_meta_data(elastic_search_client, doc_id=id)['path']

    # start query
    if query_type == 'TF-IDF':
        # alter src_path & client address if necessary
        query_results = query_database.get_sim_docs_tfidf(doc_to_search_for, src_paths=None)
        
    elif query_type == 'cluster':
        query_results = query_database.get_docs_from_same_cluster(elastic_search_client = elastic_search_client, path_to_doc = doc_to_search_for, n_results=10)#NUM_RESULTS)

    elif query_type == 'Doc2Vec':
        query_results = query_database.search_sim_doc2vec_docs_in_db(path=doc_to_search_for, client=elastic_search_client, doc2vec_model=None, src_paths=None)#SRC_PATH)

    elif query_type == 'UniversalSentenceEncoder':
       # TODO: await?
       query_results = query_database.find_sim_docs_google_univSentEnc(path=doc_to_search_for, client=elastic_search_client)
       #print(query_results.keys())

    elif query_type == 'HuggingFaceSentenceTransformer':
        query_results = query_database.find_sim_docs_hugging_face_sentTrans(path=doc_to_search_for, client=elastic_search_client)

    elif query_type == 'InferSent':
        query_results = query_database.find_sim_docs_inferSent(path=doc_to_search_for, client=elastic_search_client, src_paths=None)# glob.glob(SRC_PATH))
    
    result = [hit['_source'] for hit in query_results['hits']['hits']]
    return result


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