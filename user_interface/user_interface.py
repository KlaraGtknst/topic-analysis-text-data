import glob
from tkinter import *

from elasticsearch import Elasticsearch
from text_visualizations import visualize_texts
from elasticSearch.queries import query_database
from elasticSearch.queries.query_documents_tfidf import *

 # Create the client instance
client = Elasticsearch("http://localhost:9200")
results = {}

# Button functions
def run_wordCloud():
    visualize_texts.main([chosen_doc.get()], '/Users/klara/Downloads/')


def run_query():
    print('start')
    NUM_RESULTS = 4
    doc_to_search_for = chosen_doc.get()
    query_info_label.config(text=f'Query gestartet für {doc_to_search_for}.')

    query_type = chosen_query_type.get()
    print(query_type)

    if query_type == 'TF-IDF':
        # alter src_path & client address if necessary
        tfidf_results = query_database.get_sim_docs_tfidf(doc_to_search_for)
        results['tfidf'] = {doc_to_search_for: ['/'.join(doc_to_search_for.split('/')[:-1]) + '/' + doc for doc in tfidf_results.values()]}

        query_result_status_label.config(text=f'Query Resultat for {doc_to_search_for}:')
        query_result_content_label.config(text='\n'.join(results["tfidf"][doc_to_search_for]))

    elif query_type == 'cluster':
        cluster_results = query_database.get_docs_from_same_cluster(elastic_search_client = client, path_to_doc = doc_to_search_for, n_results=NUM_RESULTS)
        result = [hit['_source']['path'] for hit in cluster_results['hits']['hits']]
        results['cluster'] = {doc_to_search_for: result}

        query_result_status_label.config(text=f'Query Resultat for {doc_to_search_for}:')
        query_result_content_label.config(text='\n'.join(results["cluster"][doc_to_search_for]))


# build appearance of window
# window
window = Tk()
window.title("Topic Analysis Text Data")

# buttons
buttons = []
exit_button = Button(window, text="Beenden", command=window.quit)
wordCloud_button = Button(window, text="Word Cloud", command=run_wordCloud)
# # querying db
query_button = Button(window, text="Query", command=run_query)

# drop down menu 
# # for document option
docs = glob.glob('/Users/klara/Downloads/*.pdf')
chosen_doc = StringVar(window)
chosen_doc.set(docs[0]) # default value
doc_options = OptionMenu(window, chosen_doc, *docs)
# # for query type
query_options = ['TF-IDF', 'cluster']#, 'InferSent', 'Universal Sentence Encoder', 'Hugging Face Sentence Transformer']
chosen_query_type = StringVar(window)
chosen_query_type.set(query_options[0]) # default value
query_dropdown = OptionMenu(window, chosen_query_type, *query_options)


# labels
labels = []
wordCloud_label = Label(window, text="Ich führe eine Wordcloud aus:\nKlicke auf 'Word Cloud'.")
doc_options_label = Label(window, text="Wähle ein Doc:\nKlicke auf 'Ändern'.")
query_options_label = Label(window, text="Wähle ein Query Typ:\nKlicke auf 'Ändern'.")
exit_label = Label(window, text="Der Beenden Button schliesst das Programm.")
query_info_label = Label(window, text="Keine Query gestartet.")
query_result_status_label = Label(window, text="Kein Query Resultat.")
query_result_content_label = Label(window, text="-")
labels.extend([doc_options_label, wordCloud_label, query_options_label, query_info_label, query_result_status_label, exit_label])
buttons.extend([doc_options, wordCloud_button, query_dropdown, query_button, query_result_content_label, exit_button])







# place elements on canvas
'''anweisungs_label.grid(row=0, column=0, pady = 20)
change_button.grid(row=0, column=1, pady = 20)

wordCloud_label.grid(row=1, column=0, pady = 20)
wordCloud_button.grid(row=1, column=1, pady = 20)

info_label.grid(row=-1, column=0)
exit_button.grid(row=-1, column=1)

doc_options_label.grid(row=3, column=0)
doc_options.grid(row=3, column=1)

query_options_label.grid(row=4, column=0)
query_dropdown.grid(row=4, column=1)

query_info_label.grid(row=5, column=0)
query_button.grid(row=5, column=1)

query_result_label.grid(row=6, column=0)'''

for row in range(len(labels)):
    labels[row].grid(row=row, column=0)
    buttons[row].grid(row=row, column=1)




# run window
def main():
   
    window.mainloop()