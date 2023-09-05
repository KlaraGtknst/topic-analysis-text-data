import glob
from tkinter import *

from elasticsearch import Elasticsearch
from text_visualizations import visualize_texts
from elasticSearch.queries import query_database
from elasticSearch.queries.query_documents_tfidf import *
from elasticSearch import db_elasticsearch
from gensim.models.doc2vec import Doc2Vec

# TODO
SRC_PATH = glob.glob('/Users/klara/Documents/uni/bachelorarbeit/data/0/*.pdf')
NUM_DIMENSIONS = 55
NUM_COMPONENTS = 2
NUM_RESULTS = 4

 # Create the client instance
client = Elasticsearch("http://localhost:9200")
results = {}







class PageOne(Frame):

    def __init__(self, parent, controller):
        Frame.__init__(self, parent)
        # build appearance of window
        # window
        #self = Tk()
        #self.title("Topic Analysis Text Data")

        # buttons
        self.buttons = []
        self.exit_button = Button(self, text="Beenden", command=self.quit)
        self.wordCloud_button = Button(self, text="Word Cloud", command=self.run_wordCloud)
        # # querying db
        self.query_button = Button(self, text="Query", command=self.run_query)

        # drop down menu 
        # # for document option
        self.docs = glob.glob('/Users/klara/Downloads/*.pdf')
        self.chosen_doc = StringVar(self)
        self.chosen_doc.set(self.docs[0]) # default value
        self.doc_options = OptionMenu(self, self.chosen_doc, *self.docs)
        # # for query type
        self.query_options = ['TF-IDF', 'cluster', 'Doc2Vec']#, 'InferSent', 'Universal Sentence Encoder', 'Hugging Face Sentence Transformer']
        self.chosen_query_type = StringVar(self)
        self.chosen_query_type.set(self.query_options[0]) # default value
        self.query_dropdown = OptionMenu(self, self.chosen_query_type, *self.query_options)


        # labels
        self.labels = []
        self.wordCloud_label = Label(self, text="Ich führe eine Wordcloud aus:\nKlicke auf 'Word Cloud'.")
        self.doc_options_label = Label(self, text="Wähle ein Doc:\nKlicke auf 'Ändern'.")
        self.query_options_label = Label(self, text="Wähle ein Query Typ:\nKlicke auf 'Ändern'.")
        self.exit_label = Label(self, text="Der Beenden Button schliesst das Programm.")
        self.query_info_label = Label(self, text="Keine Query gestartet.")
        self.query_result_status_label = Label(self, text="Kein Query Resultat.")
        self.query_result_content_label = Label(self, text="-")
        self.labels.extend([self.doc_options_label, self.wordCloud_label, self.query_options_label, self.query_info_label, self.query_result_status_label, self.exit_label])
        self.buttons.extend([self.doc_options, self.wordCloud_button, self.query_dropdown, self.query_button, self.query_result_content_label, self.exit_button])



        # place elements on canvas
        for row in range(len(self.labels)):
            self.labels[row].grid(row=row, column=0)
            self.buttons[row].grid(row=row, column=1)

    # Button functions
    def run_wordCloud(self):
        visualize_texts.main([self.chosen_doc.get()], '/Users/klara/Downloads/')


    def run_query(self):
        doc_to_search_for = self.chosen_doc.get()
        self.query_info_label.config(text=f'Query gestartet für {doc_to_search_for}.')
        query_type = self.chosen_query_type.get()
        print(query_type)

        if query_type == 'TF-IDF':
            # alter src_path & client address if necessary
            tfidf_results = query_database.get_sim_docs_tfidf(doc_to_search_for, src_paths=SRC_PATH)
            results['tfidf'] = {doc_to_search_for: ['/'.join(doc_to_search_for.split('/')[:-1]) + '/' + doc for doc in tfidf_results.values()]}

            self.query_result_status_label.config(text=f'Query Resultat for {doc_to_search_for}:')
            self.query_result_content_label.config(text='\n'.join(results["tfidf"][doc_to_search_for]))

        elif query_type == 'cluster':
            cluster_results = query_database.get_docs_from_same_cluster(elastic_search_client = client, path_to_doc = doc_to_search_for, n_results=NUM_RESULTS)
            result = [hit['_source']['path'] for hit in cluster_results['hits']['hits']]
            results['cluster'] = {doc_to_search_for: result}

            self.query_result_status_label.config(text=f'Query Resultat for {doc_to_search_for}:')
            self.query_result_content_label.config(text='\n'.join(results["cluster"][doc_to_search_for]))

        elif query_type == 'Doc2Vec':
            train_corpus = list(db_elasticsearch.get_tagged_input_documents(SRC_PATH))
            d2v_model = Doc2Vec(train_corpus, vector_size=NUM_DIMENSIONS, window=2, min_count=2, workers=4, epochs=40)
            doc2vec_result = query_database.search_in_db(path=doc_to_search_for, client=client, model=d2v_model)
            results['Doc2Vec'] = {doc_to_search_for: doc2vec_result.values()}

            self.query_result_status_label.config(text=f'Query Resultat for {doc_to_search_for}:')
            self.query_result_content_label.config(text='\n'.join(results["Doc2Vec"][doc_to_search_for]))

class StartPage(Frame):

    def __init__(self, parent, controller):
        Frame.__init__(self,parent)
        label = Label(self, text="Start Page", font=12)
        label.pack(pady=10,padx=10)

        button = Button(self, text="Visit Page 1",
                            command=lambda: controller.show_frame(PageOne))
        button.pack()

        button2 = Button(self, text="Visit Page 2",
                            command=lambda: controller.show_frame(PageTwo))
        button2.pack()

class PageTwo(Frame):

    def __init__(self, parent, controller):
        Frame.__init__(self, parent)
        self.controller = controller
        label = Label(self, text="This is page 2", font=controller.title_font)
        label.pack(side="top", fill="x", pady=10)
        button = Button(self, text="Go to the start page",
                           command=lambda: controller.show_frame(StartPage))
        button.pack()

class SeaofBTCapp(Tk):

    def __init__(self, *args, **kwargs):
        Tk.__init__(self, *args, **kwargs)
        container = Frame(self)
        container.pack(side="top", fill="both", expand = True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)
        self.title_font = ("Helvetica", 12, "bold")
        self.frames = {}
        for F in (StartPage, PageOne, PageTwo):
            frame = F(container, self)
            self.frames[F] = frame
            frame.grid(row=0, column=0, sticky="nsew")
        print(self.frames)
        self.show_frame(StartPage)

    def show_frame(self, cont):
        frame = self.frames[cont]
        print(cont)
        frame.tkraise()



# run window
def main():
    app = SeaofBTCapp()
    app.mainloop()