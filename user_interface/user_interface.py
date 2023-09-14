import base64
import glob
import io
from tkinter import *

from elasticsearch import Elasticsearch
from text_visualizations import visualize_texts
from elasticSearch.queries import query_database
from elasticSearch.queries.query_documents_tfidf import *
from elasticSearch import db_elasticsearch
from gensim.models.doc2vec import Doc2Vec
from doc_images import pdf_matrix, convert_pdf2image
from PIL import Image
from constants import CLIENT_ADDR

# TODO
SRC_PATH = '/Users/klara/Documents/uni/bachelorarbeit/data/0/*.pdf'
DOC_PATH = '/Users/klara/Downloads/*.pdf'
NUM_DIMENSIONS = 55
NUM_COMPONENTS = 2
NUM_RESULTS = 4

 # Create the client instance
client = Elasticsearch(CLIENT_ADDR)
results = {}


class QueryPage(Frame):

    def __init__(self, parent, controller):
        Frame.__init__(self, parent)
        # buttons
        self.buttons = []
        self.exit_button = Button(self,  text="Visualization Page", command=lambda: controller.show_frame(ImgPage))
        # # querying db
        self.query_button = Button(self, text="Query", command=self.run_query)
        # # visualizing query results
        self.wordCloud_results_button = Button(self, text="Word Cloud", state='disabled', command=lambda: visualize_texts.get_one_visualization(option='wordcloud', paths=results[self.chosen_query_type.get()][self.chosen_doc.get()]))
        self.termFreq_results_button = Button(self, text="Term Frequency", state='disabled', command=lambda: visualize_texts.get_one_visualization(option='term_frequency', paths=results[self.chosen_query_type.get()][self.chosen_doc.get()]))

        # drop down menu 
        # # for document option
        self.docs = glob.glob(DOC_PATH)
        self.chosen_doc = StringVar(self)
        self.chosen_doc.set(self.docs[0]) # default value
        self.doc_options = OptionMenu(self, self.chosen_doc, *self.docs)
        # # for query type
        self.query_options = ['TF-IDF', 'cluster', 'Doc2Vec', 'Universal Sentence Encoder', 'Hugging Face Sentence Transformer', 'InferSent']
        self.chosen_query_type = StringVar(self)
        self.chosen_query_type.set(self.query_options[0]) # default value
        self.query_dropdown = OptionMenu(self, self.chosen_query_type, *self.query_options)

        # Checkbox
        self.display_query_res_matrix = IntVar()
        Checkbutton(self, text="display result matrix", variable=self.display_query_res_matrix).grid(row=2, column=3)


        # labels
        self.labels = []
        self.doc_options_label = Label(self, text="Wähle ein Doc:\nKlicke auf 'Ändern'.")
        self.query_options_label = Label(self, text="Wähle ein Query Typ:\nKlicke auf 'Ändern'.")
        self.exit_label = Label(self, text="Visit the visualization page:\nKlicke auf 'Visualisierung'.")
        self.query_info_label = Label(self, text="Keine Query gestartet.")
        self.query_result_status_label = Label(self, text="Kein Query Resultat.")
        self.query_result_content_label = Label(self, text="-")
        self.vis_query_res_wordCloud_label = Label(self, text="Visualisiere Query Resultat mittels WordCloud:")
        self.vis_query_res_termFreq_label = Label(self, text="Visualisiere Query Resultat mittels Term Frequency:")

        self.labels.extend([self.doc_options_label, self.query_options_label, self.query_info_label, self.query_result_status_label, self.vis_query_res_wordCloud_label, self.vis_query_res_termFreq_label,self.exit_label])
        self.buttons.extend([self.doc_options, self.query_dropdown, self.query_button, self.query_result_content_label, self.wordCloud_results_button, self.termFreq_results_button, self.exit_button])



        # place elements on canvas
        for row in range(len(self.labels)):
            self.labels[row].grid(row=row, column=0)
            self.buttons[row].grid(row=row, column=1)

        


    def run_query(self):
        self.wordCloud_results_button.config(state='disabled') # TODO: why does it not update?
        doc_to_search_for = self.chosen_doc.get()
        self.query_info_label.config(text=f'Query gestartet für {doc_to_search_for}.')
        query_type = self.chosen_query_type.get()
        print(query_type)

        if query_type == 'TF-IDF':
            # alter src_path & client address if necessary
            tfidf_results = query_database.get_sim_docs_tfidf(doc_to_search_for, src_paths=SRC_PATH)

            self.react_on_results(doc_to_search_for, query_type, tfidf_results)

        elif query_type == 'cluster':
            cluster_results = query_database.get_docs_from_same_cluster(elastic_search_client = client, path_to_doc = doc_to_search_for, n_results=NUM_RESULTS)

            self.react_on_results(doc_to_search_for, query_type, cluster_results)

        elif query_type == 'Doc2Vec':
            doc2vec_result = query_database.search_sim_doc2vec_docs_in_db(path=doc_to_search_for, client=client, doc2vec_model=None, src_paths=SRC_PATH)

            self.react_on_results(doc_to_search_for, query_type, doc2vec_result)

        elif query_type == 'Universal Sentence Encoder':
            univSentEnc_result = query_database.find_sim_docs_google_univSentEnc(path=doc_to_search_for, client=client)

            self.react_on_results(doc_to_search_for, query_type, univSentEnc_result)

        elif query_type == 'Hugging Face Sentence Transformer':
            hf_sentTrans_result = query_database.find_sim_docs_hugging_face_sentTrans(path=doc_to_search_for, client=client)

            self.react_on_results(doc_to_search_for, query_type, hf_sentTrans_result)

        elif query_type == 'InferSent':
            infersent_result = query_database.find_sim_docs_inferSent(src_paths= glob.glob(SRC_PATH), path=doc_to_search_for, client=client)

            self.react_on_results(doc_to_search_for, query_type, infersent_result)

    def get_result_images(self, query_results):
        '''
        :param query_results: results of querying the database. Data includes paths to most similar documents and their images as b64 strings.
        :return: list of paths to most similar documents, list of b64 strings of images
        '''
        result = [hit['_source']['path'] for hit in query_results['hits']['hits']]
        b64_images = [hit['_source']['image'] for hit in query_results['hits']['hits']] if 'image' in list(query_results['hits']['hits'][0]['_source'].keys()) else None
        return result, b64_images


    def react_on_results(self, doc_to_search_for, query_type, query_results):
        '''
        :param doc_to_search_for: path to document to search for
        :param query_type: type of query
        :param query_results: results of querying the database. Data includes paths to most similar documents and their images as b64 strings.
        :return: None
        
        This method saves the results in a dictionary, updates the query result status label, the query result content label and the wordCloud and termFreq buttons.
        If the query result contains images and the display-checkbox is checked, it displays them on the canvas.
        '''
        result, b64_images = self.get_result_images(query_results)
        results[query_type] = {doc_to_search_for: result}
        
        self.query_result_status_label.config(text=f'Query Resultat for {doc_to_search_for}:')
        self.query_result_content_label.config(text='\n'.join(results[query_type][doc_to_search_for]))
        self.wordCloud_results_button.config(state='active')
        self.termFreq_results_button.config(state='active')
        
        if b64_images and self.display_query_res_matrix.get():
            self.images = []

            for i in range(len(b64_images)):
                b64img = b64_images[i]
                b64img = b64img[2:-1]
                buffer = io.BytesIO()
                imgdata = base64.b64decode(b64img)
                img = Image.open(io.BytesIO(imgdata))
                new_img = img.resize((70, 70))
                new_img.save(buffer, format="PNG")
                b64img = base64.b64encode(buffer.getvalue())

                im = PhotoImage(data=b64img, height=70, width=70)
                imglabel = Label(self, image=im)
                imglabel.grid(row=i % 5, column= 4 + int(i / 5))
                self.images.append(im)



class ImgPage(Frame):

    def __init__(self, parent, controller):
        Frame.__init__(self, parent)
        self.controller = controller

        # labels
        labels = []
        self.wordCloud_label = Label(self, text="Ich führe eine Wordcloud aus:\nKlicke auf 'Word Cloud'.")
        self.termFrequency_label = Label(self, text="Ich führe eine Term Frequency aus:\nKlicke auf 'Term Frequency'.")
        self.doc_options_label = Label(self, text="Wähle ein Doc:\nKlicke auf 'Ändern'.")
        self.exit_label = Label(self, text="Visit the query page:\nKlicke auf 'Query'.")
      

        # buttons
        buttons = []
        self.wordCloud_button = Button(self, text="Word Cloud", command=self.run_wordCloud)
        self.termFrequency_button = Button(self, text="Term Frequency", command=self.run_term_frq)
        exit_button = Button(self, text="Go to the start page",
                           command=lambda: controller.show_frame(StartPage))

        # drop down menu 
        # # for document option
        self.docs = glob.glob(DOC_PATH)
        self.chosen_doc = StringVar(self)
        self.chosen_doc.set(self.docs[0]) # default value
        self.doc_options = OptionMenu(self, self.chosen_doc, *self.docs)

        labels.extend([self.doc_options_label, self.wordCloud_label, self.termFrequency_label, self.exit_label])
        buttons.extend([self.doc_options, self.wordCloud_button, self.termFrequency_button, exit_button])

        # place elements on canvas
        for row in range(len(labels)):
            labels[row].grid(row=row, column=0)
            buttons[row].grid(row=row, column=1)

    # Button functions
    def run_wordCloud(self):
        visualize_texts.get_one_visualization(option='wordcloud', paths=[self.chosen_doc.get()])

    def run_term_frq(self):
        visualize_texts.get_one_visualization(option='term_frequency', paths=[self.chosen_doc.get()])


class StartPage(Frame):

    def __init__(self, parent, controller):
        Frame.__init__(self,parent)
        label = Label(self, text="Start Page", font=12)
        label.pack(pady=10,padx=10)

        button = Button(self, text="Visit Query Page",
                            command=lambda: controller.show_frame(QueryPage))
        button.pack()

        button2 = Button(self, text="Visit Visualization Page",
                            command=lambda: controller.show_frame(ImgPage))
        button2.pack()



class Frame_Controller(Tk):

    def __init__(self, *args, **kwargs):
        Tk.__init__(self, *args, **kwargs)
        self.title("Topic Analysis Text Data")
        container = Frame(self)
        container.pack(side="top", fill="both", expand = True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)
        self.title_font = ("Helvetica", 12, "bold")
        self.frames = {}
        for F in (StartPage, QueryPage, ImgPage):
            frame = F(container, self)
            self.frames[F] = frame
            frame.grid(row=0, column=0, sticky="nsew")
        self.show_frame(StartPage)

    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()



# run window
def main():
    app = Frame_Controller()
    app.mainloop()