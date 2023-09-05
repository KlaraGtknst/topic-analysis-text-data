import glob
from tkinter import *
from text_visualizations import visualize_texts

# Button functions
def button_action():
    anweisungs_label.config(text="Ich wurde geändert!")

def run_wordCloud():
    visualize_texts.main([chosen_doc.get()], '/Users/klara/Downloads/')

# build appearance of window
# window
window = Tk()
window.title("Topic Analysis Text Data")

# buttons
change_button = Button(window, text="Ändern", command=button_action)
exit_button = Button(window, text="Beenden", command=window.quit)
wordCloud_button = Button(window, text="Word Cloud", command=run_wordCloud)

# labels
anweisungs_label = Label(window, text="Ich bin eine Anweisung:\nKlicke auf 'Ändern'.")
wordCloud_label = Label(window, text="Ich führe eine Wordcloud aus:\nKlicke auf 'Word Cloud'.")
doc_options_label = Label(window, text="Wähle ein Doc:\nKlicke auf 'Ändern'.")
info_label = Label(window, text="Ich bin eine Info:\nDer Beenden Button schliesst das Programm.")

# drop down menu
docs = glob.glob('/Users/klara/Downloads/*.pdf')
chosen_doc = StringVar(window)
chosen_doc.set(docs[0]) # default value
doc_options = OptionMenu(window, chosen_doc, *docs)

# place elements on canvas
anweisungs_label.grid(row=0, column=0, pady = 20)
change_button.grid(row=0, column=1, pady = 20)
wordCloud_label.grid(row=1, column=0, pady = 20)
wordCloud_button.grid(row=1, column=1, pady = 20)
info_label.grid(row=2, column=0)
exit_button.grid(row=2, column=1)
doc_options_label.grid(row=3, column=0)
doc_options.grid(row=3, column=1)




# run window
def main():
    window.mainloop()