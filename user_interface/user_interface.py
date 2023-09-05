from tkinter import *
from text_visualizations import visualize_texts

# Button functions
def button_action():
    anweisungs_label.config(text="Ich wurde geändert!")

def run_wordCloud():
    visualize_texts.main(['/Users/klara/Downloads/SAC2-12.pdf'], '/Users/klara/Downloads/')

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
info_label = Label(window, text="Ich bin eine Info:\nDer Beenden Button schliesst das Programm.")

# place elements on canvas
anweisungs_label.pack()
change_button.pack()
info_label.pack()
wordCloud_button.pack()
exit_button.pack()




# run window
def main():
    window.mainloop()