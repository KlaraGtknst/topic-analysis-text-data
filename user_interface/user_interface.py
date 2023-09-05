from tkinter import *

# Button functions
def button_action():
    anweisungs_label.config(text="Ich wurde geändert!")

# build appearance of window
# window
window = Tk()
window.title("Topic Analysis Text Data")

# buttons
change_button = Button(window, text="Ändern", command=button_action)
exit_button = Button(window, text="Beenden", command=window.quit)

# labels
anweisungs_label = Label(window, text="Ich bin eine Anweisung:\nKlicke auf 'Ändern'.")
info_label = Label(window, text="Ich bin eine Info:\nDer Beenden Button schliesst das Programm.")

# place elements on canvas
anweisungs_label.pack()
change_button.pack()
info_label.pack()
exit_button.pack()




# run window
def main():
    window.mainloop()