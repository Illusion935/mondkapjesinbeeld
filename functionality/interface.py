import tkinter as tk
from tkinter import simpledialog


def interface():
    root = tk.Tk()
    root.after(600000, root.destroy)
    root.withdraw()

    with open("tekst.txt", "r") as file:
        first_line = file.readline()
    gebruiker_input = first_line  # Not asking anymore, just reading it from textfile
    # gebruiker_input = simpledialog.askstring(title="Mondkapjes in beeld", prompt="Vul hier de boodschap van vandaag in")

    # if gebruiker_input == "":
    #     gebruiker_input=first_line

    # elif gebruiker_input== None:
    #     gebruiker_input=first_line

    # else:
    #     with open ("tekst.txt", "w") as file:
    #         file.write(gebruiker_input)

    return gebruiker_input