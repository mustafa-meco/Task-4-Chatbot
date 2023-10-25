import random
import json
import pickle
import numpy as np
import tkinter as tk
from tkinter import Scrollbar, Text
from tkinter import ttk

import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.h5')

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

def send_message():
    message = user_input.get()
    if message:
        chat_log.config(state=tk.NORMAL)
        chat_log.insert(tk.END, "You: " + message + "\n")
        user_input.delete(0, tk.END)
        ints = predict_class(message)
        res = get_response(ints, intents)
        chat_log.insert(tk.END, "🤖: " + res + "\n")
        chat_log.see(tk.END)
        chat_log.config(state=tk.DISABLED)

# Create the main GUI window
root = tk.Tk()
root.title("Chatbot")
root.geometry("550x350")  # Adjust the window size

# Create a styled text widget for the chat log
style = ttk.Style()
style.configure("TText", background="#F5F5F5", foreground="#333", font=("Arial", 12))
chat_log = Text(root, wrap=tk.WORD, state="disabled", width=50, height=15, relief="sunken", bg="#F5F5F5")
chat_log.pack(pady=10)

# Create a scrollbar for the chat log
scrollbar = Scrollbar(root, command=chat_log.yview)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
chat_log['yscrollcommand'] = scrollbar.set

# Create a frame for the input field and the "Send" button
input_frame = ttk.Frame(root)
input_frame.pack(pady=10)

# Create an entry widget for user input
user_input = tk.Entry(input_frame, font=("Arial", 12), relief="solid", width=40)
user_input.grid(row=0, column=0, padx=5)
user_input.bind("<Return>", lambda event: send_message())

# Create a "Send" button with a modern style
send_button = ttk.Button(input_frame, text="Send", command=send_message, style="TButton")
send_button.grid(row=0, column=1, padx=5)

# Define a new style for the "Send" button
style.configure("TButton", background="#007bff", foreground="white", font=("Arial", 12))

# Start the GUI event loop
root.mainloop()