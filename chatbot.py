import json
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
import numpy as np
import pickle

# Load disease database and model
diseases = json.load(open("diseases.JSN"))
model = load_model("disease_model.h5")

# Load words list
with open("words.pkl", "rb") as f:
    words = pickle.load(f)

# Initialize lemmatize
lemmatizer = WordNetLemmatizer()

# Define a function to process user input
def process_symptoms(text):
    # Tokenize and lemmatize user input
    sentence_words = nltk.word_tokenize(text)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]

    # Convert user input into a numerical representation
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1

    # Use machine learning model to predict possible diseases
    bow = np.array([bag])
    res = model.predict(bow)[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)

    # Return a list of possible diseases
    return_list = []
    for r in results:
        return_list.append({"disease": diseases[r[0]]["name"], "probability": str(r[1])})
    return return_list

# Main chatbot loop
while True:
    print("Enter your symptoms:")
    text = input()
    possible_diseases = process_symptoms(text)
    print("You might have:")
    for disease in possible_diseases:
        print(f"{disease['disease']} with a probability of {disease['probability']}")
