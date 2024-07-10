import json
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
import numpy as np
import pickle

# Load disease database and za model
diseases = json.load(open("diseases.json"))
model = load_model("disease_model.h5")

# Load words list
with open("words.pkl", "rb") as f:
    words = pickle.load(f)

# Initialize lemmatize
lemmatizer = WordNetLemmatizer()

# Download stuff for za NLTK
nltk.download('punkt')
nltk.download('wordnet')

# user input
def process_symptoms(text):
    # Tokenize and lemmatize user input
    sentence_words = nltk.word_tokenize(text)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]

    # Convert input into a numerical presentation
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1

    # Prediction
    bow = np.array([bag])
    res = model.predict(bow)[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)

    # Return list(diseases)
    return_list = []
    for r in results:
        return_list.append({"disease": diseases[r[0]]["name"], "probability": str(r[1])})
    return return_list

# Loop for bot
while True:
    print("Enter your symptoms:")
    text = input()
    possible_diseases = process_symptoms(text)
    print("It is possible you have:")
    for disease in possible_diseases:
        print(f"{disease['disease']} with a probability of {disease['probability']}")
