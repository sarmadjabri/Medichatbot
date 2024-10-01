import json
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras import models, layers
import numpy as np
import pickle

# Load disease database and model
diseases = json.load(open("diseases.json"))

# Load words list
with open("words.pkl", "rb") as f:
    words = pickle.load(f)

# Initialize lemmatizer for breaking down words
lemmatizer = WordNetLemmatizer()

# Download required NLTK data
nltk.download('punkt')
nltk.download('wordnet')

def process_symptoms(text):
    """
    Process user input symptoms and predict possible diseases.

    Args:
        text (str): User input symptoms.

    Returns:
        list: A list of possible diseases with their probabilities.
    """
    # Tokenize and lemmatize user input
    sentence_words = nltk.tokenize.regexp.word_tokenize(text, preserve_case=False)
    sentence_words = nltk.tokenize.treebank.TreebankWordTokenizer().tokenize(text)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]

    # Remove stop words
    stop_words = set(nltk.corpus.stopwords.words('english'))
    sentence_words = [word for word in sentence_words if word not in stop_words]

    # Convert user input into a numerical representation
    sequence_length = 100
    padded_sequence = np.zeros((sequence_length,))
    for i, word in enumerate(sentence_words):
        if i >= sequence_length:
            break
        if word in words:
            padded_sequence[i] = words.index(word)

    # Use LSTM model to predict possible diseases
    model = models.Sequential([
        layers.Embedding(input_dim=len(words), output_dim=128, input_length=sequence_length),
        layers.LSTM(64),
        layers.Dense(len(diseases), activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    res = model.predict(np.array([padded_sequence]))[0]
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
    print("It is possible you have:")
    for disease in possible_diseases:
        print(f"{disease['disease']} with a probability of {disease['probability']}")

