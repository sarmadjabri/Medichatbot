import json
import nltk
import numpy as np
import pickle
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
from database import save_interaction
from response_generator import generate_response
from intent_classifier import classify_intent
from user_feedback import ask_for_feedback

nltk.download()

# Load words list
with open("words.pkl", "rb") as f:
    words = pickle.load(f)

# Load disease model
model = load_model("disease_model.h5")

# Initialize lemmatizer and download required NLTK data
lemmatizer = WordNetLemmatizer()
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)

# Define constants
ERROR_THRESHOLD = 0.25
SEQUENCE_LENGTH = 100

def preprocess_input(text):
    """Tokenize and lemmatize the input text, removing stopwords."""
    tokens = nltk.word_tokenize(text.lower())
    lemmatized_words = [lemmatizer.lemmatize(word) for word in tokens if word.isalnum()]
    stop_words = set(nltk.corpus.stopwords.words('english'))
    return [word for word in lemmatized_words if word not in stop_words]

def text_to_sequence(filtered_words):
    """Convert words to a numerical sequence, padding to the defined sequence length."""
    padded_sequence = np.zeros(SEQUENCE_LENGTH)
    for i, word in enumerate(filtered_words):
        if i >= SEQUENCE_LENGTH:
            break
        if word in words:
            padded_sequence[i] = words.index(word)
    return padded_sequence

def predict_diseases(model, padded_sequence):
    """Predict possible diseases based on input sequence."""
    predictions = model.predict(np.array([padded_sequence]))[0]
    results = [(i, prob) for i, prob in enumerate(predictions) if prob > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return [{"disease": disease, "probability": round(prob, 2)} for i, prob in results]

def main():
    """Main function to run the chatbot."""
    print("Welcome to the Disease Prediction Chatbot!")
    print("Type 'exit' anytime to end the conversation.\n")

    while True:
        user_input = input("🔍 Enter your symptoms: ")

        if user_input.lower() == 'exit':
            print("👋 Thank you for chatting!")
            break

        intent = classify_intent(user_input)
        response = generate_response(intent)

        if intent == 'symptom_check':
            filtered_words = preprocess_input(user_input)
            padded_sequence = text_to_sequence(filtered_words)
            possible_diseases = predict_diseases(model, padded_sequence)

            if possible_diseases:
                print("\n🤔 Based on what you've described, here are some possible conditions:")
                for disease in possible_diseases:
                    print(f"  - {disease['disease']} (Probability: {disease['probability']})")
                ask_for_feedback()
            else:
                print("🧐 Sorry, I couldn't find any matching diseases. Try rephrasing.")

        save_interaction(user_input, response)
        print(response)

if __name__ == "__main__":
    main()
