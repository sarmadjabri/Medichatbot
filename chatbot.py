import json
import nltk
import numpy as np
import pickle
from nltk.stem import WordNetLemmatizer
from tensorflow.keras import models, layers

nltk.download()


# Define a list of diseases
diseases = [
    {"name": "Flu"},
    {"name": "Cold"},
    {"name": "COVID-19"},
    # Add more diseases as needed
]

# Load words list
with open("words.pkl", "rb") as f:
    words = pickle.load(f)

# Initialize lemmatizer and download required NLTK data
lemmatizer = WordNetLemmatizer()
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)

# Define constants
ERROR_THRESHOLD = 0.25
SEQUENCE_LENGTH = 100

def load_model():
    """Build and compile the LSTM model."""
    model = models.Sequential([
        layers.Embedding(input_dim=len(words), output_dim=128, input_length=SEQUENCE_LENGTH),
        layers.LSTM(64),
        layers.Dense(len(diseases), activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def preprocess_input(text):
    """Tokenize and lemmatize the input text, removing stopwords."""
    # Tokenize the input
    tokens = nltk.word_tokenize(text.lower())
    
    # Lemmatize and filter out non-alphanumeric tokens
    lemmatized_words = [
        lemmatizer.lemmatize(word) 
        for word in tokens 
        if word.isalnum()
    ]

    # Remove stopwords
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

    return [{"disease": diseases[i]["name"], "probability": round(prob, 2)} for i, prob in results]

def main():
    """Main function to run the chatbot."""
    model = load_model()
    print("🌟 Welcome to the Disease Prediction Chatbot! 🌟")
    print("🤖 I'm here to help you. Just type your symptoms, and I'll do my best to assist you.")
    print("💬 Type 'exit' anytime to end the conversation.\n")

    while True:
        user_input = input("🔍 Enter your symptoms: ")
        if user_input.lower() == 'exit':
            print("👋 Thank you for chatting! Remember to take care of yourself!")
            break

        filtered_words = preprocess_input(user_input)

        if not filtered_words:
            print("⚠️ I couldn't understand that. Could you please describe your symptoms more clearly?")
            continue

        padded_sequence = text_to_sequence(filtered_words)
        possible_diseases = predict_diseases(model, padded_sequence)

        if possible_diseases:
            print("\n🤔 Based on what you've described, here are some possible conditions:")
            for disease in possible_diseases:
                print(f"  - {disease['disease']} (Probability: {disease['probability']})")
            print("\nIf you're concerned, I recommend consulting a healthcare professional for a proper diagnosis.")
        else:
            print("🧐 Sorry, I couldn't find any matching diseases. Maybe try rephrasing your symptoms?")

        print("💡 Feel free to share more symptoms or type 'exit' to leave.")

if __name__ == "__main__":
    main()

