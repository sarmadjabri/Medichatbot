from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import json

# Load intents data
with open('training_data.json', 'r') as file:
    intents_data = json.load(file)

# Training data for intent classification
training_sentences = []
training_labels = []

for intent in intents_data['intents']:
    for pattern in intent['patterns']:
        training_sentences.append(pattern)
        training_labels.append(intent['tag'])

# Vectorize sentences and train classifier
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(training_sentences)

classifier = MultinomialNB()
classifier.fit(X_train, training_labels)

def classify_intent(text):
    text_features = vectorizer.transform([text])
    prediction = classifier.predict(text_features)[0]
    return prediction
