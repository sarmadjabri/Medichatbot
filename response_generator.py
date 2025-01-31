import random

def generate_response(intent):
    if intent == 'symptom_check':
        return "I'll check the symptoms for you."

    if intent == 'greeting':
        return random.choice(["Hello! How can I assist you?", "Hi there! What's bothering you today?"])

    if intent == 'goodbye':
        return "Goodbye! Take care!"

    if intent == 'feedback':
        return "Thank you for your feedback!"

    return "Sorry, I didn't understand that. Could you clarify?"
