import logging

# Set up logging configuration
logging.basicConfig(filename="chatbot.log", level=logging.DEBUG, format='%(asctime)s - %(message)s')

def log_event(message):
    logging.info(message)

def log_error(error_message):
    logging.error(error_message)

def log_interaction(user_input, bot_response):
    log_event(f"User: {user_input}, Bot: {bot_response}")
