import sqlite3

def create_db():
    conn = sqlite3.connect("chatbot_data.db")
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS interactions 
                 (id INTEGER PRIMARY KEY, 
                  user_input TEXT, 
                  bot_response TEXT, 
                  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    c.execute('''CREATE TABLE IF NOT EXISTS feedback 
                 (id INTEGER PRIMARY KEY, 
                  interaction_id INTEGER, 
                  feedback TEXT, 
                  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                  FOREIGN KEY (interaction_id) REFERENCES interactions(id))''')
    conn.commit()
    conn.close()

def save_interaction(user_input, bot_response):
    conn = sqlite3.connect("chatbot_data.db")
    c = conn.cursor()
    c.execute("INSERT INTO interactions (user_input, bot_response) VALUES (?, ?)", 
              (user_input, bot_response))
    conn.commit()
    conn.close()

def save_feedback(interaction_id, feedback):
    conn = sqlite3.connect("chatbot_data.db")
    c = conn.cursor()
    c.execute("INSERT INTO feedback (interaction_id, feedback) VALUES (?, ?)", 
              (interaction_id, feedback))
    conn.commit()
    conn.close()

def fetch_interactions():
    conn = sqlite3.connect("chatbot_data.db")
    c = conn.cursor()
    c.execute("SELECT * FROM interactions")
    rows = c.fetchall()
    conn.close()
    return rows
