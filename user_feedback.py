def collect_feedback(interaction_id, feedback):
    from database import save_feedback
    save_feedback(interaction_id, feedback)

def ask_for_feedback():
    feedback = input("Was this helpful? (yes/no): ")
    collect_feedback(feedback)
