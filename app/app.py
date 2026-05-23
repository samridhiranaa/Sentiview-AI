from flask import Flask, request, jsonify, render_template
import pickle
import os
import string
import nltk
import random
import webbrowser
import threading

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

app = Flask(__name__)

# -------------------------------
# PATH SETUP
# -------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")

# -------------------------------
# LOAD MODEL
# -------------------------------
vectorizer_path = os.path.join(MODEL_DIR, "vectorizer.pkl")
with open(vectorizer_path, "rb") as f:
    vectorizer = pickle.load(f)

svm_model_path = os.path.join(MODEL_DIR, "svm_model.pkl")
with open(svm_model_path, "rb") as f:
    model = pickle.load(f)

# -------------------------------
# NLP SETUP
# -------------------------------
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
stop_words.discard("not")
stop_words.discard("very")

lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    words = [w for w in words if w not in stop_words]
    words = [lemmatizer.lemmatize(w) for w in words]
    return " ".join(words)

# -------------------------------
# CHAT INTELLIGENCE
# -------------------------------
greetings = ["hi", "hello", "hey", "hii", "heyy"]
greeting_responses = [
    "Hey there! 😊",
    "Hello! How are you feeling today?",
    "Hi! Tell me about your experience."
]

fallback_responses = [
    "Hmm... I’m not sure I understood that 🤔",
    "Can you tell me more?",
    "Interesting... could you rephrase that?"
]

positive_responses = [
    "That sounds amazing! 😊",
    "Glad you're happy!",
    "Love to hear that!"
]

negative_responses = [
    "That sounds frustrating 😔",
    "Sorry you had that experience.",
    "Hope things improve!"
]

neutral_responses = [
    "Got it 👍",
    "Okay, noted.",
    "Thanks for sharing."
]

# -------------------------------
# ROUTES
# -------------------------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    user_input = data["message"].lower()
    is_follow_up = data.get("is_follow_up", False)

    # Greeting check (Fix: word-based matching)
    words = user_input.split()
    if any(greet in words for greet in greetings):
        return jsonify({
            "response": random.choice(greeting_responses),
            "follow_up": "How can I help you today?"
        })

    # Acknowledge follow-up response
    if is_follow_up:
        return jsonify({
            "response": "I understand! Feel free to reach out if anything else happens. We appreciate the opinion and we constantly work on providing a better experience to our users.",
            "follow_up": None
        })

    # ML prediction
    cleaned = preprocess(user_input)
    vector = vectorizer.transform([cleaned])
    prediction = model.predict(vector)[0]

    if prediction == "positive":
        reply = random.choice(positive_responses)
        follow_up = "What did you like the most?"
    elif prediction == "negative":
        reply = random.choice(negative_responses)
        follow_up = "What went wrong?"
    else:
        reply = random.choice(neutral_responses)
        follow_up = "Can you tell me more?"

    # fallback if very short / unclear
    if len(user_input.split()) <= 1:
        reply = random.choice(fallback_responses)
        follow_up = "Could you please elaborate?"

    return jsonify({
        "response": f"{reply} ({prediction})",
        "follow_up": follow_up
    })

# -------------------------------
# AUTO OPEN BROWSER
# -------------------------------
def open_browser():
    webbrowser.open_new("http://127.0.0.1:5000/")

# -------------------------------
# RUN
# -------------------------------
if __name__ == "__main__":
    threading.Timer(1, open_browser).start()
    app.run(debug=True)