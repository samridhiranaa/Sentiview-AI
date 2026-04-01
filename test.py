import pickle
import string
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("X_tfidf.pkl", "rb") as f:
    pass  

with open("svm_model.pkl", "rb") as f:
    svm_model = pickle.load(f)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    if not isinstance(text, str):
        return ""
    
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    words = text.split()
    words = [w for w in words if w not in stop_words]
    words = [lemmatizer.lemmatize(w) for w in words]
    
    return " ".join(words)

def predict_review(review):
    cleaned = preprocess(review)
    vector = vectorizer.transform([cleaned])
    prediction = svm_model.predict(vector)
    return prediction[0]

print("\n===== SENTIMENT ANALYZER =====\n")

while True:
    user_input = input("Enter a review (or type 'exit' to quit): ")
    
    if user_input.lower() == "exit":
        print("Exiting...")
        break
    
    result = predict_review(user_input)
    print("Predicted Sentiment:", result)
    print("-" * 40)