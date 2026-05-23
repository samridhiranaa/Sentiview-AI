import pandas as pd
import nltk
import string
import pickle
import os

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('stopwords')
nltk.download('wordnet')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

raw_reviews_path = os.path.join(DATA_DIR, "raw_reviews.csv")
df = pd.read_csv(raw_reviews_path)

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

df["Cleaned_Review"] = df["Review"].apply(preprocess)

print("\nSample Data:\n")
print(df[["Review", "Cleaned_Review"]].head())

vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(df["Cleaned_Review"])
y = df["Sentiment"]

print("\nTF-IDF Shape:", X_tfidf.shape)

cleaned_reviews_path = os.path.join(DATA_DIR, "cleaned_reviews.csv")
df.to_csv(cleaned_reviews_path, index=False)

x_tfidf_path = os.path.join(MODEL_DIR, "X_tfidf.pkl")
with open(x_tfidf_path, "wb") as f:
    pickle.dump(X_tfidf, f)

y_path = os.path.join(MODEL_DIR, "y.pkl")
with open(y_path, "wb") as f:
    pickle.dump(y, f)

vectorizer_path = os.path.join(MODEL_DIR, "vectorizer.pkl")
with open(vectorizer_path, "wb") as f:
    pickle.dump(vectorizer, f)

print("\nAll files saved successfully!")