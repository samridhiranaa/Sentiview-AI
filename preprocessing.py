import pandas as pd
import nltk
import string
import pickle

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('stopwords')
nltk.download('wordnet')

df = pd.read_csv("product_reviews_dataset.csv")

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

df.to_csv("cleaned_reviews.csv", index=False)

with open("X_tfidf.pkl", "wb") as f:
    pickle.dump(X_tfidf, f)

with open("y.pkl", "wb") as f:
    pickle.dump(y, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("\nAll files saved successfully!")