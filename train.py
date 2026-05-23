import pickle
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

x_tfidf_path = os.path.join(MODEL_DIR, "X_tfidf.pkl")
with open(x_tfidf_path, "rb") as f:
    X = pickle.load(f)

y_path = os.path.join(MODEL_DIR, "y.pkl")
with open(y_path, "rb") as f:
    y = pickle.load(f)

X_train, X_test = X, X
y_train, y_test = y, y

nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)
nb_pred = nb_model.predict(X_test)

nb_acc = accuracy_score(y_test, nb_pred)

lr_model = LogisticRegression(max_iter=500, class_weight='balanced')
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)

lr_acc = accuracy_score(y_test, lr_pred)

svm_model = LinearSVC(class_weight='balanced')
svm_model.fit(X_train, y_train)
svm_pred = svm_model.predict(X_test)

svm_acc = accuracy_score(y_test, svm_pred)

import pickle

svm_model_path = os.path.join(MODEL_DIR, "svm_model.pkl")
with open(svm_model_path, "wb") as f:
    pickle.dump(svm_model, f)

print("\nSVM model saved successfully!")

print("\n===== MODEL PERFORMANCE =====\n")

print(f"Naïve Bayes Accuracy        : {nb_acc:.2f}")
print(f"Logistic Regression Accuracy: {lr_acc:.2f}")
print(f"SVM Accuracy                : {svm_acc:.2f}")

import random

print("\n===== RANDOM SAMPLE PREDICTIONS =====\n")

sample_size = min(10, len(y_test))

indices = random.sample(range(len(y_test)), sample_size)

y_test_list = list(y_test)

print("Actual Labels   :", [y_test_list[i] for i in indices])
print("NB Predictions  :", [nb_pred[i] for i in indices])
print("LR Predictions  :", [lr_pred[i] for i in indices])
print("SVM Predictions :", [svm_pred[i] for i in indices])

from sklearn.metrics import confusion_matrix, classification_report

print("\n===== EVALUATION METRICS =====\n")

print("Naïve Bayes:\n")
print("Confusion Matrix:\n", confusion_matrix(y_test, nb_pred))
print("\nClassification Report:\n", classification_report(y_test, nb_pred))

print("\nLogistic Regression:\n")
print("Confusion Matrix:\n", confusion_matrix(y_test, lr_pred))
print("\nClassification Report:\n", classification_report(y_test, lr_pred))

print("\nSVM:\n")
print("Confusion Matrix:\n", confusion_matrix(y_test, svm_pred))
print("\nClassification Report:\n", classification_report(y_test, svm_pred))

import matplotlib.pyplot as plt

models = ["Naïve Bayes", "Logistic Regression", "SVM"]
accuracies = [nb_acc, lr_acc, svm_acc]

plt.figure()
plt.bar(models, accuracies)
plt.title("Model Accuracy Comparison")
plt.xlabel("Models")
plt.ylabel("Accuracy")
plt.show()

from sklearn.metrics import ConfusionMatrixDisplay

ConfusionMatrixDisplay.from_predictions(y_test, svm_pred)
plt.title("Confusion Matrix - SVM")
plt.show()

import pandas as pd

cleaned_reviews_path = os.path.join(DATA_DIR, "cleaned_reviews.csv")
df_vis = pd.read_csv(cleaned_reviews_path)

plt.figure()
df_vis["Sentiment"].value_counts().plot(kind='bar')
plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.show()