# 🧠 Product Review Sentiment Analysis using NLP

> A complete end-to-end Natural Language Processing (NLP) project that classifies product reviews into **Positive, Negative, and Neutral** sentiments using Machine Learning.

---

## 📌 Overview

With the rapid growth of e-commerce platforms, analyzing customer feedback manually has become impractical. This project automates sentiment classification of product reviews using NLP techniques and machine learning models.

The system implements a full pipeline including:
- Text preprocessing  
- Feature extraction using TF-IDF  
- Model training and evaluation  
- Real-time prediction on unseen user input  

---

## 🚀 Key Features

- 🔹 Text preprocessing (cleaning & normalization)
- 🔹 TF-IDF feature extraction
- 🔹 Multiple ML models:
  - Naïve Bayes  
  - Logistic Regression  
  - Support Vector Machine (SVM)  
- 🔹 Model evaluation using:
  - Accuracy  
  - Precision  
  - Recall  
  - F1 Score  
- 🔹 Data visualization:
  - Accuracy comparison graph  
  - Confusion matrix heatmap  
  - Sentiment distribution chart  
- 🔹 Real-time sentiment prediction system (`test.py`)

---

## 🧠 Tech Stack

- **Language:** Python  
- **Libraries:** NLTK, Scikit-learn, Pandas, Matplotlib  

---

## ⚙️ Installation & Setup

### 1. Clone the repository
git clone https://github.com/your-username/Sentiment-Analysis-NLP.git
cd Sentiment-Analysis-NLP

### 2. Install dependencies
pip install -r requitrements.txt


---

## ▶️ How to Run

### Step 1: Preprocess data
python preprocessing.py

### Step 2: Train model
python train.py

### Step 3: Test on new input
python test.py

---

## 📊 Results

| Model | Accuracy |
|------|--------|
| Naïve Bayes | 1.00 |
| Logistic Regression | 1.00 |
| SVM | 1.00 |

- Precision = 1.00  
- Recall = 1.00  
- F1 Score = 1.00  

---

## 📈 Visualizations
- Model Accuracy Comparison Graph  
- Confusion Matrix Heatmap  
- Sentiment Distribution Chart  

(Add screenshots here)

---

## 💻 Real-Time Prediction

Example:

Input: This product is amazing
Output: Positive

Input: Very bad experience
Output: Negative

Input: It's okay, nothing special
Output: Neutral


---

## ⚠️ Limitations
- Small dataset (30 samples)
- Model evaluated on full dataset (possible overfitting)
- May not generalize well to real-world data
- Cannot detect sarcasm or complex sentiment

---

## 🔮 Future Improvements
- Use larger and more diverse datasets
- Apply deep learning models (LSTM, BERT)
- Deploy as a web or mobile application

