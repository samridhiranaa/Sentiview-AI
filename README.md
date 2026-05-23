# Sentiview AI

### AI-Powered Sentiment Intelligence Platform

Sentiview AI is an NLP and machine learning-powered platform designed to analyze customer reviews, classify sentiment, and generate meaningful insights from textual feedback in real time.

Built using Python, Flask, and Scikit-learn, the platform combines natural language processing techniques with machine learning models to transform raw customer opinions into actionable sentiment analytics.

---

# Features

- Real-time sentiment prediction
- NLP preprocessing pipeline
- TF-IDF feature extraction
- Machine learning-based sentiment classification
- Interactive Flask web application
- Sentiment confidence scoring
- Review analytics and visualization
- Modular and scalable architecture

---

# Tech Stack

## Languages & Frameworks
- Python
- Flask

## Machine Learning & NLP
- Scikit-learn
- NLTK
- TF-IDF Vectorization

## Data Processing & Visualization
- Pandas
- NumPy
- Matplotlib

---

# Project Structure

```text
Sentiview-AI/
│
├── app/
│   ├── templates/
│   │   └── index.html
│   │
│   ├── static/
│   │   ├── css/
│   │   │   └── style.css
│   │   │
│   │   ├── js/
│   │   │   └── script.js
│   │   │
│   │   └── images/
│   │
│   └── app.py
│
├── models/
│   ├── svm_model.pkl
│   └── vectorizer.pkl
│
├── data/
│   ├── raw_reviews.csv
│   └── cleaned_reviews.csv
│
├── notebooks/
│
├── preprocessing.py
├── train.py
├── requirements.txt
├── README.md
├── .gitignore
└── LICENSE
```

---

# Machine Learning Pipeline

The project follows a complete NLP workflow:

1. Text preprocessing and cleaning
2. Tokenization and normalization
3. TF-IDF feature extraction
4. Model training and evaluation
5. Real-time sentiment prediction
6. Sentiment analytics visualization

---

# Installation & Setup

## 1. Clone the Repository

```bash
git clone https://github.com/your-username/Sentiview-AI.git
cd Sentiview-AI
```

---

## 2. Create Virtual Environment (Optional)

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Mac/Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Running the Project

## Step 1 — Preprocess Dataset

```bash
python preprocessing.py
```

## Step 2 — Train the Model

```bash
python train.py
```

## Step 3 — Run the Flask Application

```bash
python app/app.py
```

---

# Current Capabilities

The platform currently supports:

- Positive sentiment detection
- Negative sentiment detection
- Neutral sentiment detection
- Real-time review analysis
- NLP preprocessing
- Machine learning-based classification
- Basic analytics and visualization

---

# Planned Features

## UI & Dashboard Enhancements

- Modern AI dashboard interface
- Dark glassmorphism UI
- Interactive charts and graphs
- Prediction history panel
- Animated sentiment indicators

## Machine Learning Improvements

- Larger and more diverse datasets
- Realistic train/test evaluation
- Multiple model benchmarking
- Deep learning integration
- BERT/LSTM implementation

## Platform Features

- CSV bulk review analysis
- API support
- Cloud deployment
- Exportable analytics reports
- User authentication system

---

# Future Vision

Sentiview AI aims to evolve into a complete customer feedback intelligence platform capable of helping businesses understand customer behavior, emotional trends, and product perception through AI-driven sentiment analytics.

---

# License

This project is licensed under the MIT License.