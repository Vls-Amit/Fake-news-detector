# fake_news_detector.py
# ------------------------------------------------------------
# Fake News Detection System (exactly as per the question):
# - Clean text with NLTK stopwords
# - Vectorize with TF-IDF
# - Classify with Logistic Regression
# - Report Accuracy, Confusion Matrix, Classification Report
# - Predict on new (example) inputs
# ------------------------------------------------------------

import os
import re
import pandas as pd
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# --------------- setup --------------- #
nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords

STOP_WORDS = set(stopwords.words("english"))
# --------------- text cleaning --------------- #
def clean_text(text: str) -> str:
    """Lowercase, remove punctuation/digits, strip stopwords."""
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text) 
    words = [w for w in text.split() if w not in STOP_WORDS]
    return " ".join(words)
# --------------- data loader --------------- #
def load_dataset():
    """
    Tries to load a local CSV named 'news.csv' with columns:
      - 'text'  : the article/headline/content
      - 'label' : 0 for REAL, 1 for FAKE
    If not found, falls back to a small built-in dataset so the script runs out-of-the-box.
    """
    csv_path = "news.csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        if not {"text", "label"}.issubset(df.columns):
            raise ValueError("CSV must have columns: 'text' and 'label' (0=Real, 1=Fake).")
        df = df[["text", "label"]].dropna()
        df["label"] = df["label"].astype(int)
        return df.reset_index(drop=True)
    # Fallback demo dataset (balanced, just for pipeline correctness)
    data = {
        "text": [
            "Government announces new tax reforms for IT industry",
            "Scientists report promising results for new cancer therapy",
            "Central bank raises interest rates to curb inflation",
            "Local authorities open new public park for residents",
            "NASA confirms successful test of next-generation rocket",
            "Shocking! Secret pill guarantees instant weight loss overnight",
            "BREAKING: Aliens land in New York and meet the mayor",
            "Celebrity claims to time travel using hidden device",
            "You won the lottery! Click this link to claim your prize now",
            "Miracle drink cures all diseases according to anonymous source",
        ],
        # 0 = Real, 1 = Fake
        "label": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
    }
    return pd.DataFrame(data)

# --------------- training & evaluation --------------- #
def train_and_evaluate(df: pd.DataFrame):
    df = df.copy()
    df["clean_text"] = df["text"].apply(clean_text)

    classes = df["label"].unique()
    if len(classes) < 2:
        raise ValueError(
            f"Need at least 2 classes (0 and 1). Found only: {classes}. "
            "Check your CSV labels."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"],
        df["label"],
        test_size=0.2,
        random_state=42,
        stratify=df["label"],
    )

    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_tfidf, y_train)

    y_pred = model.predict(X_test_tfidf)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=4)

    print("========== Fake News Detection (TF-IDF + Logistic Regression) ==========")
    print(f"Accuracy: {acc:.4f}\n")
    print("Confusion Matrix:")
    print(cm, "\n")
    print("Classification Report:")
    print(report)

    return model, vectorizer

def demo_predictions(model, vectorizer):
    samples = [
        "Government announces scholarship program for STEM students",
        "Exclusive! Drink this and lose 20 kg in a week",
        "Tech giants agree to new privacy standards for user data",
        "Scientists confirm aliens disguised as world leaders",
    ]
    print("Sample Predictions:")
    for t in samples:
        cleaned = clean_text(t)
        X = vectorizer.transform([cleaned])
        pred = model.predict(X)[0]
        label = "FAKE" if pred == 1 else "REAL"
        print(f" - {t}  -->  {label}")


# --------------- main --------------- #
if __name__ == "__main__":
    df = load_dataset()
    model, vectorizer = train_and_evaluate(df)
    demo_predictions(model, vectorizer)

