import pickle
import string
from pathlib import Path

import nltk
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "dataset.csv"
VECTORIZER_PATH = BASE_DIR / "tfidf_vectorizer.pkl"


def preprocess_text(text: str) -> str:
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = text.lower()
    stop_words = set(stopwords.words("english"))
    return " ".join(word for word in text.split() if word not in stop_words)


def main() -> None:
    nltk.download("stopwords", quiet=True)
    data = pd.read_csv(DATASET_PATH)
    data["source_text"] = data["source_text"].astype(str).apply(preprocess_text)
    data["plagiarized_text"] = data["plagiarized_text"].astype(str).apply(preprocess_text)

    vectorizer = TfidfVectorizer()
    vectorizer.fit(data["source_text"] + " " + data["plagiarized_text"])

    with VECTORIZER_PATH.open("wb") as f:
        pickle.dump(vectorizer, f)

    print(f"Saved: {VECTORIZER_PATH}")


if __name__ == "__main__":
    main()
