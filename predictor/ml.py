import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from deep_translator import GoogleTranslator
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "languages_datasets.csv"
MODEL_DIR = BASE_DIR / "artifacts"
MODEL_PATH = MODEL_DIR / "language_model.pkl"
METRICS_PATH = MODEL_DIR / "model_metrics.json"
RANDOM_STATE = 42
TEST_SIZE = 0.33


def _prepare_training_data():
    data = pd.read_csv(DATASET_PATH).dropna(subset=["Text", "language"])
    texts = np.array(data["Text"])
    labels = np.array(data["language"])
    return train_test_split(
        texts,
        labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels,
    )


def train_and_save_model():
    x_train, x_test, y_train, y_test = _prepare_training_data()

    vectorizer = CountVectorizer()
    x_train_vectorized = vectorizer.fit_transform(x_train)
    x_test_vectorized = vectorizer.transform(x_test)

    model = MultinomialNB()
    model.fit(x_train_vectorized, y_train)

    predictions = model.predict(x_test_vectorized)
    accuracy = accuracy_score(y_test, predictions)

    artifact = {
        "model": model,
        "vectorizer": vectorizer,
        "accuracy": float(accuracy),
        "train_size": int(len(x_train)),
        "test_size": int(len(x_test)),
        "label_count": int(len(np.unique(y_train))),
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, MODEL_PATH)

    METRICS_PATH.write_text(
        json.dumps(
            {
                "accuracy": round(float(accuracy), 6),
                "accuracy_percent": round(float(accuracy) * 100, 2),
                "train_size": artifact["train_size"],
                "test_size": artifact["test_size"],
                "label_count": artifact["label_count"],
                "model_path": str(MODEL_PATH.name),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return artifact


def load_model_artifact():
    if not MODEL_PATH.exists():
        return train_and_save_model()
    return joblib.load(MODEL_PATH)


def predict_language(text):
    artifact = load_model_artifact()
    transformed_input = artifact["vectorizer"].transform([text])
    return artifact["model"].predict(transformed_input)[0]


def translate_text(text, target_lang):
    try:
        translator = GoogleTranslator(source="auto", target=target_lang)
        return translator.translate(text)
    except Exception:
        return text


def get_model_metrics():
    artifact = load_model_artifact()
    metrics = {
        "accuracy": round(float(artifact["accuracy"]), 6),
        "accuracy_percent": round(float(artifact["accuracy"]) * 100, 2),
        "train_size": artifact["train_size"],
        "test_size": artifact["test_size"],
        "label_count": artifact["label_count"],
        "model_path": str(MODEL_PATH),
    }

    if not METRICS_PATH.exists():
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    return metrics
