from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"
VECTORIZER_PATH = BASE_DIR / "tfidf_vectorizer.pkl"


def load_pickle_file(file_path: Path) -> Any:
    """Load and return a pickle file with a clear error if missing/corrupt."""
    if not file_path.exists():
        raise FileNotFoundError(f"Required file not found: {file_path}")

    with file_path.open("rb") as f:
        return pickle.load(f)


# Load artifacts once at startup for fast inference.
model = load_pickle_file(MODEL_PATH)
vectorizer = load_pickle_file(VECTORIZER_PATH)

app = Flask(__name__, static_folder=".")
CORS(app)


@app.get("/")
def home() -> Any:
    """Serve the frontend page."""
    return send_from_directory(BASE_DIR, "index.html")


@app.post("/predict")
def predict() -> Any:
    """Predict plagiarism from two text inputs."""
    try:
        payload = request.get_json(silent=True)
        if payload is None:
            return jsonify({"error": "Request body must be valid JSON."}), 400

        text1 = str(payload.get("text1", "")).strip()
        text2 = str(payload.get("text2", "")).strip()

        if not text1 or not text2:
            return jsonify({"error": "Both text1 and text2 are required and cannot be empty."}), 400

        # Keep feature construction identical to model training.
        combined = text1 + " " + text2
        features = vectorizer.transform([combined])
        prediction = model.predict(features)[0]

        result = "Plagiarism Detected" if int(prediction) == 1 else "No Plagiarism"
        return jsonify({"result": result})

    except Exception as exc:
        app.logger.exception("Prediction error: %s", exc)
        return jsonify({"error": "An internal server error occurred."}), 500


if __name__ == "__main__":
    app.run(debug=True)
