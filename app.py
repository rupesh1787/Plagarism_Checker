from __future__ import annotations

import os
import pickle
import shutil
from pathlib import Path
from typing import Any
from io import BytesIO

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from PyPDF2 import PdfReader
import pdfplumber
import pypdfium2 as pdfium
import pytesseract

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


def get_tesseract_executable() -> str:
    """Resolve Tesseract executable path from PATH or common Windows location."""
    tesseract_bin = shutil.which("tesseract")
    if tesseract_bin:
        return tesseract_bin

    windows_default = Path("C:/Program Files/Tesseract-OCR/tesseract.exe")
    if windows_default.exists():
        return str(windows_default)

    return ""


def run_prediction(text1: str, text2: str) -> str:
    """Run model prediction using the exact training-time text combination."""
    combined = text1 + " " + text2
    features = vectorizer.transform([combined])
    prediction = model.predict(features)[0]
    return "Plagiarism Detected" if int(prediction) == 1 else "No Plagiarism"


def validate_pdf_file(uploaded_file: Any, field_name: str) -> str | None:
    """Validate uploaded file existence, name, extension, and basic content."""
    if uploaded_file is None or uploaded_file.filename is None:
        return f"Missing required file: {field_name}."

    filename = uploaded_file.filename.strip()
    if not filename:
        return f"Missing required file: {field_name}."

    if not filename.lower().endswith(".pdf"):
        return f"{field_name} must be a PDF file."

    header = uploaded_file.stream.read(5)
    uploaded_file.stream.seek(0)
    if not header:
        return f"{field_name} is empty."

    if header != b"%PDF-":
        return f"{field_name} is not a valid PDF file."

    return None


def extract_pdf_text_with_ocr(data: bytes) -> str:
    """Run OCR on PDF pages for scanned/image-only documents."""
    tesseract_bin = get_tesseract_executable()
    if not tesseract_bin:
        return ""

    pytesseract.pytesseract.tesseract_cmd = tesseract_bin

    ocr_parts: list[str] = []
    try:
        pdf = pdfium.PdfDocument(data)
        for page_index in range(len(pdf)):
            page = pdf[page_index]
            image = page.render(scale=2.0).to_pil()
            page_text = pytesseract.image_to_string(image) or ""
            ocr_parts.append(page_text)
            page.close()
        pdf.close()
    except Exception:
        return ""

    return "\n".join(ocr_parts).strip()


def extract_pdf_text(uploaded_file: Any, field_name: str) -> str:
    """Extract full text from a PDF upload."""
    try:
        data = uploaded_file.read()
        uploaded_file.seek(0)

        if not data:
            raise ValueError(f"{field_name} is empty.")

        # First attempt with PyPDF2.
        reader = PdfReader(BytesIO(data))
        pypdf_parts: list[str] = []
        for page in reader.pages:
            page_text = page.extract_text() or ""
            pypdf_parts.append(page_text)

        text = "\n".join(pypdf_parts).strip()

        # Fallback attempt with pdfplumber for PDFs that parse better there.
        if not text:
            plumber_parts: list[str] = []
            with pdfplumber.open(BytesIO(data)) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text() or ""
                    plumber_parts.append(page_text)
            text = "\n".join(plumber_parts).strip()

        # Last fallback: OCR for scanned/image-only PDFs.
        if not text:
            text = extract_pdf_text_with_ocr(data)

        if not text:
            ocr_hint = ""
            if not get_tesseract_executable():
                ocr_hint = " Install Tesseract OCR and add it to PATH to enable scanned PDF support."
            raise ValueError(
                f"No extractable text found in {field_name}. The PDF may be scanned/image-only. "
                "Please upload a searchable PDF or run OCR before uploading."
                + ocr_hint
            )

        return text

    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"Failed to extract text from {field_name}.") from exc


@app.get("/")
def home() -> Any:
    """Serve the frontend page."""
    return send_from_directory(BASE_DIR, "index.html")


@app.get("/health")
def health() -> Any:
    """Simple health check endpoint for hosting platforms."""
    return jsonify({"status": "ok"})


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

        result = run_prediction(text1, text2)
        return jsonify({"result": result})

    except Exception as exc:
        app.logger.exception("Prediction error: %s", exc)
        return jsonify({"error": "An internal server error occurred."}), 500


@app.post("/upload")
def upload_pdf() -> Any:
    """Predict plagiarism from two uploaded PDF documents."""
    try:
        file1 = request.files.get("file1")
        file2 = request.files.get("file2")

        validation_error = validate_pdf_file(file1, "file1")
        if validation_error:
            return jsonify({"error": validation_error}), 400

        validation_error = validate_pdf_file(file2, "file2")
        if validation_error:
            return jsonify({"error": validation_error}), 400

        text1 = extract_pdf_text(file1, "file1")
        text2 = extract_pdf_text(file2, "file2")

        result = run_prediction(text1, text2)
        return jsonify(
            {
                "result": result,
                "length1": len(text1.split()),
                "length2": len(text2.split()),
            }
        )

    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        app.logger.exception("PDF upload prediction error: %s", exc)
        return jsonify({"error": "An internal server error occurred."}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    debug = os.environ.get("FLASK_DEBUG", "").lower() in {"1", "true", "yes"}
    app.run(host="0.0.0.0", port=port, debug=debug)
