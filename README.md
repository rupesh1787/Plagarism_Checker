# Plagiarism Checker

Flask-based plagiarism checker with two modes:
- Text vs text comparison
- PDF vs PDF comparison (with OCR fallback for scanned PDFs)

## Project Files

- app.py: Flask backend and inference API
- index.html: Single-page frontend UI
- model.pkl: Trained classifier artifact
- tfidf_vectorizer.pkl: Trained vectorizer artifact (required)
- requirements.txt: Python dependencies for local and Render deploy
- render.yaml: Render blueprint config

## Important Requirement

Before running locally or deploying, make sure tfidf_vectorizer.pkl exists in the project root.

The app loads both artifacts during startup. If tfidf_vectorizer.pkl is missing, startup will fail.

## Local Run

1. Create and activate a virtual environment

Windows cmd:

python -m venv .venv
.venv\Scripts\activate

2. Install dependencies

pip install -r requirements.txt

3. Start app

python app.py

4. Open browser

http://127.0.0.1:5000

## Render Deployment

This repo includes render.yaml, so deployment can be done with Blueprint or manual web service setup.

### Option A: Blueprint Deploy (recommended)

1. Push this repository to GitHub.
2. In Render dashboard, choose New + Blueprint.
3. Select this GitHub repo.
4. Render reads render.yaml and creates the service.
5. Deploy.

### Option B: Manual Web Service

1. In Render, choose New + Web Service.
2. Connect this GitHub repo.
3. Configure:
- Runtime: Python
- Build command: pip install -r requirements.txt
- Start command: gunicorn app:app
4. Deploy.

## Endpoints

- GET / : Frontend page
- GET /health : Health check
- POST /predict : JSON text plagiarism check
- POST /upload : Multipart PDF plagiarism check

## OCR Note

For scanned/image-only PDFs, OCR depends on Tesseract availability.

- Local Windows: install Tesseract and ensure it is in PATH
- Render/Linux: if OCR is needed in production, prefer Docker-based deployment so system-level Tesseract can be installed