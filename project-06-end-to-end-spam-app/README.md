# Project 06: End-to-End Spam Detection App

Turn the NLP model from Project 4 into a usable application. This project covers
the complete path from raw data to a browser interface and prediction API.

## What you will learn

- Train and version a reproducible model artifact
- Load a model once when an API starts
- Validate prediction requests and return structured JSON
- Connect a browser interface to an ML API
- Separate training code from inference code
- Test health checks, validation, and predictions
- Package an application in Docker
- Understand deployment concerns such as thresholds and model monitoring

## Architecture

```text
Browser form
    |
    | POST /predict
    v
FastAPI application
    |
    | TF-IDF transform + Naive Bayes probability
    v
Saved model pipeline
```

## Structure

```text
project-06-end-to-end-spam-app/
|-- data/
|-- models/
|-- static/
|-- templates/
|-- tests/
|-- app.py
|-- train.py
|-- Dockerfile
|-- .dockerignore
|-- .gitignore
|-- requirements.txt
`-- README.md
```

## Setup

```powershell
cd "F:\Machine Learning\project-06-end-to-end-spam-app"
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Stage 1: Train the production artifact

```powershell
python train.py
```

This downloads the UCI SMS Spam Collection on the first run, removes duplicate
messages, evaluates the final Project 4 configuration, and saves
`models/spam_pipeline.joblib` with metadata and a decision threshold.

## Stage 2: Run the application

```powershell
uvicorn app:app --reload
```

Open these local pages:

- App: <http://127.0.0.1:8000>
- Interactive API docs: <http://127.0.0.1:8000/docs>
- Health check: <http://127.0.0.1:8000/health>

Stop the server with `Ctrl+C`.

## API example

```powershell
$body = @{ message = "Congratulations! You won a free prize. Call now." } |
    ConvertTo-Json
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/predict" `
    -ContentType "application/json" -Body $body
```

Example response:

```json
{
  "label": "spam",
  "spam_probability": 0.91,
  "threshold": 0.5
}
```

## Stage 3: Run automated tests

Train the model first, then run:

```powershell
pytest -q
```

## Stage 4: Build with Docker (optional)

```powershell
docker build -t spam-detector .
docker run --rm -p 8000:8000 spam-detector
```

## Production questions

1. Why should the model load once at startup instead of on every request?
2. Why is the decision threshold stored with the model metadata?
3. What should happen if the model artifact is missing?
4. Which input and prediction statistics should be monitored over time?
5. Why should retraining and deployment be separate controlled processes?

