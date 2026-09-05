"""FastAPI application serving spam predictions and a browser interface."""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import joblib
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "models" / "spam_pipeline.joblib"
templates = Jinja2Templates(directory=PROJECT_DIR / "templates")
model_artifact: dict = {}


class PredictionRequest(BaseModel):
    message: Annotated[str, Field(min_length=1, max_length=5_000)]


class PredictionResponse(BaseModel):
    label: str
    spam_probability: float
    threshold: float
    model_version: str


@asynccontextmanager
async def lifespan(_: FastAPI):
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model missing at {MODEL_PATH}. Run: python train.py")
    model_artifact.update(joblib.load(MODEL_PATH))
    yield
    model_artifact.clear()


app = FastAPI(
    title="SMS Spam Detector",
    version="1.0.0",
    description="Classify an SMS message using TF-IDF and Naive Bayes.",
    lifespan=lifespan,
)
app.mount("/static", StaticFiles(directory=PROJECT_DIR / "static"), name="static")


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "healthy",
        "model_version": model_artifact["model_version"],
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest) -> PredictionResponse:
    message = payload.message.strip()
    if not message:
        # Whitespace passed initial length validation but is not useful input.
        from fastapi import HTTPException

        raise HTTPException(status_code=422, detail="Message cannot be blank.")

    probability = float(model_artifact["pipeline"].predict_proba([message])[0, 1])
    threshold = float(model_artifact["threshold"])
    label = "spam" if probability >= threshold else "ham"
    return PredictionResponse(
        label=label,
        spam_probability=round(probability, 6),
        threshold=threshold,
        model_version=model_artifact["model_version"],
    )

