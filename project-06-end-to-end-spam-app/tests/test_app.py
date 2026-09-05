"""Integration tests for the prediction API."""

from fastapi.testclient import TestClient

from app import app


def test_health_check() -> None:
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_prediction_response_contract() -> None:
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json={"message": "Congratulations! You won a free prize. Call now."},
        )
    body = response.json()
    assert response.status_code == 200
    assert body["label"] in {"ham", "spam"}
    assert 0.0 <= body["spam_probability"] <= 1.0
    assert 0.0 <= body["threshold"] <= 1.0


def test_blank_message_is_rejected() -> None:
    with TestClient(app) as client:
        response = client.post("/predict", json={"message": "   "})
    assert response.status_code == 422


def test_missing_message_is_rejected() -> None:
    with TestClient(app) as client:
        response = client.post("/predict", json={})
    assert response.status_code == 422
