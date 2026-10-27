import os

import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.skipif(
    not os.path.exists("model"),
    reason="model/ folder not found - run `python src/train.py` first",
)

HIGH_RISK = {
    "Pregnancies": 6, "Glucose": 148, "BloodPressure": 72, "SkinThickness": 35,
    "Insulin": 0, "BMI": 33.6, "DiabetesPedigreeFunction": 0.627, "Age": 50,
}
LOW_RISK = {
    "Pregnancies": 1, "Glucose": 85, "BloodPressure": 66, "SkinThickness": 29,
    "Insulin": 0, "BMI": 26.6, "DiabetesPedigreeFunction": 0.351, "Age": 31,
}


@pytest.fixture(scope="module")
def client():
    from src.app import app  # imported here so the skip above applies first

    return TestClient(app)


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_predict_returns_valid_response(client):
    response = client.post("/predict", json=HIGH_RISK)
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"prediction", "probability", "status"}
    assert body["prediction"] in (0, 1)
    assert 0.0 <= body["probability"] <= 1.0
    assert body["status"] == ("diabetic" if body["prediction"] == 1 else "healthy")


def test_high_risk_scores_higher_than_low_risk(client):
    high = client.post("/predict", json=HIGH_RISK).json()["probability"]
    low = client.post("/predict", json=LOW_RISK).json()["probability"]
    assert high > low


def test_predict_rejects_missing_field(client):
    incomplete = {k: v for k, v in HIGH_RISK.items() if k != "Glucose"}
    response = client.post("/predict", json=incomplete)
    assert response.status_code == 422


def test_predict_rejects_non_numeric_value(client):
    bad = {**HIGH_RISK, "Age": "fifty"}
    response = client.post("/predict", json=bad)
    assert response.status_code == 422