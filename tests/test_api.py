from fastapi.testclient import TestClient

from app.features import RAW_FEATURE_COLUMNS
from app.main import CustomerFeatures, app

client = TestClient(app)

VALID_PAYLOAD = {
    "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
    "tenure": 1, "PhoneService": "No", "MultipleLines": "No phone service",
    "InternetService": "DSL", "OnlineSecurity": "No", "OnlineBackup": "Yes",
    "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No", "StreamingMovies": "No",
    "Contract": "Month-to-month", "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check",
    "MonthlyCharges": 29.85, "TotalCharges": 29.85,
}


def test_health_returns_200():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_predict_valid_input_returns_probability_in_range():
    resp = client.post("/predict", json=VALID_PAYLOAD)
    assert resp.status_code == 200

    body = resp.json()
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["predicted_class"] in ("Yes", "No")
    assert body["model_version"]


def test_predict_missing_field_returns_422():
    incomplete = dict(VALID_PAYLOAD)
    del incomplete["tenure"]

    resp = client.post("/predict", json=incomplete)
    assert resp.status_code == 422
    assert any(err["loc"][-1] == "tenure" for err in resp.json()["detail"])


def test_schema_matches_training_columns():
    # Guards against the API's request schema silently drifting from the
    # columns the model was actually trained on.
    assert set(CustomerFeatures.model_fields.keys()) == set(RAW_FEATURE_COLUMNS)
