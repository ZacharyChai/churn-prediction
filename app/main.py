import logging
import time
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, Request
from pydantic import BaseModel, ConfigDict, Field

from app.features import RAW_FEATURE_COLUMNS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("churn-api")

MODEL_PATH = Path(__file__).resolve().parent.parent / "model.joblib"
_bundle = joblib.load(MODEL_PATH)
model = _bundle["pipeline"]
MODEL_VERSION = _bundle["version"]

# Docs (Swagger UI) served at "/" instead of the FastAPI default "/docs".
app = FastAPI(title="Churn Prediction API", docs_url="/")


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.monotonic()
    response = await call_next(request)
    duration_ms = (time.monotonic() - start) * 1000
    logger.info(
        "%s %s -> %d (%.1fms)",
        request.method, request.url.path, response.status_code, duration_ms,
    )
    return response


class CustomerFeatures(BaseModel):
    gender: Literal["Male", "Female"]
    SeniorCitizen: Literal[0, 1]
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    tenure: int = Field(ge=0)
    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal[
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)",
    ]
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 1,
                "PhoneService": "No",
                "MultipleLines": "No phone service",
                "InternetService": "DSL",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 29.85,
                "TotalCharges": 29.85,
            }
        }
    )


# CustomerFeatures must cover exactly the columns the pipeline was trained
# on — no more, no less — or the API and the model would silently drift
# apart. Fail fast at import time rather than on the first bad prediction.
_schema_fields = set(CustomerFeatures.model_fields.keys())
_training_fields = set(RAW_FEATURE_COLUMNS)
assert _schema_fields == _training_fields, (
    f"CustomerFeatures has drifted from the training schema: "
    f"missing={_training_fields - _schema_fields} extra={_schema_fields - _training_fields}"
)


class ChurnPrediction(BaseModel):
    churn_probability: float
    predicted_class: Literal["Yes", "No"]
    model_version: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=ChurnPrediction)
def predict(customer: CustomerFeatures):
    row = pd.DataFrame([customer.model_dump()])
    probability = float(model.predict_proba(row)[0, 1])
    predicted_class = "Yes" if probability >= 0.5 else "No"
    return ChurnPrediction(
        churn_probability=probability,
        predicted_class=predicted_class,
        model_version=MODEL_VERSION,
    )
