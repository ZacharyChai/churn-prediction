# Feature engineering shared by train_model.py (fits the pipeline) and the
# API (unpickles it) — must stay importable at this same module path in both
# places, since joblib/pickle resolves classes by module + name.

from sklearn.base import BaseEstimator, TransformerMixin

# Bump when the training pipeline or feature set changes. Baked into
# model.joblib at train time and echoed back in every /predict response,
# so a caller can tell which model version scored their request.
MODEL_VERSION = "1.0.0"

YES_NO_COLS = ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]
ADDON_COLS = [
    "MultipleLines", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
]
CATEGORICAL_COLS = ["InternetService", "Contract", "PaymentMethod"]
RAW_FEATURE_COLUMNS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges",
]


class ChurnFeatureEngineer(BaseEstimator, TransformerMixin):
    """Yes/No -> 0/1 conversion plus the two derived features from churn_analysis.py."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X[RAW_FEATURE_COLUMNS].copy()

        for col in YES_NO_COLS:
            df[col] = (df[col] == "Yes").astype(int)
        for col in ADDON_COLS:
            df[col] = (df[col] == "Yes").astype(int)
        df["gender"] = (df["gender"] == "Male").astype(int)

        df["charges_per_tenure"] = df["MonthlyCharges"] / (df["tenure"] + 1)
        df["num_services"] = df[ADDON_COLS].sum(axis=1)

        return df
