# Trains the churn model and serializes it as a single joblib pipeline
# (feature engineering + scaling + classifier) so the API can load one
# artifact and score raw customer records directly.
#
# Mirrors the feature engineering in churn_analysis.py — see that file
# for the EDA and model comparison this choice of model (logistic
# regression, AUC 0.847) is based on.

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.features import CATEGORICAL_COLS, MODEL_VERSION, RAW_FEATURE_COLUMNS, ChurnFeatureEngineer


def build_pipeline():
    numeric_cols = [
        c for c in RAW_FEATURE_COLUMNS + ["charges_per_tenure", "num_services"]
        if c not in CATEGORICAL_COLS
    ]
    preprocess = ColumnTransformer([
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CATEGORICAL_COLS),
        ("num", StandardScaler(), numeric_cols),
    ])
    return Pipeline([
        ("engineer", ChurnFeatureEngineer()),
        ("preprocess", preprocess),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
    ])


def main():
    df = pd.read_csv("Telco-Customer-Churn.csv")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    X = df[RAW_FEATURE_COLUMNS]
    y = (df["Churn"] == "Yes").astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    proba = pipeline.predict_proba(X_test)[:, 1]
    print(f"Holdout AUC: {roc_auc_score(y_test, proba):.4f}")

    # Bundle the version alongside the fitted pipeline so the API always
    # reports the version of the artifact it actually loaded, not whatever
    # a constant elsewhere in the code happens to say.
    joblib.dump({"pipeline": pipeline, "version": MODEL_VERSION}, "model.joblib")
    print(f"Saved model.joblib (version {MODEL_VERSION})")


if __name__ == "__main__":
    main()
