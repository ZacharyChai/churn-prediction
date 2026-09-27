# Trains the churn model and serializes it as a single joblib pipeline
# (feature engineering + scaling + classifier) so the API can load one
# artifact and score raw customer records directly.
#
# Mirrors the feature engineering in churn_analysis.py; see that file
# for the EDA and model comparison this choice of model (logistic
# regression, AUC 0.847) is based on.
#
# The decision threshold is chosen here as well, from cross-validated
# predictions on the training split only, so the holdout numbers printed
# at the end are an honest read of the operating point the API serves.

import argparse

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.features import CATEGORICAL_COLS, MODEL_VERSION, RAW_FEATURE_COLUMNS, ChurnFeatureEngineer
from app.threshold import cost_minimizing_threshold

# Cost ratios reported in threshold_analysis.csv alongside the default.
REPORTED_COST_RATIOS = [1, 2, 3, 5, 10]


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


def holdout_row(cost_ratio, threshold, y_test, proba):
    flagged = proba >= threshold
    return {
        "cost_ratio": round(cost_ratio, 2),
        "threshold": threshold,
        "holdout_recall": round(recall_score(y_test, flagged), 3),
        "holdout_precision": round(precision_score(y_test, flagged), 3),
        "share_flagged": round(float(flagged.mean()), 3),
    }


def main(cost_ratio=None):
    df = pd.read_csv("Telco-Customer-Churn.csv")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    X = df[RAW_FEATURE_COLUMNS]
    y = (df["Churn"] == "Yes").astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Out-of-fold probabilities on the training split: every customer is
    # scored by a model that never saw them, and the holdout stays untouched.
    oof = cross_val_predict(
        build_pipeline(), X_train, y_train,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        method="predict_proba",
    )[:, 1]

    # With no real retention economics to work from, the default prices a
    # missed churner at the class ratio (about 2.8 non-churners per churner),
    # the same trade the balanced class weights already make. Pass
    # --cost-ratio when the business knows its numbers.
    class_ratio = float((y_train == 0).sum() / (y_train == 1).sum())
    if cost_ratio is None:
        cost_ratio = class_ratio
    threshold = cost_minimizing_threshold(y_train, oof, cost_ratio)

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    proba = pipeline.predict_proba(X_test)[:, 1]

    served = holdout_row(cost_ratio, threshold, y_test, proba)
    print(f"Holdout AUC: {roc_auc_score(y_test, proba):.4f}")
    print(
        f"Threshold {threshold:.2f} at cost ratio {cost_ratio:.2f}: "
        f"recall {served['holdout_recall']:.3f}, precision {served['holdout_precision']:.3f}, "
        f"{served['share_flagged']:.1%} of customers flagged"
    )

    ratios = sorted(set(REPORTED_COST_RATIOS + [round(class_ratio, 2), round(cost_ratio, 2)]))
    rows = [holdout_row(r, cost_minimizing_threshold(y_train, oof, r), y_test, proba) for r in ratios]
    pd.DataFrame(rows).to_csv("threshold_analysis.csv", index=False)
    print("Wrote threshold_analysis.csv")

    # Bundle the version and the threshold with the fitted pipeline so the API
    # always serves the artifact's own settings, not whatever a constant
    # elsewhere in the code happens to say.
    joblib.dump(
        {
            "pipeline": pipeline,
            "version": MODEL_VERSION,
            "threshold": threshold,
            "cost_ratio": round(cost_ratio, 2),
        },
        "model.joblib",
    )
    print(f"Saved model.joblib (version {MODEL_VERSION}, threshold {threshold:.2f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the churn model and choose its decision threshold.")
    parser.add_argument(
        "--cost-ratio", type=float, default=None,
        help="Cost of a missed churner relative to one wasted retention offer. "
             "Defaults to the training class ratio.",
    )
    main(parser.parse_args().cost_ratio)
