import json
from pathlib import Path

import joblib
import pandas as pd
from preprocess import load_and_preprocess_data
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "churn_model.pkl"
METRICS_PATH = PROJECT_ROOT / "reports" / "metrics.json"
RANDOM_STATE = 42


def evaluate_model(model, X_test, y_test) -> dict[str, float]:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }


def train_and_evaluate() -> tuple[Pipeline, dict[str, dict[str, float]]]:
    X_train, X_test, y_train, y_test, preprocessor = load_and_preprocess_data(
        random_state=RANDOM_STATE
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE,
            class_weight="balanced",
        ),
    }

    trained_pipelines = {}
    metrics = {}

    for model_name, model in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", clone(preprocessor)),
                ("model", model),
            ]
        )
        pipeline.fit(X_train, y_train)

        trained_pipelines[model_name] = pipeline
        metrics[model_name] = evaluate_model(pipeline, X_test, y_test)

    comparison_table = pd.DataFrame(metrics).T.sort_values(
        by="f1_score", ascending=False
    )
    print("\nModel comparison:")
    print(comparison_table.round(4))

    best_model_name = comparison_table.index[0]
    best_pipeline = trained_pipelines[best_model_name]
    print(f"\nBest model based on F1-score: {best_model_name}")

    metrics["best_model"] = best_model_name

    return best_pipeline, metrics


def save_artifacts(model: Pipeline, metrics: dict[str, dict[str, float]]) -> None:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_PATH)

    with METRICS_PATH.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    print(f"\nSaved model to: {MODEL_PATH}")
    print(f"Saved metrics to: {METRICS_PATH}")


def main() -> None:
    best_model, metrics = train_and_evaluate()
    save_artifacts(best_model, metrics)


if __name__ == "__main__":
    main()
