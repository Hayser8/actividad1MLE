"""Orquestación reproducible de entrenamiento, evaluación y artefactos."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from spotify_sklearn_pipeline.data import filter_training_rows, load_dataset
from spotify_sklearn_pipeline.pipeline import build_model_pipeline


@dataclass
class TrainingResult:
    """In-memory results and paths produced by a training run."""

    pipeline: Pipeline
    metrics: dict[str, Any]
    audit: dict[str, int]
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    predictions: pd.DataFrame
    model_path: Path
    metrics_path: Path
    predictions_path: Path


def regression_metrics(
    actual: pd.Series,
    predicted: object,
) -> dict[str, float]:
    """Return JSON-safe regression metrics."""
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(root_mean_squared_error(actual, predicted)),
        "r2": float(r2_score(actual, predicted)),
    }


def train_and_evaluate(
    csv_path: str | Path,
    output_dir: str | Path,
    *,
    random_state: int = 42,
    test_size: float = 0.2,
) -> TrainingResult:
    """Run extraction, filtering, splitting, fitting and serialization."""
    data = load_dataset(csv_path)
    features, target, audit = filter_training_rows(data)
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
    )

    pipeline = build_model_pipeline(random_state=random_state)
    pipeline.fit(X_train, y_train)
    model_predictions = pipeline.predict(X_test)

    baseline = DummyRegressor(strategy="median")
    baseline.fit(X_train, y_train)
    baseline_predictions = baseline.predict(X_test)

    model_metrics = regression_metrics(y_test, model_predictions)
    baseline_metrics = regression_metrics(y_test, baseline_predictions)
    baseline_mae = baseline_metrics["mae"]
    improvement_percent = (
        100.0 * (baseline_mae - model_metrics["mae"]) / baseline_mae
        if baseline_mae
        else 0.0
    )

    metrics: dict[str, Any] = {
        "model": model_metrics,
        "baseline": baseline_metrics,
        "mae_improvement_percent": float(improvement_percent),
        "split": {
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "test_size": float(test_size),
            "random_state": int(random_state),
        },
        "audit": audit,
    }
    predictions = pd.DataFrame(
        {
            "actual_track_score": y_test.to_numpy(),
            "predicted_track_score": model_predictions,
            "baseline_prediction": baseline_predictions,
        },
        index=y_test.index,
    ).sort_index()

    artifact_dir = Path(output_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifact_dir / "pipeline.joblib"
    metrics_path = artifact_dir / "metrics.json"
    predictions_path = artifact_dir / "predictions.csv"

    joblib.dump(pipeline, model_path)
    metrics_path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    predictions.to_csv(predictions_path, index_label="source_row")

    return TrainingResult(
        pipeline=pipeline,
        metrics=metrics,
        audit=audit,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        predictions=predictions,
        model_path=model_path,
        metrics_path=metrics_path,
        predictions_path=predictions_path,
    )
