import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from spotify_sklearn_pipeline.data import MODEL_FEATURES
from spotify_sklearn_pipeline.train import train_and_evaluate


def make_training_csv(tmp_path: Path, rows: int = 60) -> Path:
    records: list[dict[str, object]] = []
    for index in range(rows):
        record = {
            column: f"{10_000 + (index * 137):,}" for column in MODEL_FEATURES
        }
        record["Release Date"] = f"{(index % 12) + 1}/15/{2020 + index % 5}"
        record["Explicit Track"] = index % 2
        record["Spotify Popularity"] = 30 + index % 70
        record["Apple Music Playlist Count"] = 5 + index
        record["Deezer Playlist Count"] = np.nan if index % 7 == 0 else index
        record["Amazon Playlist Count"] = index % 40
        record["Track Score"] = 20 + index * 2.5
        record["All Time Rank"] = str(index + 1)
        records.append(record)
    csv_path = tmp_path / "training.csv"
    pd.DataFrame(records).to_csv(csv_path, index=False, encoding="latin-1")
    return csv_path


def test_train_and_evaluate_creates_reusable_artifacts(tmp_path: Path) -> None:
    csv_path = make_training_csv(tmp_path)

    result = train_and_evaluate(csv_path, tmp_path / "artifacts")

    assert result.model_path.is_file()
    assert result.metrics_path.is_file()
    assert result.predictions_path.is_file()
    assert set(result.metrics["model"]) == {"mae", "rmse", "r2"}
    assert set(result.metrics["baseline"]) == {"mae", "rmse", "r2"}

    serialized = json.loads(result.metrics_path.read_text(encoding="utf-8"))
    assert serialized["split"]["test_rows"] == 12

    loaded_pipeline = joblib.load(result.model_path)
    predictions = loaded_pipeline.predict(result.X_test.head(2))
    assert len(predictions) == 2
    assert np.isfinite(predictions).all()
