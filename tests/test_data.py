from pathlib import Path

import pandas as pd
import pytest

from spotify_sklearn_pipeline.data import (
    LEAKAGE_COLUMNS,
    MODEL_FEATURES,
    filter_training_rows,
    load_dataset,
)


def valid_model_row(score: float = 100.0) -> dict[str, object]:
    row = {column: 1 for column in MODEL_FEATURES}
    row["Release Date"] = "4/26/2024"
    row["Track Score"] = score
    row["All Time Rank"] = "1"
    return row


def test_load_dataset_reads_latin1(tmp_path: Path) -> None:
    csv_path = tmp_path / "songs.csv"
    pd.DataFrame(
        {"Track": ["Canción"], "Track Score": [1.0]}
    ).to_csv(csv_path, index=False, encoding="latin-1")

    loaded = load_dataset(csv_path)

    assert loaded.loc[0, "Track"] == "Canción"


def test_load_dataset_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="No se encontró el CSV"):
        load_dataset(tmp_path / "missing.csv")


def test_filter_training_rows_deduplicates_and_excludes_leakage() -> None:
    row = valid_model_row()
    data = pd.DataFrame([row, row])

    features, target, audit = filter_training_rows(data, minimum_rows=1)

    assert len(features) == len(target) == 1
    assert set(LEAKAGE_COLUMNS).isdisjoint(features.columns)
    assert audit == {
        "input_rows": 2,
        "duplicates_removed": 1,
        "invalid_target_rows_removed": 0,
        "output_rows": 1,
    }


def test_filter_training_rows_removes_invalid_target_values() -> None:
    valid = valid_model_row(10.0)
    invalid = valid_model_row()
    invalid["Track Score"] = "not-a-number"

    _, target, audit = filter_training_rows(
        pd.DataFrame([valid, invalid]), minimum_rows=1
    )

    assert target.tolist() == [10.0]
    assert audit["invalid_target_rows_removed"] == 1


def test_filter_training_rows_lists_missing_required_columns() -> None:
    with pytest.raises(ValueError, match="Faltan columnas requeridas"):
        filter_training_rows(
            pd.DataFrame({"Track Score": [1.0]}), minimum_rows=1
        )
