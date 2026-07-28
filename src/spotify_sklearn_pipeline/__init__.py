"""Pipeline reproducible para estimar el rendimiento de canciones."""

from spotify_sklearn_pipeline.data import (
    LEAKAGE_COLUMNS,
    MODEL_FEATURES,
    TARGET_COLUMN,
    filter_training_rows,
    load_dataset,
)

__all__ = [
    "LEAKAGE_COLUMNS",
    "MODEL_FEATURES",
    "TARGET_COLUMN",
    "filter_training_rows",
    "load_dataset",
]

__version__ = "0.1.0"
