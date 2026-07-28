"""Extracción, validación y filtrado del dataset de canciones."""

from pathlib import Path

import pandas as pd

TARGET_COLUMN = "Track Score"

COUNT_COLUMNS = [
    "Spotify Streams",
    "Spotify Playlist Count",
    "Spotify Playlist Reach",
    "YouTube Views",
    "YouTube Likes",
    "TikTok Posts",
    "TikTok Likes",
    "TikTok Views",
    "YouTube Playlist Reach",
    "AirPlay Spins",
    "SiriusXM Spins",
    "Deezer Playlist Reach",
    "Pandora Streams",
    "Pandora Track Stations",
    "Soundcloud Streams",
    "Shazam Counts",
]

CONTINUOUS_COLUMNS = [
    "Spotify Popularity",
    "Apple Music Playlist Count",
    "Deezer Playlist Count",
    "Amazon Playlist Count",
]

DATE_COLUMNS = ["Release Date"]
CATEGORICAL_COLUMNS = ["Explicit Track"]

MODEL_FEATURES = [
    *COUNT_COLUMNS,
    *CONTINUOUS_COLUMNS,
    *DATE_COLUMNS,
    *CATEGORICAL_COLUMNS,
]

LEAKAGE_COLUMNS = [
    "All Time Rank",
    "Track",
    "Album Name",
    "Artist",
    "ISRC",
    "TIDAL Popularity",
]

REQUIRED_COLUMNS = [TARGET_COLUMN, *MODEL_FEATURES]


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load the source CSV using its known latin-1 encoding."""
    csv_path = Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"No se encontró el CSV: {csv_path}")
    return pd.read_csv(csv_path, encoding="latin-1")


def filter_training_rows(
    data: pd.DataFrame,
    *,
    minimum_rows: int = 10,
) -> tuple[pd.DataFrame, pd.Series, dict[str, int]]:
    """Validate the schema and return aligned model features and target."""
    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(data.columns))
    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise ValueError(f"Faltan columnas requeridas: {missing_text}")

    deduplicated = data.drop_duplicates().copy()
    duplicates_removed = len(data) - len(deduplicated)

    numeric_target = pd.to_numeric(
        deduplicated[TARGET_COLUMN], errors="coerce"
    )
    valid_target = numeric_target.notna()
    invalid_target_rows_removed = int((~valid_target).sum())
    filtered = deduplicated.loc[valid_target].copy()
    filtered[TARGET_COLUMN] = numeric_target.loc[valid_target].astype(float)

    if len(filtered) < minimum_rows:
        raise ValueError(
            f"Se requieren al menos {minimum_rows} filas válidas para entrenar."
        )

    features = filtered.loc[:, MODEL_FEATURES].copy()
    target = filtered[TARGET_COLUMN].copy()
    audit = {
        "input_rows": len(data),
        "duplicates_removed": duplicates_removed,
        "invalid_target_rows_removed": invalid_target_rows_removed,
        "output_rows": len(filtered),
    }
    return features, target, audit
