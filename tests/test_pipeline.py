import numpy as np
import pandas as pd

from spotify_sklearn_pipeline.data import MODEL_FEATURES
from spotify_sklearn_pipeline.pipeline import (
    CommaNumberTransformer,
    DateFeatureTransformer,
    build_model_pipeline,
)


def representative_features(rows: int = 20) -> pd.DataFrame:
    values: list[dict[str, object]] = []
    for index in range(rows):
        row = {column: f"{1_000 + index:,}" for column in MODEL_FEATURES}
        row["Release Date"] = f"{(index % 12) + 1}/15/2024"
        row["Explicit Track"] = index % 2
        row["Spotify Popularity"] = 40 + index
        row["Apple Music Playlist Count"] = 10 + index
        row["Deezer Playlist Count"] = np.nan if index == 0 else index
        row["Amazon Playlist Count"] = index + 2
        values.append(row)
    return pd.DataFrame(values)


def test_comma_number_transformer_parses_and_preserves_missing() -> None:
    values = pd.DataFrame({"streams": ["1,234", None, "bad"]})

    result = CommaNumberTransformer().fit_transform(values)

    assert result[0, 0] == 1234
    assert np.isnan(result[1, 0])
    assert np.isnan(result[2, 0])


def test_date_transformer_returns_hand_checked_features() -> None:
    values = pd.DataFrame({"Release Date": ["4/26/2024", None]})

    result = DateFeatureTransformer(
        reference_date="2024-12-31"
    ).fit_transform(values)

    assert result.shape == (2, 4)
    assert result[0].tolist() == [2024.0, 4.0, 4.0, 249.0]
    assert np.isnan(result[1]).all()


def test_model_pipeline_fits_and_predicts_with_missing_values() -> None:
    features = representative_features()
    target = pd.Series(np.linspace(10, 200, len(features)))

    pipeline = build_model_pipeline(random_state=42, n_estimators=10)
    pipeline.fit(features, target)
    predictions = pipeline.predict(features.head(3))

    assert predictions.shape == (3,)
    assert np.isfinite(predictions).all()
