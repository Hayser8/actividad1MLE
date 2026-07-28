"""Transformadores y composición del pipeline de scikit-learn."""

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from spotify_sklearn_pipeline.data import (
    CATEGORICAL_COLUMNS,
    CONTINUOUS_COLUMNS,
    COUNT_COLUMNS,
    DATE_COLUMNS,
)


class CommaNumberTransformer(BaseEstimator, TransformerMixin):
    """Convert comma-separated numeric strings into a float matrix."""

    def fit(self, X: object, y: object = None) -> "CommaNumberTransformer":
        frame = pd.DataFrame(X)
        self.n_features_in_ = frame.shape[1]
        self.feature_names_in_ = np.asarray(frame.columns, dtype=object)
        return self

    def transform(self, X: object) -> np.ndarray:
        frame = pd.DataFrame(X)
        parsed = frame.apply(
            lambda column: pd.to_numeric(
                column.astype("string").str.replace(",", "", regex=False),
                errors="coerce",
            )
        )
        return parsed.to_numpy(dtype=float)

    def get_feature_names_out(
        self, input_features: Sequence[str] | None = None
    ) -> np.ndarray:
        if input_features is not None:
            return np.asarray(input_features, dtype=object)
        return self.feature_names_in_


class DateFeatureTransformer(BaseEstimator, TransformerMixin):
    """Derive calendar fields and age in days from one release-date column."""

    def __init__(self, reference_date: str = "2024-12-31") -> None:
        self.reference_date = reference_date

    def fit(self, X: object, y: object = None) -> "DateFeatureTransformer":
        frame = pd.DataFrame(X)
        if frame.shape[1] != 1:
            raise ValueError("DateFeatureTransformer requiere exactamente una columna.")
        self.n_features_in_ = 1
        return self

    def transform(self, X: object) -> np.ndarray:
        dates = pd.to_datetime(pd.DataFrame(X).iloc[:, 0], errors="coerce")
        reference = pd.Timestamp(self.reference_date)
        age_days = (reference - dates).dt.days
        return np.column_stack(
            [
                dates.dt.year,
                dates.dt.month,
                dates.dt.dayofweek,
                age_days,
            ]
        ).astype(float)

    def get_feature_names_out(
        self, input_features: Sequence[str] | None = None
    ) -> np.ndarray:
        return np.asarray(
            [
                "release_year",
                "release_month",
                "release_day_of_week",
                "release_age_days",
            ],
            dtype=object,
        )


def build_model_pipeline(
    *,
    random_state: int = 42,
    n_estimators: int = 200,
) -> Pipeline:
    """Build the complete preprocessing and random-forest estimator pipeline."""
    count_pipeline = Pipeline(
        [
            ("parse_commas", CommaNumberTransformer()),
            ("impute", SimpleImputer(strategy="median")),
            (
                "log1p",
                FunctionTransformer(
                    np.log1p,
                    feature_names_out="one-to-one",
                ),
            ),
            ("scale", StandardScaler()),
        ]
    )
    continuous_pipeline = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    date_pipeline = Pipeline(
        [
            ("date_parts", DateFeatureTransformer()),
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("scale", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            (
                "one_hot",
                OneHotEncoder(handle_unknown="ignore"),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        [
            ("counts", count_pipeline, COUNT_COLUMNS),
            ("continuous", continuous_pipeline, CONTINUOUS_COLUMNS),
            ("date", date_pipeline, DATE_COLUMNS),
            ("categorical", categorical_pipeline, CATEGORICAL_COLUMNS),
        ],
        remainder="drop",
    )

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
    )
    return Pipeline(
        [
            ("prepare", preprocessor),
            ("model", model),
        ]
    )
