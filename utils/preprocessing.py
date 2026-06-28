from __future__ import annotations

import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def split_feature_types(dataframe: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return numeric and categorical feature names."""
    numeric_columns = [
        column for column in dataframe.columns if is_numeric_dtype(dataframe[column])
    ]
    categorical_columns = [
        column for column in dataframe.columns if column not in numeric_columns
    ]
    return numeric_columns, categorical_columns


def build_preprocessor(dataframe: pd.DataFrame) -> ColumnTransformer:
    """Build a safe preprocessing pipeline for mixed tabular data."""
    numeric_columns, categorical_columns = split_feature_types(dataframe)

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
    )


def basic_clean_dataframe(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Remove duplicates and fully empty columns while preserving raw values."""
    before_rows, before_columns = dataframe.shape
    cleaned = dataframe.dropna(axis=1, how="all").drop_duplicates().reset_index(drop=True)
    after_rows, after_columns = cleaned.shape

    return cleaned, {
        "rows_removed": before_rows - after_rows,
        "columns_removed": before_columns - after_columns,
        "remaining_rows": after_rows,
        "remaining_columns": after_columns,
    }
