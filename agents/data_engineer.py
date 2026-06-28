"""
Data Engineer Agent

Responsibilities:
- Remove duplicates
- Fill missing values
- Encode categorical columns
- Scale numerical columns

Provides a clean() method that fits a preprocessing pipeline and
returns a cleaned DataFrame ready for modeling, plus the fitted
preprocessor for reuse.
"""
from typing import Optional, Tuple, List, Dict

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer


class DataEngineer:
    """Data engineering utilities for ML experiments."""

    def __init__(self) -> None:
        self.preprocessor: Optional[ColumnTransformer] = None
        self.numeric_features: List[str] = []
        self.categorical_features: List[str] = []

    def _identify_columns(self, df: pd.DataFrame, target: Optional[str] = None) -> None:
        cols = df.columns.tolist()
        if target and target in cols:
            cols = [c for c in cols if c != target]

        self.numeric_features = df[cols].select_dtypes(include="number").columns.tolist()
        self.categorical_features = df[cols].select_dtypes(exclude="number").columns.tolist()

    def build_preprocessor(self) -> ColumnTransformer:
        """Builds a ColumnTransformer with imputation, encoding and scaling."""
        # Numeric pipeline: median imputation + standard scaling
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        # Categorical pipeline: most frequent imputation + one-hot encoding
        # OneHotEncoder changed keyword in newer sklearn versions (sparse -> sparse_output)
        try:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
        except TypeError:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", ohe)
        ])

        preprocessor = ColumnTransformer([
            ("num", numeric_pipeline, self.numeric_features),
            ("cat", categorical_pipeline, self.categorical_features)
        ], remainder="drop")

        self.preprocessor = preprocessor
        return preprocessor

    def clean(self, df: pd.DataFrame, target: Optional[str] = None) -> Tuple[pd.DataFrame, Optional[pd.Series], ColumnTransformer]:
        """
        Clean the dataset and return (X_clean, y, preprocessor).

        Steps:
        - Drop duplicate rows
        - Separate target (if provided)
        - Identify numeric/categorical columns
        - Build and fit a preprocessor
        - Transform features and return a cleaned DataFrame

        Returns
        -------
        X_clean: pd.DataFrame
            Cleaned feature matrix with properly encoded/scaled columns
        y: Optional[pd.Series]
            Target series if `target` provided, else None
        preprocessor: ColumnTransformer
            The fitted preprocessor pipeline
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError("`df` must be a pandas DataFrame")

        # 1. Remove duplicate rows
        df_cleaned = df.drop_duplicates().reset_index(drop=True)

        # 2. Separate target
        y = None
        if target and target in df_cleaned.columns:
            y = df_cleaned[target].copy()
            X = df_cleaned.drop(columns=[target])
        else:
            X = df_cleaned.copy()

        # 3. Identify columns
        self._identify_columns(df_cleaned, target=target)

        # 4. Build preprocessor
        preprocessor = self.build_preprocessor()

        # 5. Fit and transform
        if (len(self.numeric_features) + len(self.categorical_features)) == 0:
            # No features to process; return empty DataFrame
            X_transformed = pd.DataFrame(index=X.index)
        else:
            X_transformed_np = preprocessor.fit_transform(X)

            # Build column names for transformed output
            feature_names: List[str] = []
            if len(self.numeric_features) > 0:
                feature_names.extend(self.numeric_features)

            if len(self.categorical_features) > 0:
                # Extract feature names from OneHotEncoder
                # ColumnTransformer's named transformers can be accessed
                cat_transformer = preprocessor.named_transformers_["cat"].named_steps["onehot"]
                try:
                    cat_feature_names = cat_transformer.get_feature_names_out(self.categorical_features).tolist()
                except Exception:
                    # Fallback
                    cat_feature_names = [f"{col}_ohe_{i}" for col in self.categorical_features for i in range(1)]
                feature_names.extend(cat_feature_names)

            X_transformed = pd.DataFrame(X_transformed_np, columns=feature_names, index=X.index)

        return X_transformed, y, preprocessor

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply a previously fitted preprocessor to a new DataFrame.

        Raises
        ------
        RuntimeError if the preprocessor has not been fitted.
        """
        if self.preprocessor is None:
            raise RuntimeError("Preprocessor has not been fitted. Call clean() first.")

        # Remove duplicates similarly to training
        df_cleaned = df.drop_duplicates().reset_index(drop=True)

        # Drop columns not present in the preprocessor input (safeguard)
        expected_cols = self.numeric_features + self.categorical_features
        missing = [c for c in expected_cols if c not in df_cleaned.columns]
        if missing:
            raise ValueError(f"Missing expected columns for transform: {missing}")

        X = df_cleaned[expected_cols]
        X_transformed_np = self.preprocessor.transform(X)

        # Recreate column names
        feature_names: List[str] = []
        if len(self.numeric_features) > 0:
            feature_names.extend(self.numeric_features)
        if len(self.categorical_features) > 0:
            cat_transformer = self.preprocessor.named_transformers_["cat"].named_steps["onehot"]
            try:
                cat_feature_names = cat_transformer.get_feature_names_out(self.categorical_features).tolist()
            except Exception:
                cat_feature_names = [f"{col}_ohe_{i}" for col in self.categorical_features for i in range(1)]
            feature_names.extend(cat_feature_names)

        X_transformed = pd.DataFrame(X_transformed_np, columns=feature_names, index=df_cleaned.index)
        return X_transformed
