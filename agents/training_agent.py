<<<<<<< HEAD

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import joblib
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from agents.base_agent import BaseAgent
from core.context import ExperimentContext


class TrainingAgent(BaseAgent):
    """Train all selected models on the prepared dataset and persist the results.

    The agent splits the dataset into training and test sets, fits every model
    selected by the model-selection stage, and records trained model instances,
    predictions, and evaluation scores in the shared experiment context.
    """

    def __init__(self, logger: Optional[logging.Logger] = None, test_size: float = 0.2, random_state: int = 42) -> None:
        self.logger = logger or logging.getLogger(__name__)
        self.test_size = test_size
        self.random_state = random_state

    def execute(self, context: ExperimentContext) -> ExperimentContext:
        """Train the models referenced by the experiment context."""
        started_at = self._log_execution(context, "training_agent")
        dataframe = self._resolve_dataframe(context)
        target_column = self._resolve_target_column(context)
        if target_column is None:
            raise ValueError("ExperimentContext must define a target_column")

        X, y = self._prepare_features_and_target(dataframe, target_column)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

        results: Dict[str, Dict[str, Any]] = {}
        for model_name in context.selected_models:
            model = self._instantiate_model(model_name)
            self.logger.info("Training model '%s'", model_name)
            model.fit(X_train, y_train)

            train_predictions = model.predict(X_train)
            test_predictions = model.predict(X_test)

            results[model_name] = {
                "model": model,
                "train_predictions": train_predictions,
                "test_predictions": test_predictions,
                "train_score": self._score_predictions(y_train, train_predictions, context.problem_type),
                "test_score": self._score_predictions(y_test, test_predictions, context.problem_type),
            }

        context.trained_models = {name: result["model"] for name, result in results.items()}
        context.predictions = {
            name: {
                "train_predictions": result["train_predictions"].tolist() if hasattr(result["train_predictions"], "tolist") else result["train_predictions"],
                "test_predictions": result["test_predictions"].tolist() if hasattr(result["test_predictions"], "tolist") else result["test_predictions"],
            }
            for name, result in results.items()
        }
        context.metrics = {
            name: {
                "train_score": result["train_score"],
                "test_score": result["test_score"],
            }
            for name, result in results.items()
        }
        context.experiment_metadata["training_results"] = {
            "train_test_split": {
                "test_size": self.test_size,
                "random_state": self.random_state,
                "train_shape": list(X_train.shape),
                "test_shape": list(X_test.shape),
            },
            "models": {
                name: {
                    "train_score": result["train_score"],
                    "test_score": result["test_score"],
                    "train_predictions": result["train_predictions"].tolist() if hasattr(result["train_predictions"], "tolist") else result["train_predictions"],
                    "test_predictions": result["test_predictions"].tolist() if hasattr(result["test_predictions"], "tolist") else result["test_predictions"],
                    "y_train": y_train.tolist() if hasattr(y_train, "tolist") else y_train,
                    "y_test": y_test.tolist() if hasattr(y_test, "tolist") else y_test,
                    "test_features": X_test.to_dict(orient="records"),
                }
                for name, result in results.items()
            },
        }

        self._record_agent_thought(
            context,
            "training_agent",
            f"I trained {len(results)} model(s) and recorded their training and test scores.",
        )
        self._log_completion("training_agent", started_at)
        self.logger.info("Training completed for %s model(s)", len(results))
        return context

    def run(self, context: ExperimentContext) -> ExperimentContext:
        """Backward-compatible alias for execute."""
        return self.execute(context)

    def _resolve_dataframe(self, context: ExperimentContext) -> pd.DataFrame:
        """Resolve the dataframe for training from the experiment context."""
        if context.cleaned_dataframe is not None:
            return context.cleaned_dataframe.copy()
        if context.dataset is None:
            raise ValueError("ExperimentContext must contain dataset or cleaned_dataframe")
        if isinstance(context.dataset, pd.DataFrame):
            return context.dataset.copy()
        return pd.DataFrame(context.dataset)

    def _resolve_target_column(self, context: ExperimentContext) -> Optional[str]:
        """Resolve the target column from the experiment context."""
        if context.target_column:
            return context.target_column
        return None

    def _prepare_features_and_target(self, dataframe: pd.DataFrame, target_column: str) -> Tuple[pd.DataFrame, pd.Series]:
        """Separate features and target and ensure the features are numeric for modeling."""
        features = dataframe.drop(columns=[target_column]).copy()
        target = dataframe[target_column]

        for column in features.columns:
            if not pd.api.types.is_numeric_dtype(features[column]):
                features[column] = features[column].astype(str)
                features[column] = pd.factorize(features[column])[0]

        return features, target

    def _instantiate_model(self, model_name: str) -> BaseEstimator:
        """Instantiate a model by name using the same selection names as the model-selection agent."""
        model_map = {
            "LogisticRegression": lambda: __import__("sklearn.linear_model", fromlist=["LogisticRegression"]).LogisticRegression(max_iter=1000),
            "DecisionTreeClassifier": lambda: __import__("sklearn.tree", fromlist=["DecisionTreeClassifier"]).DecisionTreeClassifier(random_state=42),
            "RandomForestClassifier": lambda: __import__("sklearn.ensemble", fromlist=["RandomForestClassifier"]).RandomForestClassifier(random_state=42),
            "GradientBoostingClassifier": lambda: __import__("sklearn.ensemble", fromlist=["GradientBoostingClassifier"]).GradientBoostingClassifier(random_state=42),
            "LinearRegression": lambda: __import__("sklearn.linear_model", fromlist=["LinearRegression"]).LinearRegression(),
            "DecisionTreeRegressor": lambda: __import__("sklearn.tree", fromlist=["DecisionTreeRegressor"]).DecisionTreeRegressor(random_state=42),
            "RandomForestRegressor": lambda: __import__("sklearn.ensemble", fromlist=["RandomForestRegressor"]).RandomForestRegressor(random_state=42),
            "GradientBoostingRegressor": lambda: __import__("sklearn.ensemble", fromlist=["GradientBoostingRegressor"]).GradientBoostingRegressor(random_state=42),
        }

        if model_name not in model_map:
            raise ValueError(f"Unsupported model name: {model_name}")
        return model_map[model_name]()

    def _score_predictions(self, actual: pd.Series, predictions: Any, problem_type: Optional[str]) -> float:
        """Calculate a suitable metric based on the detected problem type."""
        if problem_type == "classification":
            return float(accuracy_score(actual, predictions))
        if problem_type == "regression":
            return float(r2_score(actual, predictions))
        raise ValueError(f"Unsupported problem type: {problem_type}")


__all__ = ["TrainingAgent"]
=======
from __future__ import annotations

from time import perf_counter
from typing import Any

from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from core.context import ExperimentContext
from utils.metrics import classification_metrics, regression_metrics
from utils.preprocessing import build_preprocessor


class TrainingAgent:
    """Train every selected model and store metrics in the context."""

    RANDOM_STATE = 42

    def execute(self, context: ExperimentContext) -> ExperimentContext:
        if context.problem_type is None:
            raise ValueError("Problem type must exist before training.")
        if not context.selected_models:
            raise ValueError("No models were selected for training.")

        dataframe = context.active_dataset
        x_data = dataframe[context.feature_columns]
        y_data = dataframe[context.target_column]
        stratify = y_data if self._can_stratify(context.problem_type, y_data) else None

        x_train, x_test, y_train, y_test = train_test_split(
            x_data,
            y_data,
            test_size=context.test_size,
            random_state=self.RANDOM_STATE,
            stratify=stratify,
        )

        context.train_shape = tuple(x_train.shape)
        context.test_shape = tuple(x_test.shape)
        context.metadata["y_test"] = list(y_test)

        for model_name in context.selected_models:
            estimator = self._build_estimator(model_name)
            pipeline = Pipeline(
                steps=[
                    ("preprocessor", build_preprocessor(x_train)),
                    ("model", estimator),
                ]
            )

            started_at = perf_counter()
            pipeline.fit(x_train, y_train)
            training_seconds = perf_counter() - started_at

            predictions = pipeline.predict(x_test)
            probability_scores = self._probability_scores(pipeline, x_test)
            metrics = self._score(context.problem_type, y_test, predictions, probability_scores)
            metrics["training_seconds"] = float(training_seconds)

            context.trained_models[model_name] = pipeline
            context.metrics[model_name] = metrics
            context.predictions[model_name] = list(predictions)
            if probability_scores is not None:
                context.probabilities[model_name] = list(probability_scores)
            if context.problem_type == "classification":
                context.classification_reports[model_name] = self._classification_report(
                    y_test,
                    predictions,
                )
            context.feature_importances[model_name] = self._feature_importance(pipeline)

        return context

    def _build_estimator(self, model_name: str) -> Any:
        models: dict[str, Any] = {
            "logistic_regression": LogisticRegression(max_iter=1000),
            "decision_tree": DecisionTreeClassifier(random_state=self.RANDOM_STATE),
            "random_forest": RandomForestClassifier(
                n_estimators=100,
                random_state=self.RANDOM_STATE,
            ),
            "gradient_boosting": GradientBoostingClassifier(
                random_state=self.RANDOM_STATE
            ),
            "linear_regression": LinearRegression(),
            "decision_tree_regressor": DecisionTreeRegressor(
                random_state=self.RANDOM_STATE
            ),
            "random_forest_regressor": RandomForestRegressor(
                n_estimators=100,
                random_state=self.RANDOM_STATE,
            ),
            "gradient_boosting_regressor": GradientBoostingRegressor(
                random_state=self.RANDOM_STATE
            ),
        }

        if model_name not in models:
            raise ValueError(f"Unsupported model selected: {model_name}")
        return models[model_name]

    def _score(
        self,
        problem_type: str,
        y_true: Any,
        predictions: Any,
        probability_scores: Any | None,
    ) -> dict[str, float]:
        if problem_type == "classification":
            return classification_metrics(y_true, predictions, probability_scores)
        return regression_metrics(y_true, predictions)

    def _can_stratify(self, problem_type: str, y_data: Any) -> bool:
        if problem_type != "classification":
            return False
        return bool(y_data.value_counts().min() >= 2)

    def _probability_scores(self, pipeline: Pipeline, x_test: Any) -> Any | None:
        model = pipeline.named_steps["model"]
        if not hasattr(model, "predict_proba"):
            return None

        probabilities = pipeline.predict_proba(x_test)
        if probabilities.shape[1] != 2:
            return None
        return probabilities[:, 1]

    def _classification_report(self, y_true: Any, predictions: Any) -> list[dict[str, Any]]:
        report = classification_report(
            y_true,
            predictions,
            output_dict=True,
            zero_division=0,
        )
        rows = []
        for label, values in report.items():
            if isinstance(values, dict):
                rows.append({"label": label, **values})
            else:
                rows.append({"label": label, "score": values})
        return rows

    def _feature_importance(self, pipeline: Pipeline) -> list[dict[str, Any]]:
        preprocessor = pipeline.named_steps["preprocessor"]
        model = pipeline.named_steps["model"]

        try:
            feature_names = list(preprocessor.get_feature_names_out())
        except Exception:
            feature_names = []

        raw_importances = None
        if hasattr(model, "feature_importances_"):
            raw_importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            coefficients = model.coef_
            raw_importances = abs(coefficients).mean(axis=0) if coefficients.ndim > 1 else abs(coefficients)

        if raw_importances is None or not feature_names:
            return []

        total = float(sum(raw_importances)) or 1.0
        rows = [
            {
                "feature": feature.replace("numeric__", "").replace("categorical__", ""),
                "importance": float(value),
                "contribution": float(value / total),
            }
            for feature, value in zip(feature_names, raw_importances)
        ]
        return sorted(rows, key=lambda row: row["importance"], reverse=True)[:10]
>>>>>>> origin/main
