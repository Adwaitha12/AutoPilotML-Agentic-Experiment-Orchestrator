
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
