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
