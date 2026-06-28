from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, precision_score, r2_score, recall_score, roc_auc_score

from agents.base_agent import BaseAgent
from core.context import ExperimentContext


class EvaluationAgent(BaseAgent):
    """Evaluate trained models and record the leaderboard and best model."""

    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        self.logger = logger or logging.getLogger(__name__)

    def execute(self, context: ExperimentContext) -> ExperimentContext:
        """Evaluate every trained model and update the shared experiment context."""
        started_at = self._log_execution(context, "evaluation_agent")
        if not context.trained_models:
            raise ValueError("ExperimentContext does not contain any trained models")

        if context.problem_type == "classification":
            metrics_by_model = self._evaluate_classification_models(context)
            leaderboard = self._build_classification_leaderboard(metrics_by_model)
        elif context.problem_type == "regression":
            metrics_by_model = self._evaluate_regression_models(context)
            leaderboard = self._build_regression_leaderboard(metrics_by_model)
        else:
            raise ValueError(f"Unsupported problem type: {context.problem_type}")

        context.metrics = self._merge_metrics(context.metrics, metrics_by_model)
        context.leaderboard = leaderboard
        context.experiment_metadata["evaluation_results"] = {
            "leaderboard": leaderboard,
            "best_model": leaderboard[0]["model_name"] if leaderboard else None,
        }
        context.best_model = context.trained_models.get(leaderboard[0]["model_name"]) if leaderboard else None

        best_model_name = leaderboard[0]["model_name"] if leaderboard else None
        explanation = self._build_best_model_explanation(leaderboard, context.problem_type)
        context.experiment_metadata["best_model_explanation"] = explanation
        self._record_agent_thought(
            context,
            "evaluation_agent",
            f"I ranked the models and selected {best_model_name or 'no model'} as the best performer.",
        )
        self._log_completion("evaluation_agent", started_at)
        self.logger.info("Evaluation completed; best model is '%s'", best_model_name)

        return context

    def run(self, context: ExperimentContext) -> ExperimentContext:
        """Backward-compatible alias for execute."""
        return self.execute(context)

    def _evaluate_classification_models(self, context: ExperimentContext) -> Dict[str, Dict[str, float]]:
        metrics_by_model: Dict[str, Dict[str, float]] = {}
        training_results = context.experiment_metadata.get("training_results", {})
        model_summaries = training_results.get("models", {})

        for model_name, model in context.trained_models.items():
            summary = model_summaries.get(model_name, {})
            y_true = summary.get("y_test", [])
            predictions = summary.get("test_predictions", [])
            features = summary.get("test_features", [])

            if not y_true or not predictions:
                continue

            y_true_series = pd.Series(y_true)
            predictions_series = pd.Series(predictions)
            metrics = {
                "accuracy": float(accuracy_score(y_true_series, predictions_series)),
                "precision": float(precision_score(y_true_series, predictions_series, average="weighted", zero_division=0)),
                "recall": float(recall_score(y_true_series, predictions_series, average="weighted", zero_division=0)),
                "f1": float(f1_score(y_true_series, predictions_series, average="weighted", zero_division=0)),
            }

            if features:
                try:
                    proba = self._predict_probabilities(model, features)
                    metrics["roc_auc"] = float(roc_auc_score(y_true_series, proba))
                except Exception:
                    metrics["roc_auc"] = float("nan")
            else:
                metrics["roc_auc"] = float("nan")

            metrics_by_model[model_name] = metrics

        return metrics_by_model

    def _evaluate_regression_models(self, context: ExperimentContext) -> Dict[str, Dict[str, float]]:
        metrics_by_model: Dict[str, Dict[str, float]] = {}
        training_results = context.experiment_metadata.get("training_results", {})
        model_summaries = training_results.get("models", {})

        for model_name, model in context.trained_models.items():
            summary = model_summaries.get(model_name, {})
            y_true = summary.get("y_test", [])
            predictions = summary.get("test_predictions", [])
            if not y_true or not predictions:
                continue

            y_true_series = pd.Series(y_true)
            predictions_series = pd.Series(predictions)
            metrics = {
                "mae": float(mean_absolute_error(y_true_series, predictions_series)),
                "rmse": float(mean_squared_error(y_true_series, predictions_series, squared=False)),
                "r2": float(r2_score(y_true_series, predictions_series)),
            }
            metrics_by_model[model_name] = metrics

        return metrics_by_model

    def _build_classification_leaderboard(self, metrics_by_model: Dict[str, Dict[str, float]]) -> List[Dict[str, Any]]:
        leaderboard: List[Dict[str, Any]] = []
        for model_name, metrics in metrics_by_model.items():
            leaderboard.append(
                {
                    "model_name": model_name,
                    "metrics": metrics,
                    "ranking_score": self._classification_ranking_score(metrics),
                }
            )

        leaderboard.sort(key=lambda item: item["ranking_score"], reverse=True)
        return leaderboard

    def _build_regression_leaderboard(self, metrics_by_model: Dict[str, Dict[str, float]]) -> List[Dict[str, Any]]:
        leaderboard: List[Dict[str, Any]] = []
        for model_name, metrics in metrics_by_model.items():
            leaderboard.append(
                {
                    "model_name": model_name,
                    "metrics": metrics,
                    "ranking_score": self._regression_ranking_score(metrics),
                }
            )

        leaderboard.sort(key=lambda item: item["ranking_score"], reverse=True)
        return leaderboard

    def _merge_metrics(self, existing_metrics: Dict[str, Dict[str, float]], new_metrics: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        merged = dict(existing_metrics)
        for model_name, metrics in new_metrics.items():
            merged[model_name] = {**merged.get(model_name, {}), **metrics}
        return merged

    def _build_best_model_explanation(self, leaderboard: Sequence[Dict[str, Any]], problem_type: Optional[str]) -> str:
        if not leaderboard:
            return "No models were evaluated."

        best = leaderboard[0]
        if problem_type == "classification":
            metrics = best["metrics"]
            return (
                f"{best['model_name']} was selected as the best model because it achieved the strongest "
                f"overall classification performance with accuracy {metrics.get('accuracy', float('nan')):.3f}, "
                f"precision {metrics.get('precision', float('nan')):.3f}, recall {metrics.get('recall', float('nan')):.3f}, "
                f"F1 {metrics.get('f1', float('nan')):.3f}, and ROC AUC {metrics.get('roc_auc', float('nan')):.3f}."
            )

        metrics = best["metrics"]
        return (
            f"{best['model_name']} was selected as the best model because it achieved the highest R² "
            f"({metrics.get('r2', float('nan')):.3f}) while keeping MAE {metrics.get('mae', float('nan')):.3f} "
            f"and RMSE {metrics.get('rmse', float('nan')):.3f} low."
        )

    @staticmethod
    def _classification_ranking_score(metrics: Dict[str, float]) -> float:
        values = [metrics.get(name, 0.0) for name in ("accuracy", "precision", "recall", "f1", "roc_auc") if not pd.isna(metrics.get(name, 0.0))]
        return float(sum(values) / len(values)) if values else 0.0

    @staticmethod
    def _regression_ranking_score(metrics: Dict[str, float]) -> float:
        return float(metrics.get("r2", 0.0) - 0.1 * metrics.get("mae", 0.0) - 0.05 * metrics.get("rmse", 0.0))

    @staticmethod
    def _predict_probabilities(model: Any, features: Sequence[Dict[str, Any]]) -> Any:
        dataframe = pd.DataFrame(features)
        if hasattr(model, "predict_proba"):
            return model.predict_proba(dataframe)[:, 1]
        if hasattr(model, "decision_function"):
            return model.decision_function(dataframe)
        raise ValueError("Model does not expose probability or decision scores")


__all__ = ["EvaluationAgent"]
