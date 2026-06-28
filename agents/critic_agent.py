from __future__ import annotations

<<<<<<< HEAD
import logging
from typing import Any, Dict, List, Optional

import pandas as pd

from agents.base_agent import BaseAgent
from core.context import ExperimentContext


class CriticAgent(BaseAgent):
    """Act as a senior ML reviewer for an experiment run.

    This agent inspects the experiment context and produces a qualitative review
    of potential issues such as overfitting, underfitting, class imbalance,
    small datasets, data leakage, metric suitability, and when cross-validation
    is recommended. It does not train or modify models.
    """

    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        self.logger = logger or logging.getLogger(__name__)

    def execute(self, context: ExperimentContext) -> ExperimentContext:
        """Analyze the experiment context and produce reviewer recommendations."""
        started_at = self._log_execution(context, "critic_agent")
        report = self._build_report(context)
        context.critic_analysis = report
        context.experiment_metadata["critic_analysis"] = report
        summary = report.get("summary", "") or ""
        self._record_agent_thought(
            context,
            "critic_agent",
            f"I recommend {', '.join(report.get('recommendations', [])[:2]) if report.get('recommendations') else 'continuing validation'}.",
        )
        self._log_completion("critic_agent", started_at)
        self.logger.info("Critic analysis completed")
        return context

    def run(self, context: ExperimentContext) -> ExperimentContext:
        """Backward-compatible alias for execute."""
        return self.execute(context)

    def _build_report(self, context: ExperimentContext) -> Dict[str, Any]:
        """Construct a comprehensive review of the experiment state."""
        quality_report = context.quality_report or {}
        metrics = context.metrics or {}
        problem_type = context.problem_type or "unknown"
        dataset = self._resolve_dataset(context)

        recommendations: List[str] = []
        issues: List[str] = []

        if dataset is not None:
            row_count = int(dataset.shape[0])
            if row_count < 100:
                issues.append("small dataset")
                recommendations.append("The dataset appears small; collect more data or use stricter validation to reduce variance.")

            if problem_type == "classification":
                target_values = dataset[context.target_column].dropna() if context.target_column in dataset.columns else []
                class_counts = target_values.value_counts(dropna=False).to_dict() if not target_values.empty else {}
                if class_counts:
                    min_count = min(class_counts.values())
                    max_count = max(class_counts.values())
                    if min_count / max_count < 0.2:
                        issues.append("class imbalance")
                        recommendations.append("The target distribution is imbalanced; consider class weighting, resampling, or evaluation with balanced metrics.")

        if problem_type == "classification":
            best_metrics = self._get_best_metrics(metrics)
            if best_metrics:
                if best_metrics.get("train_score", 0.0) - best_metrics.get("test_score", 0.0) > 0.15:
                    issues.append("possible overfitting")
                    recommendations.append("A notable gap between training and test performance suggests overfitting; inspect feature complexity and regularization.")
                elif best_metrics.get("train_score", 0.0) < 0.6 and best_metrics.get("test_score", 0.0) < 0.6:
                    issues.append("possible underfitting")
                    recommendations.append("Both train and test performance are weak, suggesting the model may be underfitting; try simpler or more expressive baselines and better features.")

                if context.best_model is None:
                    recommendations.append("A best model was not recorded; ensure evaluation has completed before relying on model rankings.")

            if context.problem_type == "classification" and metrics:
                recommendations.append("Use precision, recall, and F1 as primary metrics when class imbalance is present; accuracy alone can be misleading.")

        if problem_type == "regression":
            best_metrics = self._get_best_metrics(metrics)
            if best_metrics:
                if best_metrics.get("train_score", 0.0) - best_metrics.get("test_score", 0.0) > 0.2:
                    issues.append("possible overfitting")
                    recommendations.append("A large train-test gap suggests overfitting; consider regularization or feature selection.")
                elif best_metrics.get("train_score", 0.0) < 0.3 and best_metrics.get("test_score", 0.0) < 0.3:
                    issues.append("possible underfitting")
                    recommendations.append("The model is not capturing the target well; review preprocessing and feature engineering.")

        if quality_report:
            missing_values = quality_report.get("missing_values", {}).get("total_missing_values", 0)
            duplicates = quality_report.get("duplicate_rows", {}).get("total_duplicate_rows", 0)
            if missing_values > 0:
                recommendations.append("Missing values were present; confirm that imputation did not introduce bias or distort distributions.")
            if duplicates > 0:
                recommendations.append("Duplicate rows were found; ensure the data cleaning step did not remove meaningful repeated observations accidentally.")

        if dataset is not None and context.target_column and context.target_column in dataset.columns:
            recommendations.append("If the dataset is time-dependent or grouped, verify that train/test splits do not leak future information into training data.")

        if len(metrics) >= 2:
            recommendations.append("Use cross-validation for a more reliable estimate of generalization, especially with a small or imbalanced dataset.")

        if not recommendations:
            recommendations.append("No major issues were detected from the available evidence; continue validating with additional data and monitoring.")

        return {
            "problem_type": problem_type,
            "issues": issues,
            "recommendations": recommendations,
            "summary": self._build_summary(issues, recommendations),
        }

    def _resolve_dataset(self, context: ExperimentContext) -> Optional[pd.DataFrame]:
        """Resolve the dataset to review from the experiment context."""
        if context.cleaned_dataframe is not None:
            return context.cleaned_dataframe.copy()
        if context.dataset is None:
            return None
        if isinstance(context.dataset, pd.DataFrame):
            return context.dataset.copy()
        return pd.DataFrame(context.dataset)

    def _get_best_metrics(self, metrics: Dict[str, Dict[str, float]]) -> Optional[Dict[str, float]]:
        """Return the best-performing model metrics when available."""
        if not metrics:
            return None
        best_model_name = max(metrics, key=lambda name: self._ranking_score(metrics[name]))
        return metrics[best_model_name]

    @staticmethod
    def _ranking_score(metric_dict: Dict[str, float]) -> float:
        if not metric_dict:
            return 0.0
        if "r2" in metric_dict:
            return metric_dict.get("r2", 0.0)
        return metric_dict.get("accuracy", 0.0)

    @staticmethod
    def _build_summary(issues: List[str], recommendations: List[str]) -> str:
        if not issues:
            return "The experiment review did not surface major red flags, but the recommendations below can strengthen confidence in the results."
        return "The review identified the following concerns: " + ", ".join(issues) + "."


__all__ = ["CriticAgent"]
=======
from core.context import ExperimentContext


class CriticAgent:
    """Review experiment quality and produce practical recommendations."""

    def execute(self, context: ExperimentContext) -> ExperimentContext:
        comments: list[str] = []
        dataframe = context.active_dataset

        if len(dataframe) < 200:
            comments.append(
                "Dataset is small; use cross-validation before trusting the ranking."
            )

        if context.problem_type == "classification":
            distribution = dataframe[context.target_column].value_counts(normalize=True)
            if not distribution.empty and float(distribution.max()) > 0.75:
                comments.append(
                    "Target classes are imbalanced; prioritize F1 score over accuracy."
                )

        if context.best_model_score is not None:
            if context.problem_type == "classification" and context.best_model_score > 0.98:
                comments.append(
                    "Very high score detected; verify there is no target leakage."
                )
            if context.problem_type == "regression" and context.best_model_score < 0.2:
                comments.append(
                    "Low R2 score; consider feature engineering or more predictive data."
                )

        if context.optimization_goal == "most_explainable":
            comments.append(
                "Explainability was prioritized, so simpler models may be preferred over peak performance."
            )

        if not comments:
            comments.append(
                "No major experiment risks detected in this first-pass automated review."
            )

        context.critic_comments = comments
        context.critic_analysis = {
            "checks": [
                "small_dataset",
                "class_imbalance",
                "data_leakage_signal",
                "metric_recommendation",
            ],
            "recommendations": comments,
            "business_recommendation": self._business_recommendation(context),
        }
        return context

    def _business_recommendation(self, context: ExperimentContext) -> str:
        goal = context.business_goal or "the stated business outcome"
        if context.best_model is None:
            return f"Run the full experiment before using the model for {goal}."
        return (
            f"Use {context.best_model.replace('_', ' ')} as the baseline model for "
            f"{goal}, then validate it with domain-specific acceptance criteria."
        )
>>>>>>> origin/main
