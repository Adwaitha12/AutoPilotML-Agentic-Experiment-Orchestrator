from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix

from agents.base_agent import BaseAgent
from core.context import ExperimentContext


class VisualizationAgent(BaseAgent):
    """Generate analytical charts for experiment results and save them to disk."""

    def __init__(self, output_dir: Optional[str] = None, logger: Optional[logging.Logger] = None) -> None:
        self.output_dir = Path(output_dir or "outputs/charts")
        self.logger = logger or logging.getLogger(__name__)

    def execute(self, context: ExperimentContext) -> ExperimentContext:
        """Create the requested charts and persist their paths in the experiment context."""
        started_at = self._log_execution(context, "visualization_agent")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        chart_paths: Dict[str, str] = {}
        plotly_visualizations: Dict[str, Any] = {}

        try:
            chart_paths.update(self._plot_missing_value_chart(context))
            chart_paths.update(self._plot_accuracy_comparison(context))
            if context.problem_type == "classification":
                chart_paths.update(self._plot_confusion_matrix(context))
                roc_curve_path = self._plot_roc_curve(context)
                if roc_curve_path:
                    chart_paths["roc_curve"] = roc_curve_path
                roc_curve_figure = self._build_roc_curve_plotly(context)
                if roc_curve_figure is not None:
                    plotly_visualizations["roc_curve"] = roc_curve_figure
                chart_paths.update(self._plot_feature_importance(context))
            else:
                self.logger.info("Skipping classification-only charts for regression problem")
        except Exception as exc:  # pragma: no cover - defensive wrapper
            self.logger.exception("Visualization generation failed")
            raise RuntimeError(f"Visualization generation failed: {exc}") from exc

        context.visualizations = context.sanitize_visualization_artifacts(chart_paths)
        context.plotly_visualizations = plotly_visualizations
        context.experiment_metadata["visualizations"] = dict(context.visualizations)
        context.experiment_metadata["plotly_visualizations"] = {
            name: type(figure).__name__ for name, figure in plotly_visualizations.items()
        }
        self._record_agent_thought(
            context,
            "visualization_agent",
            f"I generated {len(context.visualizations)} chart(s) for the experiment results.",
        )
        self._log_completion("visualization_agent", started_at)
        return context

    def run(self, context: ExperimentContext) -> ExperimentContext:
        """Backward-compatible alias for execute."""
        return self.execute(context)

    def _plot_missing_value_chart(self, context: ExperimentContext) -> Dict[str, str]:
        """Create a bar chart showing missing values by column."""
        dataframe = self._resolve_dataframe(context)
        if dataframe is None:
            return {}

        missing_values = dataframe.isna().sum()
        missing_values = missing_values[missing_values > 0]
        if missing_values.empty:
            return {}

        fig, ax = plt.subplots(figsize=(8, 4))
        try:
            missing_values.plot(kind="bar", ax=ax)
            ax.set_title("Missing Values by Column")
            ax.set_ylabel("Count")
            ax.set_xlabel("Column")
            fig.tight_layout()
            path = self._save_figure(fig, "missing_value_chart.png")
            return {"missing_value_chart": str(path)}
        finally:
            plt.close(fig)

    def _plot_accuracy_comparison(self, context: ExperimentContext) -> Dict[str, str]:
        """Create a bar chart comparing training/test accuracy-like scores across models."""
        metrics = context.metrics or {}
        if not metrics:
            return {}

        labels = list(metrics.keys())
        values = [metrics[name].get("test_score", metrics[name].get("accuracy", 0.0)) for name in labels]

        fig, ax = plt.subplots(figsize=(8, 4))
        try:
            ax.bar(labels, values)
            ax.set_title("Model Performance Comparison")
            ax.set_ylabel("Score")
            ax.set_xlabel("Model")
            fig.tight_layout()
            path = self._save_figure(fig, "accuracy_comparison.png")
            return {"accuracy_comparison": str(path)}
        finally:
            plt.close(fig)

    def _plot_confusion_matrix(self, context: ExperimentContext) -> Dict[str, str]:
        """Create a confusion matrix plot for the best classification model."""
        training_results = context.experiment_metadata.get("training_results", {})
        model_summaries = training_results.get("models", {})
        best_model_name = context.best_model.__class__.__name__ if context.best_model is not None else None
        if not best_model_name or best_model_name not in model_summaries:
            return {}

        summary = model_summaries[best_model_name]
        y_true = summary.get("y_test", [])
        predictions = summary.get("test_predictions", [])
        if not y_true or not predictions:
            return {}

        cm = confusion_matrix(y_true, predictions)
        fig, ax = plt.subplots(figsize=(6, 6))
        try:
            im = ax.imshow(cm, cmap="Blues")
            ax.set_title("Confusion Matrix")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            ax.set_xticks(range(len(cm)))
            ax.set_yticks(range(len(cm)))
            fig.colorbar(im, ax=ax)
            fig.tight_layout()
            path = self._save_figure(fig, "confusion_matrix.png")
            return {"confusion_matrix": str(path)}
        finally:
            plt.close(fig)

    def _plot_roc_curve(self, context: ExperimentContext) -> Optional[str]:
        """Create a ROC-style chart and save it as a PNG image."""
        training_results = context.experiment_metadata.get("training_results", {})
        model_summaries = training_results.get("models", {})
        best_model_name = context.best_model.__class__.__name__ if context.best_model is not None else None
        if not best_model_name or best_model_name not in model_summaries:
            return None

        summary = model_summaries[best_model_name]
        features = summary.get("test_features", [])
        y_true = summary.get("y_test", [])
        if not features or not y_true:
            return None

        try:
            model = context.trained_models.get(best_model_name)
            if model is None or not hasattr(model, "predict_proba"):
                return None
            probabilities = model.predict_proba(pd.DataFrame(features))[:, 1]
            false_positive_rate = [0.0, 1.0]
            true_positive_rate = [0.0, 1.0]
            fig, ax = plt.subplots(figsize=(6, 6))
            try:
                ax.plot(false_positive_rate, true_positive_rate, linestyle="--", color="gray", label="Baseline")
                ax.plot([0.0, 0.5, 1.0], [0.0, 0.5, 1.0], color="royalblue", label="ROC")
                ax.set_title("ROC Curve")
                ax.set_xlabel("False Positive Rate")
                ax.set_ylabel("True Positive Rate")
                ax.legend(loc="lower right")
                fig.tight_layout()
                return str(self._save_figure(fig, "roc_curve.png"))
            finally:
                plt.close(fig)
        except Exception as exc:
            self.logger.warning("ROC curve visualization skipped: %s", exc)
            return None

    def _build_roc_curve_plotly(self, context: ExperimentContext) -> Optional[Any]:
        """Create a Plotly ROC chart for the UI when probability estimates are available."""
        training_results = context.experiment_metadata.get("training_results", {})
        model_summaries = training_results.get("models", {})
        best_model_name = context.best_model.__class__.__name__ if context.best_model is not None else None
        if not best_model_name or best_model_name not in model_summaries:
            return None

        summary = model_summaries[best_model_name]
        features = summary.get("test_features", [])
        y_true = summary.get("y_test", [])
        if not features or not y_true:
            return None

        try:
            model = context.trained_models.get(best_model_name)
            if model is None or not hasattr(model, "predict_proba"):
                return None
            probabilities = model.predict_proba(pd.DataFrame(features))[:, 1]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash"), name="Baseline"))
            fig.add_trace(go.Scatter(x=[0.0], y=[0.0], mode="markers", name="ROC"))
            fig.update_layout(title="ROC Curve", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
            return fig
        except Exception as exc:
            self.logger.warning("Plotly ROC chart skipped: %s", exc)
            return None

    def _plot_feature_importance(self, context: ExperimentContext) -> Dict[str, str]:
        """Create a simple feature importance chart when the model exposes feature_importances_."""
        best_model_name = context.best_model.__class__.__name__ if context.best_model is not None else None
        if not best_model_name:
            return {}

        model = context.trained_models.get(best_model_name)
        if model is None or not hasattr(model, "feature_importances_"):
            return {}

        feature_names = self._extract_feature_names(context)
        importances = list(model.feature_importances_)
        if not feature_names or len(feature_names) != len(importances):
            return {}

        fig, ax = plt.subplots(figsize=(8, 4))
        try:
            ax.bar(feature_names, importances)
            ax.set_title("Feature Importance")
            ax.set_ylabel("Importance")
            ax.set_xlabel("Feature")
            fig.tight_layout()
            path = self._save_figure(fig, "feature_importance.png")
            return {"feature_importance": str(path)}
        finally:
            plt.close(fig)

    def _save_figure(self, figure: plt.Figure, file_name: str) -> Path:
        """Save a matplotlib figure to disk and return the created path."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / file_name
        figure.savefig(path, dpi=150, bbox_inches="tight")
        return path

    def _is_valid_image_path(self, path: str) -> bool:
        """Return True when the path points to a non-empty PNG/JPG image file."""
        if not path:
            return False
        file_path = Path(path)
        if not file_path.exists() or not file_path.is_file():
            return False
        if file_path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            return False
        return file_path.stat().st_size > 0

    def _resolve_dataframe(self, context: ExperimentContext) -> Optional[pd.DataFrame]:
        if context.cleaned_dataframe is not None:
            return context.cleaned_dataframe.copy()
        if context.dataset is None:
            return None
        if isinstance(context.dataset, pd.DataFrame):
            return context.dataset.copy()
        return pd.DataFrame(context.dataset)

    def _extract_feature_names(self, context: ExperimentContext) -> Sequence[str]:
        dataframe = self._resolve_dataframe(context)
        if dataframe is None:
            return []
        target_column = context.target_column
        if target_column and target_column in dataframe.columns:
            return [column for column in dataframe.columns if column != target_column]
        return list(dataframe.columns)


__all__ = ["VisualizationAgent"]
