from __future__ import annotations

<<<<<<< HEAD
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
=======
import html
from collections import Counter

import pandas as pd
from sklearn.metrics import roc_curve

from core.context import ExperimentContext
from utils.helpers import ensure_directory, safe_filename


class VisualizationAgent:
    """Create chart artifacts for the experiment report and dashboard."""

    def execute(self, context: ExperimentContext) -> ExperimentContext:
        output_dir = ensure_directory("outputs/charts")
        self._plot_leaderboard(context, output_dir)
        self._plot_missing_values(context, output_dir)
        self._plot_target_distribution(context, output_dir)
        self._plot_feature_importance(context, output_dir)
        self._plot_correlation_heatmap(context, output_dir)

        if context.problem_type == "classification" and context.best_model:
            self._plot_confusion_matrix(context, output_dir)
            self._plot_roc_curve(context, output_dir)

        return context

    def _plot_leaderboard(self, context: ExperimentContext, output_dir: object) -> None:
        leaderboard = context.metadata.get("leaderboard", [])
        if not leaderboard:
            return

        metric_name = context.metadata.get("primary_metric", "score")
        frame = pd.DataFrame(leaderboard)
        path = output_dir / f"{safe_filename(context.experiment_id)}_leaderboard.svg"
        values = list(frame[metric_name])
        labels = [str(model).replace("_", " ") for model in frame["model"]]

        path.write_text(
            self._bar_chart_svg(labels, values, f"Model Leaderboard by {metric_name}"),
            encoding="utf-8",
        )
        context.add_chart("leaderboard", str(path))

    def _plot_missing_values(self, context: ExperimentContext, output_dir: object) -> None:
        missing = context.dataset.isna().sum()
        missing = missing[missing > 0].sort_values(ascending=False).head(12)
        if missing.empty:
            return

        path = output_dir / f"{safe_filename(context.experiment_id)}_missing_values.svg"
        path.write_text(
            self._bar_chart_svg(
                [str(label) for label in missing.index],
                [float(value) for value in missing.values],
                "Missing Values by Column",
            ),
            encoding="utf-8",
        )
        context.add_chart("missing_values", str(path))

    def _plot_target_distribution(
        self,
        context: ExperimentContext,
        output_dir: object,
    ) -> None:
        target_counts = context.active_dataset[context.target_column].value_counts().head(12)
        if target_counts.empty:
            return

        path = output_dir / f"{safe_filename(context.experiment_id)}_target_distribution.svg"
        path.write_text(
            self._bar_chart_svg(
                [str(label) for label in target_counts.index],
                [float(value) for value in target_counts.values],
                "Target Distribution",
            ),
            encoding="utf-8",
        )
        context.add_chart("target_distribution", str(path))

    def _plot_feature_importance(
        self,
        context: ExperimentContext,
        output_dir: object,
    ) -> None:
        if not context.best_model:
            return

        rows = context.feature_importances.get(context.best_model, [])
        if not rows:
            return

        path = output_dir / f"{safe_filename(context.experiment_id)}_feature_importance.svg"
        path.write_text(
            self._horizontal_bar_chart_svg(
                [str(row["feature"]) for row in rows],
                [float(row["contribution"]) for row in rows],
                "Top Feature Importance",
            ),
            encoding="utf-8",
        )
        context.add_chart("feature_importance", str(path))

    def _plot_correlation_heatmap(
        self,
        context: ExperimentContext,
        output_dir: object,
    ) -> None:
        numeric = context.active_dataset.select_dtypes(include="number")
        if numeric.shape[1] < 2:
            return

        correlation = numeric.corr().fillna(0).round(2)
        path = output_dir / f"{safe_filename(context.experiment_id)}_correlation_heatmap.svg"
        path.write_text(
            self._correlation_heatmap_svg(correlation),
            encoding="utf-8",
        )
        context.add_chart("correlation_heatmap", str(path))

    def _plot_confusion_matrix(self, context: ExperimentContext, output_dir: object) -> None:
        y_test = context.metadata.get("y_test")
        predictions = context.predictions.get(context.best_model or "")
        if y_test is None or predictions is None:
            return

        path = output_dir / f"{safe_filename(context.experiment_id)}_confusion_matrix.svg"
        path.write_text(
            self._confusion_matrix_svg(y_test, predictions),
            encoding="utf-8",
        )
        context.add_chart("confusion_matrix", str(path))

    def _plot_roc_curve(self, context: ExperimentContext, output_dir: object) -> None:
        if not context.best_model:
            return

        y_test = context.metadata.get("y_test")
        probabilities = context.probabilities.get(context.best_model)
        if y_test is None or probabilities is None:
            return

        try:
            false_positive_rate, true_positive_rate, _ = roc_curve(y_test, probabilities)
        except ValueError:
            return

        path = output_dir / f"{safe_filename(context.experiment_id)}_roc_curve.svg"
        path.write_text(
            self._line_chart_svg(
                [float(value) for value in false_positive_rate],
                [float(value) for value in true_positive_rate],
                "ROC Curve",
                "False Positive Rate",
                "True Positive Rate",
            ),
            encoding="utf-8",
        )
        context.add_chart("roc_curve", str(path))

    def _bar_chart_svg(
        self,
        labels: list[str],
        values: list[float],
        title: str,
    ) -> str:
        width = 820
        height = 420
        chart_left = 72
        chart_top = 64
        chart_width = 680
        chart_height = 240
        max_value = max(values) if values else 1
        bar_gap = 14
        bar_width = max(24, int((chart_width - bar_gap * len(values)) / max(len(values), 1)))

        bars = []
        for index, (label, value) in enumerate(zip(labels, values)):
            x = chart_left + index * (bar_width + bar_gap)
            bar_height = 0 if max_value == 0 else int((value / max_value) * chart_height)
            y = chart_top + chart_height - bar_height
            bars.append(
                f"<rect x='{x}' y='{y}' width='{bar_width}' height='{bar_height}' fill='#2563eb' rx='3' />"
                f"<text x='{x + bar_width / 2}' y='{chart_top + chart_height + 28}' text-anchor='middle' font-size='11'>{html.escape(label[:16])}</text>"
                f"<text x='{x + bar_width / 2}' y='{y - 8}' text-anchor='middle' font-size='11'>{value:.3f}</text>"
            )

        return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="#ffffff"/>
<text x="32" y="36" font-family="Arial" font-size="22" font-weight="700" fill="#102a43">{html.escape(title)}</text>
<line x1="{chart_left}" y1="{chart_top + chart_height}" x2="{chart_left + chart_width}" y2="{chart_top + chart_height}" stroke="#bcccdc"/>
<line x1="{chart_left}" y1="{chart_top}" x2="{chart_left}" y2="{chart_top + chart_height}" stroke="#bcccdc"/>
{''.join(bars)}
</svg>"""

    def _horizontal_bar_chart_svg(
        self,
        labels: list[str],
        values: list[float],
        title: str,
    ) -> str:
        width = 820
        height = max(360, 80 + len(values) * 34)
        left = 240
        top = 64
        chart_width = 520
        max_value = max(values) if values else 1
        rows = []

        for index, (label, value) in enumerate(zip(labels, values)):
            y = top + index * 34
            bar_width = 0 if max_value == 0 else int((value / max_value) * chart_width)
            rows.append(
                f"<text x='32' y='{y + 18}' font-size='12' fill='#243b53'>{html.escape(label[:28])}</text>"
                f"<rect x='{left}' y='{y}' width='{bar_width}' height='22' fill='#0f766e' rx='4' />"
                f"<text x='{left + bar_width + 8}' y='{y + 16}' font-size='12'>{value:.1%}</text>"
            )

        return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="#ffffff"/>
<text x="32" y="36" font-family="Arial" font-size="22" font-weight="700" fill="#102a43">{html.escape(title)}</text>
{''.join(rows)}
</svg>"""

    def _line_chart_svg(
        self,
        x_values: list[float],
        y_values: list[float],
        title: str,
        x_label: str,
        y_label: str,
    ) -> str:
        width = 640
        height = 420
        left = 72
        top = 56
        chart_width = 480
        chart_height = 280
        points = []

        for x_value, y_value in zip(x_values, y_values):
            x = left + x_value * chart_width
            y = top + chart_height - y_value * chart_height
            points.append(f"{x:.1f},{y:.1f}")

        return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="#ffffff"/>
<text x="32" y="32" font-family="Arial" font-size="22" font-weight="700" fill="#102a43">{html.escape(title)}</text>
<line x1="{left}" y1="{top + chart_height}" x2="{left + chart_width}" y2="{top + chart_height}" stroke="#bcccdc"/>
<line x1="{left}" y1="{top}" x2="{left}" y2="{top + chart_height}" stroke="#bcccdc"/>
<polyline points="{' '.join(points)}" fill="none" stroke="#7c3aed" stroke-width="3"/>
<line x1="{left}" y1="{top + chart_height}" x2="{left + chart_width}" y2="{top}" stroke="#cbd5e1" stroke-dasharray="5 5"/>
<text x="{left + chart_width / 2}" y="{height - 28}" text-anchor="middle" font-size="12">{html.escape(x_label)}</text>
<text x="18" y="{top + chart_height / 2}" transform="rotate(-90 18,{top + chart_height / 2})" text-anchor="middle" font-size="12">{html.escape(y_label)}</text>
</svg>"""

    def _correlation_heatmap_svg(self, correlation: pd.DataFrame) -> str:
        labels = [str(label)[:12] for label in correlation.columns]
        cell = 54
        left = 160
        top = 88
        width = left + len(labels) * cell + 60
        height = top + len(labels) * cell + 80
        cells = []

        for row_index, row_label in enumerate(correlation.index):
            for column_index, column_label in enumerate(correlation.columns):
                value = float(correlation.loc[row_label, column_label])
                color = self._correlation_color(value)
                x = left + column_index * cell
                y = top + row_index * cell
                cells.append(
                    f"<rect x='{x}' y='{y}' width='{cell}' height='{cell}' fill='{color}' stroke='#ffffff'/>"
                    f"<text x='{x + cell / 2}' y='{y + cell / 2 + 4}' text-anchor='middle' font-size='11'>{value:.2f}</text>"
                )

        axis_labels = []
        for index, label in enumerate(labels):
            axis_labels.append(
                f"<text x='{left + index * cell + cell / 2}' y='{top - 12}' text-anchor='middle' font-size='10'>{html.escape(label)}</text>"
                f"<text x='{left - 12}' y='{top + index * cell + cell / 2 + 4}' text-anchor='end' font-size='10'>{html.escape(label)}</text>"
            )

        return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="#ffffff"/>
<text x="32" y="36" font-family="Arial" font-size="22" font-weight="700" fill="#102a43">Correlation Heatmap</text>
{''.join(axis_labels)}
{''.join(cells)}
</svg>"""

    def _correlation_color(self, value: float) -> str:
        if value >= 0:
            intensity = int(245 - min(value, 1.0) * 130)
            return f"rgb({intensity},{intensity},255)"
        intensity = int(245 - min(abs(value), 1.0) * 130)
        return f"rgb(255,{intensity},{intensity})"

    def _confusion_matrix_svg(self, y_true: list[object], y_pred: list[object]) -> str:
        labels = sorted({str(value) for value in y_true} | {str(value) for value in y_pred})
        counts = Counter((str(actual), str(predicted)) for actual, predicted in zip(y_true, y_pred))
        max_count = max(counts.values()) if counts else 1
        cell = 80
        left = 150
        top = 80
        width = left + cell * len(labels) + 48
        height = top + cell * len(labels) + 80
        cells = []

        for row, actual in enumerate(labels):
            for column, predicted in enumerate(labels):
                count = counts.get((actual, predicted), 0)
                intensity = int(235 - (count / max_count) * 150) if max_count else 235
                x = left + column * cell
                y = top + row * cell
                cells.append(
                    f"<rect x='{x}' y='{y}' width='{cell}' height='{cell}' fill='rgb({intensity},{intensity},255)' stroke='#334e68'/>"
                    f"<text x='{x + cell / 2}' y='{y + cell / 2 + 5}' text-anchor='middle' font-size='18'>{count}</text>"
                )

        axis_labels = []
        for index, label in enumerate(labels):
            axis_labels.append(
                f"<text x='{left + index * cell + cell / 2}' y='{top - 18}' text-anchor='middle' font-size='12'>Pred {html.escape(label)}</text>"
                f"<text x='{left - 16}' y='{top + index * cell + cell / 2 + 5}' text-anchor='end' font-size='12'>Actual {html.escape(label)}</text>"
            )

        return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="#ffffff"/>
<text x="32" y="36" font-family="Arial" font-size="22" font-weight="700" fill="#102a43">Confusion Matrix</text>
{''.join(axis_labels)}
{''.join(cells)}
</svg>"""
>>>>>>> origin/main
