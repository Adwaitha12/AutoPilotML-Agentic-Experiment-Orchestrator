from __future__ import annotations

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
