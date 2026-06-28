from __future__ import annotations

import html
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from agents.base_agent import BaseAgent
from core.context import ExperimentContext


class ReportAgent(BaseAgent):
    """Generate a professional HTML report for an experiment run."""

    def __init__(self, output_dir: Optional[str] = None, logger: Optional[logging.Logger] = None) -> None:
        self.output_dir = Path(output_dir or "outputs/reports")
        self.logger = logger or logging.getLogger(__name__)

    def execute(self, context: ExperimentContext) -> ExperimentContext:
        """Generate the report and store its path in the experiment context."""
        started_at = self._log_execution(context, "report_agent")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        report_path = self.output_dir / "experiment_report.html"

        html_content = self._build_html(context)
        report_path.write_text(html_content, encoding="utf-8")

        context.report_path = report_path
        context.experiment_metadata["report_path"] = str(report_path)
        self._record_agent_thought(
            context,
            "report_agent",
            "I generated a polished HTML report for the experiment run.",
        )
        self._log_completion("report_agent", started_at)
        self.logger.info("Report generated at %s", report_path)
        return context

    def run(self, context: ExperimentContext) -> ExperimentContext:
        """Backward-compatible alias for execute."""
        return self.execute(context)

    def _build_html(self, context: ExperimentContext) -> str:
        """Construct the report HTML from the experiment context."""
        summary = self._render_summary(context)
        dataset_summary = self._render_dataset_summary(context)
        cleaning_summary = self._render_cleaning_summary(context)
        models_comparison = self._render_models_comparison(context)
        leaderboard = self._render_leaderboard(context)
        best_model = self._render_best_model(context)
        critic_analysis = self._render_critic_analysis(context)
        charts = self._render_charts(context)
        recommendations = self._render_recommendations(context)

        return f"""<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <title>Machine Learning Experiment Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #1f2937; }}
    h1, h2 {{ color: #111827; }}
    .card {{ border: 1px solid #e5e7eb; border-radius: 8px; padding: 16px; margin-bottom: 16px; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 8px; }}
    th, td {{ border: 1px solid #e5e7eb; padding: 8px; text-align: left; }}
    th {{ background: #f3f4f6; }}
    .pill {{ display: inline-block; padding: 4px 8px; border-radius: 999px; background: #eff6ff; color: #1d4ed8; margin-right: 6px; }}
    img {{ max-width: 100%; border: 1px solid #e5e7eb; border-radius: 6px; margin-top: 8px; }}
  </style>
</head>
<body>
  <h1>Machine Learning Experiment Report</h1>
  <div class=\"card\">{summary}</div>
  <div class=\"card\">{dataset_summary}</div>
  <div class=\"card\">{cleaning_summary}</div>
  <div class=\"card\">{models_comparison}</div>
  <div class=\"card\">{leaderboard}</div>
  <div class=\"card\">{best_model}</div>
  <div class=\"card\">{critic_analysis}</div>
  <div class=\"card\">{charts}</div>
  <div class=\"card\">{recommendations}</div>
</body>
</html>
"""

    def _render_summary(self, context: ExperimentContext) -> str:
        problem_type = context.problem_type or "unknown"
        return f"""<h2>Executive Summary</h2><p>This report summarizes an experiment for a {problem_type} task. It includes the data quality assessment, preprocessing steps, model performance, reviewer feedback, and recommended next actions.</p>"""

    def _render_dataset_summary(self, context: ExperimentContext) -> str:
        dataset = self._resolve_dataset(context)
        if dataset is None:
            return "<h2>Dataset Summary</h2><p>No dataset was available.</p>"

        rows, cols = dataset.shape
        target_column = context.target_column or "n/a"
        return f"""<h2>Dataset Summary</h2><p><strong>Rows:</strong> {rows}<br><strong>Columns:</strong> {cols}<br><strong>Target Column:</strong> {html.escape(str(target_column))}</p>"""

    def _render_cleaning_summary(self, context: ExperimentContext) -> str:
        preprocessing_summary = context.experiment_metadata.get("preprocessing_summary", {})
        if not preprocessing_summary:
            return "<h2>Cleaning Summary</h2><p>No preprocessing summary was recorded.</p>"

        steps = preprocessing_summary.get("steps", [])
        items = "".join(f"<li>{html.escape(str(step.get('step', 'step')))}: {html.escape(str(step.get('summary', '')))}</li>" for step in steps)
        return f"""<h2>Cleaning Summary</h2><ul>{items}</ul>"""

    def _render_models_comparison(self, context: ExperimentContext) -> str:
        metrics = context.metrics or {}
        if not metrics:
            return "<h2>Models Compared</h2><p>No model metrics were available.</p>"

        rows = []
        for model_name, metric_values in metrics.items():
            metric_text = " ; ".join(f"{key}={value:.3f}" for key, value in metric_values.items())
            rows.append(f"<tr><td>{html.escape(model_name)}</td><td>{html.escape(metric_text)}</td></tr>")

        return f"""<h2>Models Compared</h2><table><tr><th>Model</th><th>Metrics</th></tr>{''.join(rows)}</table>"""

    def _render_leaderboard(self, context: ExperimentContext) -> str:
        leaderboard = context.experiment_metadata.get("evaluation_results", {}).get("leaderboard", [])
        if not leaderboard:
            return "<h2>Leader Board</h2><p>No leaderboard was generated.</p>"

        rows = []
        for index, entry in enumerate(leaderboard, start=1):
            metrics = entry.get("metrics", {})
            metric_text = " ; ".join(f"{key}={value:.3f}" for key, value in metrics.items())
            rows.append(f"<tr><td>{index}</td><td>{html.escape(entry.get('model_name', 'n/a'))}</td><td>{html.escape(metric_text)}</td></tr>")

        return f"""<h2>Leader Board</h2><table><tr><th>Rank</th><th>Model</th><th>Metrics</th></tr>{''.join(rows)}</table>"""

    def _render_best_model(self, context: ExperimentContext) -> str:
        best_model_name = context.experiment_metadata.get("evaluation_results", {}).get("best_model")
        explanation = context.experiment_metadata.get("best_model_explanation", "No explanation was recorded.")
        if not best_model_name:
            return "<h2>Best Model</h2><p>No best model was selected.</p>"

        return f"""<h2>Best Model</h2><p><strong>{html.escape(str(best_model_name))}</strong></p><p>{html.escape(str(explanation))}</p>"""

    def _render_critic_analysis(self, context: ExperimentContext) -> str:
        critic_analysis = context.critic_analysis or {}
        issues = critic_analysis.get("issues", [])
        recommendations = critic_analysis.get("recommendations", [])

        issues_html = "".join(f"<li>{html.escape(str(issue))}</li>" for issue in issues)
        recommendations_html = "".join(f"<li>{html.escape(str(rec))}</li>" for rec in recommendations)
        return f"""<h2>Critic Analysis</h2><h3>Issues</h3><ul>{issues_html}</ul><h3>Recommendations</h3><ul>{recommendations_html}</ul>"""

    def _render_charts(self, context: ExperimentContext) -> str:
        visualizations = context.visualizations or {}
        if not visualizations:
            return "<h2>Charts</h2><p>No charts were generated.</p>"

        items = []
        for name, path in visualizations.items():
            items.append(f"<li><strong>{html.escape(name)}</strong><br><img src=\"{html.escape(path)}\" alt=\"{html.escape(name)}\"></li>")
        return f"""<h2>Charts</h2><ul>{''.join(items)}</ul>"""

    def _render_recommendations(self, context: ExperimentContext) -> str:
        critic_analysis = context.critic_analysis or {}
        recommendations = critic_analysis.get("recommendations", [])
        if not recommendations:
            return "<h2>Business Recommendations</h2><p>No recommendations were generated.</p>"

        items = "".join(f"<li>{html.escape(str(item))}</li>" for item in recommendations)
        return f"""<h2>Business Recommendations</h2><ul>{items}</ul>"""

    def _resolve_dataset(self, context: ExperimentContext):
        if context.cleaned_dataframe is not None:
            return context.cleaned_dataframe
        if context.dataset is None:
            return None
        if hasattr(context.dataset, "shape"):
            return context.dataset
        return None


__all__ = ["ReportAgent"]
