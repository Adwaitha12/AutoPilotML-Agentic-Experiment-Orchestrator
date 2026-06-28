from __future__ import annotations

import html
<<<<<<< HEAD
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
=======
from pathlib import Path
from typing import Any

import pandas as pd

from core.context import ExperimentContext
from utils.helpers import ensure_directory, safe_filename


class ReportAgent:
    """Generate professional HTML and PDF reports for the completed experiment."""

    def execute(self, context: ExperimentContext) -> ExperimentContext:
        output_dir = ensure_directory("outputs/reports")
        report_path = output_dir / f"{safe_filename(context.experiment_id)}.html"
        report_path.write_text(self._build_html(context), encoding="utf-8")

        pdf_path = output_dir / f"{safe_filename(context.experiment_id)}.pdf"
        self._write_pdf(context, pdf_path)

        context.report_path = str(report_path)
        context.html_report_path = str(report_path)
        context.add_artifact("html_report", str(report_path))
        if pdf_path.exists():
            context.pdf_report_path = str(pdf_path)
            context.add_artifact("pdf_report", str(pdf_path))

        return context

    def _build_html(self, context: ExperimentContext) -> str:
        charts = "".join(
            f"<section><h2>{self._title(name)}</h2>"
            f"<img src='{html.escape(self._relative_chart_path(path))}' alt='{html.escape(name)}' />"
            f"</section>"
            for name, path in context.charts.items()
        )

        return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>AgentLab AI Experiment Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 0; color: #172033; background: #f8fafc; }}
    main {{ max-width: 1040px; margin: 0 auto; background: #ffffff; padding: 48px; }}
    .cover {{ border-bottom: 1px solid #e2e8f0; padding-bottom: 28px; margin-bottom: 28px; }}
    .eyebrow {{ color: #2563eb; font-weight: 700; text-transform: uppercase; letter-spacing: .06em; }}
    h1 {{ font-size: 40px; margin: 8px 0; color: #0f172a; }}
    h2 {{ color: #0f172a; margin-top: 36px; border-bottom: 1px solid #e2e8f0; padding-bottom: 8px; }}
    h3 {{ color: #1e293b; }}
    .grid {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; }}
    .card {{ border: 1px solid #e2e8f0; border-radius: 8px; padding: 14px; background: #ffffff; }}
    .label {{ color: #64748b; font-size: 12px; }}
    .value {{ font-size: 20px; font-weight: 700; color: #0f172a; margin-top: 6px; }}
    table {{ border-collapse: collapse; width: 100%; margin: 12px 0 24px; font-size: 13px; }}
    th, td {{ border-bottom: 1px solid #e2e8f0; padding: 10px; text-align: left; vertical-align: top; }}
    th {{ background: #f1f5f9; color: #334155; }}
    img {{ max-width: 100%; border: 1px solid #e2e8f0; border-radius: 8px; background: #fff; }}
    .callout {{ border-left: 4px solid #2563eb; background: #eff6ff; padding: 14px; border-radius: 6px; }}
  </style>
</head>
<body>
<main>
  <section class="cover">
    <div class="eyebrow">AgentLab AI</div>
    <h1>Machine Learning Experiment Report</h1>
    <p>{html.escape(context.business_goal or "Automated ML experiment")} · {html.escape(context.dataset_name)}</p>
  </section>

  <section>
    <h2>Executive Summary</h2>
    <p>{html.escape(self._executive_summary(context))}</p>
    <div class="grid">{self._summary_cards(context)}</div>
  </section>

  <section>
    <h2>Dataset Overview</h2>
    {self._dataset_table(context)}
  </section>

  <section>
    <h2>Data Quality</h2>
    {self._quality_table(context)}
  </section>

  <section>
    <h2>Cleaning Summary</h2>
    {self._cleaning_table(context)}
  </section>

  <section>
    <h2>Problem Information</h2>
    {self._problem_table(context)}
  </section>

  <section>
    <h2>Model Leaderboard</h2>
    {self._leaderboard_table(context)}
  </section>

  <section>
    <h2>Best Model</h2>
    <div class="callout">{html.escape(self._best_model_explanation(context))}</div>
  </section>

  <section>
    <h2>Evaluation Metrics</h2>
    {self._best_metrics_table(context)}
  </section>

  <section>
    <h2>Visualizations</h2>
    {charts or "<p>No charts were generated for this experiment.</p>"}
  </section>

  <section>
    <h2>Critic Review</h2>
    {self._critic_review(context)}
  </section>

  <section>
    <h2>Business Recommendation</h2>
    <p>{html.escape(self._business_recommendation(context))}</p>
  </section>

  <section>
    <h2>Future Improvements</h2>
    <ul>
      <li>Run cross-validation before production use.</li>
      <li>Validate feature definitions with domain experts.</li>
      <li>Monitor model drift after deployment.</li>
    </ul>
  </section>

  <section>
    <h2>Appendix</h2>
    <p>Experiment ID: {html.escape(context.experiment_id)}</p>
    <p>Generated By: Planner Agent and specialized AgentLab AI workflow agents.</p>
  </section>
</main>
>>>>>>> origin/main
</body>
</html>
"""

<<<<<<< HEAD
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
=======
    def _summary_cards(self, context: ExperimentContext) -> str:
        cards = {
            "Problem Type": self._title(context.problem_type or "Pending"),
            "Best Model": self._title((context.best_model or "Pending").replace("_", " ")),
            "Best Score": self._metric(context.best_model_score),
            "Report": "HTML + PDF",
        }
        return "".join(
            f"<div class='card'><div class='label'>{html.escape(label)}</div><div class='value'>{html.escape(value)}</div></div>"
            for label, value in cards.items()
        )

    def _dataset_table(self, context: ExperimentContext) -> str:
        dataframe = context.dataset
        rows = {
            "Dataset Name": context.dataset_name,
            "Rows": f"{dataframe.shape[0]:,}",
            "Columns": f"{dataframe.shape[1]:,}",
            "Target Column": context.target_column,
            "Memory Usage": self._format_bytes(dataframe.memory_usage(deep=True).sum()),
            "Numerical Features": str(len(dataframe.select_dtypes(include="number").columns)),
            "Categorical Features": str(dataframe.shape[1] - len(dataframe.select_dtypes(include="number").columns)),
        }
        return self._key_value_table(rows)

    def _quality_table(self, context: ExperimentContext) -> str:
        report = context.quality_report or {}
        target = report.get("target", {})
        rows = {
            "Missing Values": str(report.get("missing_values_total", 0)),
            "Duplicate Rows": str(report.get("duplicate_rows", 0)),
            "Target Missing Values": str(target.get("missing_values", 0)),
            "Target Unique Values": str(target.get("unique_values", 0)),
        }
        return self._key_value_table(rows)

    def _cleaning_table(self, context: ExperimentContext) -> str:
        rows = context.cleaning_changes[:20]
        if not rows:
            return "<p>No cleaning changes were recorded.</p>"
        return self._table(rows)

    def _problem_table(self, context: ExperimentContext) -> str:
        rows = {
            "Problem Type": self._title(context.problem_type or "Pending"),
            "Target Column": context.target_column,
            "Optimization Goal": context.optimization_goal.replace("_", " ").title(),
            "Test Size": f"{context.test_size:.0%}",
            "Business Goal": context.business_goal or "Not Provided",
        }
        return self._key_value_table(rows)

    def _leaderboard_table(self, context: ExperimentContext) -> str:
        leaderboard = context.metadata.get("leaderboard", [])
        if not leaderboard:
            return "<p>No leaderboard was generated.</p>"
        rows = []
        for index, item in enumerate(leaderboard, start=1):
            row = {"Rank": index, "Model": self._title(str(item.get("model", "")).replace("_", " "))}
            row.update({self._title(key.replace("_", " ")): self._metric(value) for key, value in item.items() if key != "model"})
            rows.append(row)
        return self._table(rows)

    def _best_metrics_table(self, context: ExperimentContext) -> str:
        if not context.best_model:
            return "<p>No best model was selected.</p>"
        metrics = context.metrics.get(context.best_model, {})
        rows = {
            self._title(key.replace("_", " ")): self._metric(value)
            for key, value in metrics.items()
        }
        return self._key_value_table(rows)

    def _critic_review(self, context: ExperimentContext) -> str:
        if not context.critic_comments:
            return "<p>No critic comments were generated.</p>"
        return "<ul>" + "".join(f"<li>{html.escape(comment)}</li>" for comment in context.critic_comments) + "</ul>"

    def _executive_summary(self, context: ExperimentContext) -> str:
        if not context.best_model:
            return "The experiment was initialized but did not complete model evaluation."
        return (
            f"AgentLab AI evaluated {len(context.selected_models)} candidate models for "
            f"{context.business_goal or 'the selected target'} and selected "
            f"{context.best_model.replace('_', ' ')} as the best current baseline."
        )

    def _best_model_explanation(self, context: ExperimentContext) -> str:
        if not context.best_model:
            return "No best model was selected."
        metric_name = context.metadata.get("primary_metric", "score")
        return (
            f"{context.best_model.replace('_', ' ').title()} was selected because it achieved "
            f"the strongest {metric_name.replace('_', ' ')} result under the chosen optimization goal."
        )

    def _business_recommendation(self, context: ExperimentContext) -> str:
        if context.critic_analysis:
            return str(context.critic_analysis.get("business_recommendation", ""))
        return "Use this model as a baseline and validate with domain-specific acceptance criteria."

    def _table(self, rows: list[dict[str, Any]]) -> str:
        if not rows:
            return ""
        columns = list(rows[0].keys())
        header = "".join(f"<th>{html.escape(str(column))}</th>" for column in columns)
        body = ""
        for row in rows:
            body += "<tr>" + "".join(
                f"<td>{html.escape(self._stringify(row.get(column, '')))}</td>"
                for column in columns
            ) + "</tr>"
        return f"<table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table>"

    def _key_value_table(self, rows: dict[str, Any]) -> str:
        return self._table([{"Metric": key, "Value": value} for key, value in rows.items()])

    def _relative_chart_path(self, chart_path: str) -> str:
        try:
            return str(Path("..") / Path(chart_path).relative_to("outputs"))
        except ValueError:
            return chart_path

    def _write_pdf(self, context: ExperimentContext, pdf_path: Path) -> None:
        try:
            from reportlab.lib import colors
            from reportlab.lib.pagesizes import letter
            from reportlab.lib.styles import getSampleStyleSheet
            from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
        except ImportError:
            return

        styles = getSampleStyleSheet()
        document = SimpleDocTemplate(str(pdf_path), pagesize=letter)
        story: list[Any] = [
            Paragraph("AgentLab AI Experiment Report", styles["Title"]),
            Paragraph(self._executive_summary(context), styles["BodyText"]),
            Spacer(1, 16),
            Paragraph("Dataset Overview", styles["Heading2"]),
            self._pdf_key_value_table(
                {
                    "Dataset": context.dataset_name,
                    "Target": context.target_column,
                    "Problem Type": self._title(context.problem_type or "Pending"),
                    "Best Model": self._title((context.best_model or "Pending").replace("_", " ")),
                }
            ),
            Spacer(1, 12),
            Paragraph("Model Leaderboard", styles["Heading2"]),
            self._pdf_table(context.metadata.get("leaderboard", [])[:6]),
            Spacer(1, 12),
            Paragraph("Critic Review", styles["Heading2"]),
        ]

        for comment in context.critic_comments:
            story.append(Paragraph(f"- {comment}", styles["BodyText"]))

        story.extend(
            [
                Spacer(1, 12),
                Paragraph("Business Recommendation", styles["Heading2"]),
                Paragraph(self._business_recommendation(context), styles["BodyText"]),
                Spacer(1, 12),
                Paragraph("Generated Visualizations", styles["Heading2"]),
            ]
        )
        for name in context.charts:
            story.append(Paragraph(self._title(name.replace("_", " ")), styles["BodyText"]))

        for element in story:
            if isinstance(element, Table):
                element.setStyle(
                    TableStyle(
                        [
                            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e2e8f0")),
                            ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cbd5e1")),
                            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                            ("FONTSIZE", (0, 0), (-1, -1), 8),
                        ]
                    )
                )

        document.build(story)

    def _pdf_key_value_table(self, rows: dict[str, Any]) -> Any:
        from reportlab.platypus import Table

        return Table([["Metric", "Value"], *[[key, str(value)] for key, value in rows.items()]])

    def _pdf_table(self, rows: list[dict[str, Any]]) -> Any:
        from reportlab.platypus import Table

        if not rows:
            return Table([["Result"], ["No data"]])
        columns = list(rows[0].keys())[:6]
        return Table(
            [columns]
            + [[self._stringify(row.get(column, ""))[:30] for column in columns] for row in rows]
        )

    def _metric(self, value: Any) -> str:
        if value is None:
            return "Pending"
        if isinstance(value, (int, float)):
            if -1 <= value <= 1:
                return f"{value:.2%}"
            return f"{value:.4f}"
        return str(value)

    def _format_bytes(self, size: int) -> str:
        units = ["B", "KB", "MB", "GB"]
        value = float(size)
        for unit in units:
            if value < 1024:
                return f"{value:.1f} {unit}"
            value /= 1024
        return f"{value:.1f} TB"

    def _stringify(self, value: Any) -> str:
        text = str(value)
        return text if len(text) <= 160 else f"{text[:157]}..."

    def _title(self, value: str) -> str:
        return value.title()
>>>>>>> origin/main
