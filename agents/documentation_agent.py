"""
Documentation Agent

Generates a professional HTML report summarizing the experiment:
- Dataset Summary
- Cleaning Summary
- Model Comparison (table + charts)
- Winner and metrics
- Recommendations

Also attempts to generate a PDF using ReportLab if installed.
"""
from typing import Dict, Any, Optional
import os
import json
import pandas as pd


class DocumentationAgent:
    def __init__(self, output_dir: str = "outputs/reports") -> None:
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def generate_html_report(self, analysis_report: Dict[str, Any], experiment_results: Dict[str, Any], performance_analysis: Dict[str, Any], insights_summary: Dict[str, Any]) -> str:
        title = "MLForge AI Experiment Report"
        html_parts = [f"<html><head><meta charset='utf-8'><title>{title}</title></head><body>"]
        html_parts.append(f"<h1>{title}</h1>")

        # Dataset summary
        base = analysis_report.get('base_report', {})
        html_parts.append("<h2>Dataset Summary</h2>")
        html_parts.append(f"<p>Rows: {base.get('rows', '--')}, Columns: {base.get('columns', '--')}</p>")
        html_parts.append("<h3>Data types</h3>")
        html_parts.append(f"<pre>{json.dumps(base.get('data_types', {}), indent=2)}</pre>")

        # Cleaning summary
        html_parts.append("<h2>Cleaning Summary</h2>")
        html_parts.append(f"<p>Missing values per column: <pre>{json.dumps(analysis_report.get('missing_per_column', {}), indent=2)}</pre></p>")
        html_parts.append(f"<p>Duplicate rows removed: {analysis_report.get('duplicate_count', 0)}</p>")

        # Model comparison
        html_parts.append("<h2>Model Comparison</h2>")
        html_parts.append("<table border='1' cellpadding='5' cellspacing='0'>")
        html_parts.append("<tr><th>Model</th><th>Metrics</th></tr>")
        for entry in experiment_results.get('leaderboard', []):
            html_parts.append(f"<tr><td>{entry.get('name')}</td><td><pre>{json.dumps(entry.get('metrics', {}), indent=2)}</pre></td></tr>")
        html_parts.append("</table>")

        # Winner & recommendation
        html_parts.append("<h2>Winner</h2>")
        best = performance_analysis.get('best_model')
        if best:
            html_parts.append(f"<p><strong>{best.get('name')}</strong></p>")
            html_parts.append(f"<pre>{json.dumps(best.get('metrics', {}), indent=2)}</pre>")

        html_parts.append("<h2>AI Review Recommendations</h2>")
        for r in analysis_report.get('recommendations', []):
            html_parts.append(f"<p>{r}</p>")

        html_parts.append("</body></html>")

        html_content = "\n".join(html_parts)
        out_path = os.path.join(self.output_dir, "experiment_report.html")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        return out_path

    def generate_pdf_report(self, html_path: str) -> Optional[str]:
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.pdfgen import canvas
            from reportlab.lib.utils import ImageReader
            # Very simple PDF conversion: embed the HTML as text (not full rendering)
            pdf_path = os.path.join(self.output_dir, "experiment_report.pdf")
            c = canvas.Canvas(pdf_path, pagesize=letter)
            with open(html_path, 'r', encoding='utf-8') as f:
                html_text = f.read()
            # Truncate for now: write the HTML as plain text into PDF
            textobject = c.beginText(40, 750)
            for i, line in enumerate(html_text.splitlines()[:200]):
                textobject.textLine(line[:90])
            c.drawText(textobject)
            c.save()
            return pdf_path
        except Exception:
            return None

    def generate_report(self, analysis_report: Dict[str, Any], experiment_results: Dict[str, Any], performance_analysis: Dict[str, Any], insights_summary: Dict[str, Any]) -> Dict[str, Any]:
        html_path = self.generate_html_report(analysis_report, experiment_results, performance_analysis, insights_summary)
        pdf_path = self.generate_pdf_report(html_path)
        return {"html": html_path, "pdf": pdf_path}
