"""
Results UI Component

Displays experiment results, metrics, charts, recommendations, and
downloaded artifacts from a completed experiment.
"""
from typing import Dict, Any, Optional
import streamlit as st
import pandas as pd
import json


class ResultsUI:
    """Render completed experiment results."""

    @staticmethod
    def render_results(experiment: Dict[str, Any]) -> None:
        """Display results from an experiment dict."""
        if not experiment:
            st.warning("No experiment results available.")
            return

        # Metadata
        st.subheader("📋 Experiment Metadata")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Experiment Name", experiment.get("experiment_name", "--"))
        with col2:
            st.metric("KB ID", experiment.get("kb_id", "--"))
        with col3:
            st.metric("Task Type", experiment.get("strategy", {}).get("task_type", "--").title())

        st.divider()

        # Strategy
        st.subheader("🎯 Strategy")
        strat = experiment.get("strategy", {})
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Detected Task", strat.get("task_type", "--").title())
        with col2:
            st.metric("Confidence", f"{strat.get('confidence', 0):.1%}")
        st.write(f"**Reasoning:** {strat.get('reason', 'N/A')}")

        st.divider()

        # Performance
        st.subheader("🏆 Best Model")
        perf = experiment.get("performance", {})
        best = perf.get("best_model", {})
        if best:
            st.write(f"**Model:** {best.get('name', 'Unknown')}")
            st.write(f"**Metrics:**")
            st.json(best.get("metrics", {}))
        else:
            st.warning("No best model found.")

        st.divider()

        # Leaderboard
        st.subheader("📊 Model Leaderboard")
        leaderboard = experiment.get("experiment_results", {}).get("leaderboard", [])
        if leaderboard:
            lb_data = []
            for entry in leaderboard:
                lb_data.append({
                    "Model": entry.get("name", "Unknown"),
                    "Metrics": json.dumps(entry.get("metrics", {})),
                })
            lb_df = pd.DataFrame(lb_data)
            st.dataframe(lb_df, use_container_width=True)
        else:
            st.info("No leaderboard data available.")

        st.divider()

        # Review findings
        st.subheader("🧠 AI Reviewer Insights")
        review = experiment.get("review", {})
        findings = review.get("findings", [])
        recommendations = review.get("recommendations", [])

        if findings:
            st.write("**Findings:**")
            for f in findings:
                st.write(f"- {f}")
        else:
            st.info("No findings detected.")

        if recommendations:
            st.write("**Recommendations:**")
            for r in recommendations:
                st.write(f"- {r}")

        st.divider()

        # Reports
        st.subheader("📄 Reports")
        report_paths = experiment.get("report_paths", {})
        html_path = report_paths.get("html")
        pdf_path = report_paths.get("pdf")

        if html_path:
            try:
                with open(html_path, "r", encoding="utf-8") as f:
                    html_data = f.read()
                st.download_button(
                    label="📥 Download HTML Report",
                    data=html_data,
                    file_name="experiment_report.html",
                    mime="text/html",
                )
            except Exception as e:
                st.warning(f"Could not load HTML report: {e}")

        if pdf_path:
            try:
                with open(pdf_path, "rb") as f:
                    pdf_bytes = f.read()
                st.download_button(
                    label="📥 Download PDF Report",
                    data=pdf_bytes,
                    file_name="experiment_report.pdf",
                    mime="application/pdf",
                )
            except Exception as e:
                st.info(f"PDF report generated but could not load for download: {e}")
