"""
Dataset Analysis UI Component

Provides a production-ready Streamlit component to inspect and summarize
an uploaded dataset. Keeps UI logic separated from business logic and
stores analysis results in Streamlit session state for later agents.
"""
from typing import Dict, Optional

import pandas as pd
import streamlit as st

from agents.data_inspector import DataInspector
from ui.uploads_handler import UploadHandler


class DatasetAnalysis:
    """
    Renders dataset analysis UI and computes analysis artifacts.

    Methods
    -------
    render() -> Optional[Dict]:
        Render the analysis UI and return the analysis report dictionary.
    """

    def __init__(self) -> None:
        self.inspector = DataInspector()

    def _compute_additional_metrics(self, df: pd.DataFrame, target: Optional[str]) -> Dict:
        missing_per_column = df.isnull().sum().to_dict()
        duplicate_count = int(df.duplicated().sum())
        memory_safe_head = df.head(100)
        describe = df.describe(include='all').transpose().fillna('')

        target_distribution = None
        if target and target in df.columns:
            try:
                target_distribution = df[target].value_counts(dropna=False).to_dict()
            except Exception:
                target_distribution = None

        return {
            "missing_per_column": missing_per_column,
            "duplicate_count": duplicate_count,
            "head_preview": memory_safe_head,
            "describe": describe,
            "target_distribution": target_distribution,
        }

    def render(self) -> Optional[Dict]:
        """
        Render the dataset analysis UI.

        Returns
        -------
        Optional[Dict]
            Analysis report dictionary stored in session state under
            `st.session_state['analysis_report']` when available. Returns
            None if no dataset is present.
        """
        st.subheader("🔎 Dataset Analysis")

        df, target = UploadHandler.get_dataset_from_session()

        if df is None:
            st.warning("No dataset available. Upload a dataset first in 'Upload Dataset'.")
            return None

        # Use DataInspector for baseline report
        with st.spinner("Running quick dataset inspection..."):
            base_report = self.inspector.inspect(df)

        # Present top-level metrics
        c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
        c1.metric("Rows", base_report.get("rows", "--"))
        c2.metric("Columns", base_report.get("columns", "--"))
        c3.metric("Missing Values", base_report.get("missing_values", "--"))
        c4.metric("Duplicate Rows", base_report.get("duplicate_rows", "--"))

        # Data types and column listing
        with st.expander("View data types and columns", expanded=False):
            st.write("**Data types**")
            st.json(base_report.get("data_types", {}))

            st.write("**Numeric columns**")
            st.write(base_report.get("numeric_columns", []))

            st.write("**Categorical columns**")
            st.write(base_report.get("categorical_columns", []))

        # Additional metrics and visuals
        extras = self._compute_additional_metrics(df, target)

        # Missing values chart
        with st.expander("Missing values per column", expanded=True):
            missing = pd.Series(extras["missing_per_column"])  # type: ignore
            if missing.sum() == 0:
                st.success("No missing values detected.")
            else:
                st.bar_chart(missing.sort_values(ascending=False))
                st.dataframe(missing.sort_values(ascending=False).to_frame("missing_count"))

        # Duplicate rows
        with st.expander("Duplicate rows", expanded=False):
            dup_count = extras["duplicate_count"]
            st.write(f"Duplicate rows detected: **{dup_count}**")
            if dup_count > 0:
                st.dataframe(df[df.duplicated(keep=False)].head(100))

        # Basic statistics
        with st.expander("Basic statistics (describe)", expanded=False):
            describe_df = extras["describe"]
            st.dataframe(describe_df)

        # Dataset preview
        with st.expander("Dataset preview (first 100 rows)", expanded=False):
            st.dataframe(extras["head_preview"], use_container_width=True)

        # Target distribution
        if target and extras.get("target_distribution") is not None:
            with st.expander(f"Target distribution: {target}", expanded=True):
                td = pd.Series(extras["target_distribution"])  # type: ignore
                st.bar_chart(td)
                st.dataframe(td.to_frame("count"))

        # Store report in session state for downstream agents
        analysis_report = {
            "base_report": base_report,
            "missing_per_column": extras["missing_per_column"],
            "duplicate_count": extras["duplicate_count"],
            "describe": extras["describe"].to_dict(),
            "target": target,
            "target_distribution": extras.get("target_distribution"),
        }

        st.session_state["analysis_report"] = analysis_report
        st.success("Dataset analysis completed and saved to session state.")

        return analysis_report


# Allow simple invocation when imported
def render_dataset_analysis() -> Optional[Dict]:
    """Convenience function for other modules to call the analysis UI."""
    return DatasetAnalysis().render()
