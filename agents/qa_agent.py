from __future__ import annotations

<<<<<<< HEAD
from datetime import datetime, timezone
import logging
from typing import Any, Dict, Optional

import pandas as pd

from agents.base_agent import BaseAgent
from core.context import ExperimentContext


class DataQualityAgent(BaseAgent):
    """Assess the quality of a dataset and record the findings in context.

    The agent inspects the dataframe provided through the experiment context and
    produces a structured report covering missing values, duplicate rows,
    column types, dimensions, and target distribution. The report is stored in
    the shared context so that downstream agents can use it without re-running
    the analysis.
    """

    def __init__(self, target_column: Optional[str] = None, logger: Optional[logging.Logger] = None) -> None:
        self.target_column = target_column
        self.logger = logger or logging.getLogger(__name__)

    def execute(self, context: ExperimentContext) -> ExperimentContext:
        """Analyze the dataset referenced by the context and update it."""
        started_at = self._log_execution(context, "qa_agent")
        dataframe = self._resolve_dataframe(context)
        report = self._build_report(dataframe, context)
        context.quality_report = report
        context.cleaned_dataframe = dataframe.copy()
        context.experiment_metadata["quality_report_generated_at"] = self._utc_timestamp()
        missing_count = report.get("missing_values", {}).get("total_missing_values", 0)
        duplicate_count = report.get("duplicate_rows", {}).get("total_duplicate_rows", 0)
        self._record_agent_thought(
            context,
            "qa_agent",
            f"I detected {missing_count} missing values and {duplicate_count} duplicate rows.",
        )
        self._log_completion("qa_agent", started_at)
        return context

    def run(self, context: ExperimentContext) -> ExperimentContext:
        """Backward-compatible alias for execute."""
        return self.execute(context)

    def _resolve_dataframe(self, context: ExperimentContext) -> pd.DataFrame:
        """Return the dataframe to inspect from the experiment context."""
        if context.cleaned_dataframe is not None:
            return context.cleaned_dataframe.copy()

        if context.dataset is None:
            raise ValueError("ExperimentContext must contain dataset or cleaned_dataframe")

        if isinstance(context.dataset, pd.DataFrame):
            return context.dataset.copy()

        return pd.DataFrame(context.dataset)

    def _build_report(self, dataframe: pd.DataFrame, context: ExperimentContext) -> Dict[str, Any]:
        """Construct a structured quality report from the dataframe."""
        target_column = self._resolve_target_column(context, dataframe)
        missing_values = self._analyze_missing_values(dataframe)
        duplicate_rows = self._analyze_duplicate_rows(dataframe)
        column_types = self._analyze_column_types(dataframe)
        dimensions = self._analyze_dimensions(dataframe)
        target_distribution = self._analyze_target_distribution(dataframe, target_column)

        return {
            "summary": {
                "has_missing_values": missing_values["total_missing_values"] > 0,
                "has_duplicate_rows": duplicate_rows["total_duplicate_rows"] > 0,
                "target_column_available": bool(target_column and target_column in dataframe.columns),
                "row_count": dimensions["rows"],
                "column_count": dimensions["columns"],
            },
            "missing_values": missing_values,
            "duplicate_rows": duplicate_rows,
            "column_types": column_types,
            "dimensions": dimensions,
            "target_distribution": target_distribution,
        }

    def _resolve_target_column(self, context: ExperimentContext, dataframe: pd.DataFrame) -> Optional[str]:
        """Determine the target column to inspect for class balance or value distribution."""
        if context.target_column:
            return context.target_column
        if self.target_column:
            return self.target_column
        if "target" in dataframe.columns:
            return "target"
        return None

    def _analyze_missing_values(self, dataframe: pd.DataFrame) -> Dict[str, Any]:
        """Summarize missing values by column and overall totals."""
        missing_counts = dataframe.isna().sum()
        missing_by_column = {
            column: int(count) for column, count in missing_counts.items() if int(count) > 0
        }
        return {
            "total_missing_values": int(missing_counts.sum()),
            "by_column": missing_by_column,
            "columns_with_missing_values": list(missing_by_column.keys()),
        }

    def _analyze_duplicate_rows(self, dataframe: pd.DataFrame) -> Dict[str, Any]:
        """Count duplicate rows and report whether duplicates were found."""
        duplicate_count = int(dataframe.duplicated().sum())
        return {
            "total_duplicate_rows": duplicate_count,
            "has_duplicate_rows": duplicate_count > 0,
        }

    def _analyze_column_types(self, dataframe: pd.DataFrame) -> Dict[str, Any]:
        """Describe the dtype of each column in the dataset."""
        return {
            column: {
                "dtype": str(dtype),
                "is_numeric": bool(pd.api.types.is_numeric_dtype(dtype)),
                "is_categorical": bool(pd.api.types.is_object_dtype(dtype) or pd.api.types.is_categorical_dtype(dtype)),
            }
            for column, dtype in dataframe.dtypes.items()
        }

    def _analyze_dimensions(self, dataframe: pd.DataFrame) -> Dict[str, Any]:
        """Return the current row and column counts."""
        return {"rows": int(dataframe.shape[0]), "columns": int(dataframe.shape[1])}

    def _analyze_target_distribution(self, dataframe: pd.DataFrame, target_column: Optional[str]) -> Dict[str, Any]:
        """Inspect the distribution of values in the target column when available."""
        if not target_column or target_column not in dataframe.columns:
            return {"available": False, "message": "target column not available"}

        values = dataframe[target_column].dropna()
        if values.empty:
            return {"available": True, "counts": {}, "message": "target column has no non-null values"}

        counts = values.value_counts(dropna=False).to_dict()
        return {
            "available": True,
            "counts": {str(key): int(value) for key, value in counts.items()},
            "unique_values": int(values.nunique()),
        }

    @staticmethod
    def _utc_timestamp() -> str:
        return datetime.now(timezone.utc).isoformat()


__all__ = ["DataQualityAgent"]
=======
from typing import Any

from core.context import ExperimentContext


class QAAgent:
    """Inspect dataset health before modeling starts."""

    def execute(self, context: ExperimentContext) -> ExperimentContext:
        dataframe = context.dataset
        target = context.target_column

        report: dict[str, Any] = {
            "shape": {"rows": int(dataframe.shape[0]), "columns": int(dataframe.shape[1])},
            "duplicate_rows": int(dataframe.duplicated().sum()),
            "missing_values_total": int(dataframe.isna().sum().sum()),
            "missing_values_by_column": {
                column: int(count)
                for column, count in dataframe.isna().sum().items()
                if int(count) > 0
            },
            "data_types": {
                column: str(dtype) for column, dtype in dataframe.dtypes.items()
            },
            "target": {
                "column": target,
                "missing_values": int(dataframe[target].isna().sum()),
                "unique_values": int(dataframe[target].nunique(dropna=True)),
            },
        }

        if dataframe[target].nunique(dropna=True) <= 20:
            report["target"]["class_distribution"] = {
                str(label): int(count)
                for label, count in dataframe[target].value_counts(dropna=False).items()
            }

        context.quality_report = report
        return context
>>>>>>> origin/main
