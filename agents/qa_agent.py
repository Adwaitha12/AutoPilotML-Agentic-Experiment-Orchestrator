from __future__ import annotations

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
