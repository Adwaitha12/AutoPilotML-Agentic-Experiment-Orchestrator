from __future__ import annotations

from core.context import ExperimentContext
from utils.preprocessing import basic_clean_dataframe


class CleaningAgent:
    """Perform lightweight, model-safe dataframe cleanup."""

    def execute(self, context: ExperimentContext) -> ExperimentContext:
        duplicate_rows = context.dataset[context.dataset.duplicated(keep="first")]
        empty_columns = [
            column for column in context.dataset.columns if context.dataset[column].isna().all()
        ]
        cleaned, summary = basic_clean_dataframe(context.dataset)

        if context.target_column not in cleaned.columns:
            raise ValueError("Target column was removed during cleaning.")

        missing_target_rows = cleaned[cleaned[context.target_column].isna()]
        rows_before_target_drop = int(cleaned.shape[0])
        cleaned = cleaned.dropna(subset=[context.target_column]).reset_index(drop=True)

        summary["rows_removed_missing_target"] = rows_before_target_drop - int(
            cleaned.shape[0]
        )
        summary["missing_values_remaining"] = int(cleaned.isna().sum().sum())

        context.clean_dataframe = cleaned
        context.cleaning_summary = summary
        context.cleaning_changes = self._build_change_log(
            duplicate_rows=duplicate_rows,
            empty_columns=empty_columns,
            missing_target_rows=missing_target_rows,
        )
        context.feature_columns = [
            column for column in cleaned.columns if column != context.target_column
        ]
        return context

    def _build_change_log(
        self,
        duplicate_rows,
        empty_columns: list[str],
        missing_target_rows,
    ) -> list[dict[str, object]]:
        changes: list[dict[str, object]] = []

        for row_index, row in duplicate_rows.head(25).iterrows():
            changes.append(
                {
                    "change": "Removed Duplicate Row",
                    "row_index": int(row_index),
                    "column": "All Columns",
                    "before": row.to_dict(),
                    "after": "Row Removed",
                }
            )

        for column in empty_columns:
            changes.append(
                {
                    "change": "Removed Empty Column",
                    "row_index": "All Rows",
                    "column": column,
                    "before": "All values missing",
                    "after": "Column Removed",
                }
            )

        for row_index, row in missing_target_rows.head(25).iterrows():
            changes.append(
                {
                    "change": "Removed Missing Target Row",
                    "row_index": int(row_index),
                    "column": "Target",
                    "before": row.to_dict(),
                    "after": "Row Removed",
                }
            )

        if not changes:
            changes.append(
                {
                    "change": "No Destructive Cleaning Required",
                    "row_index": "-",
                    "column": "-",
                    "before": "Dataset already structurally valid",
                    "after": "Dataset preserved",
                }
            )

        return changes
