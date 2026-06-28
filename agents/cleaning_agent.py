from __future__ import annotations

<<<<<<< HEAD
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from agents.base_agent import BaseAgent
from core.context import ExperimentContext


class DataCleaningAgent(BaseAgent):
    """Prepare a dataset for downstream modeling and record every transformation.

    The agent removes duplicates, imputes missing values using dataset-appropriate
    strategies, encodes categorical features, standardizes numeric features, and
    stores both the cleaned dataframe and a detailed preprocessing summary in the
    shared experiment context.
    """

    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        self.logger = logger or logging.getLogger(__name__)

    def execute(self, context: ExperimentContext) -> ExperimentContext:
        """Clean the dataset referenced by the context and update its state."""
        started_at = self._log_execution(context, "cleaning_agent")
        try:
            dataframe = self._resolve_dataframe(context)
            self.logger.info("Starting preprocessing for %s rows and %s columns", dataframe.shape[0], dataframe.shape[1])

            cleaned_dataframe = dataframe.copy()
            transformed_columns: List[str] = []
            summary_steps: List[Dict[str, Any]] = []

            cleaned_dataframe, duplicate_summary = self._remove_duplicates(cleaned_dataframe)
            summary_steps.append(duplicate_summary)

            cleaned_dataframe, missing_summary = self._handle_missing_values(cleaned_dataframe, context.target_column)
            summary_steps.append(missing_summary)

            cleaned_dataframe, encoding_summary = self._encode_categorical_columns(cleaned_dataframe, context.target_column)
            summary_steps.append(encoding_summary)

            cleaned_dataframe, scaling_summary = self._scale_numeric_columns(cleaned_dataframe, context.target_column)
            summary_steps.append(scaling_summary)

            context.cleaned_dataframe = cleaned_dataframe
            preprocessing_summary = self._build_summary(
                original_shape=dataframe.shape,
                final_shape=cleaned_dataframe.shape,
                target_column=context.target_column,
                summary_steps=summary_steps,
                transformed_columns=transformed_columns,
            )
            context.preprocessing_summary = preprocessing_summary
            context.experiment_metadata["preprocessing_summary"] = preprocessing_summary
            context.experiment_metadata["preprocessing_completed_at"] = self._utc_timestamp()
            self._record_agent_thought(
                context,
                "cleaning_agent",
                "I removed duplicates and filled missing values using dataset-appropriate strategies.",
            )
            self._log_completion("cleaning_agent", started_at)
            self.logger.info("Preprocessing completed successfully")
            return context
        except Exception as exc:  # pragma: no cover - defensive wrapper
            self.logger.exception("Preprocessing failed")
            raise RuntimeError(f"Data cleaning failed: {exc}") from exc

    def run(self, context: ExperimentContext) -> ExperimentContext:
        """Backward-compatible alias for execute."""
        return self.execute(context)

    def _resolve_dataframe(self, context: ExperimentContext) -> pd.DataFrame:
        """Resolve the dataframe to preprocess from the experiment context."""
        if context.cleaned_dataframe is not None:
            return context.cleaned_dataframe.copy()
        if context.dataset is None:
            raise ValueError("ExperimentContext must contain dataset or cleaned_dataframe")
        if isinstance(context.dataset, pd.DataFrame):
            return context.dataset.copy()
        return pd.DataFrame(context.dataset)

    def _remove_duplicates(self, dataframe: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Remove duplicate rows and record the transformation summary."""
        rows_before = int(dataframe.shape[0])
        cleaned_dataframe = dataframe.drop_duplicates().reset_index(drop=True)
        rows_removed = rows_before - int(cleaned_dataframe.shape[0])
        return cleaned_dataframe, {
            "step": "remove_duplicates",
            "applied": True,
            "rows_before": rows_before,
            "rows_after": int(cleaned_dataframe.shape[0]),
            "rows_removed": rows_removed,
            "details": f"Removed {rows_removed} duplicate row(s).",
        }

    def _handle_missing_values(self, dataframe: pd.DataFrame, target_column: Optional[str]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Impute missing values using median for numeric data and mode for categorical data."""
        cleaned_dataframe = dataframe.copy()
        excluded_columns = {target_column} if target_column else set()
        imputation_details: List[Dict[str, Any]] = []

        for column in cleaned_dataframe.columns:
            if column in excluded_columns:
                continue
            series = cleaned_dataframe[column]
            if series.isna().sum() == 0:
                continue

            if pd.api.types.is_numeric_dtype(series):
                fill_value = float(series.median())
                cleaned_dataframe[column] = series.fillna(fill_value)
                imputation_details.append({
                    "column": column,
                    "strategy": "median",
                    "fill_value": fill_value,
                })
            else:
                mode_value = series.mode(dropna=True)
                fill_value = str(mode_value.iloc[0]) if not mode_value.empty else "__missing__"
                cleaned_dataframe[column] = series.fillna(fill_value)
                imputation_details.append({
                    "column": column,
                    "strategy": "most_frequent",
                    "fill_value": fill_value,
                })

        return cleaned_dataframe, {
            "step": "handle_missing_values",
            "applied": True,
            "details": imputation_details,
            "summary": f"Imputed missing values for {len(imputation_details)} column(s).",
        }

    def _encode_categorical_columns(self, dataframe: pd.DataFrame, target_column: Optional[str]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """One-hot encode low-cardinality categoricals and label encode high-cardinality ones."""
        cleaned_dataframe = dataframe.copy()
        excluded_columns = {target_column} if target_column else set()
        encoding_details: List[Dict[str, Any]] = []

        for column in [col for col in cleaned_dataframe.columns if col not in excluded_columns]:
            series = cleaned_dataframe[column]
            if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
                continue

            unique_values = series.dropna().nunique()
            if unique_values <= 10:
                encoded = pd.get_dummies(cleaned_dataframe[[column]], prefix=column, dtype=int)
                cleaned_dataframe = pd.concat([cleaned_dataframe.drop(columns=[column]), encoded], axis=1)
                encoding_details.append({
                    "column": column,
                    "method": "one_hot",
                    "created_columns": list(encoded.columns),
                })
            else:
                encoded_values, uniques = pd.factorize(series.astype(str), sort=True)
                cleaned_dataframe[column] = encoded_values
                encoding_details.append({
                    "column": column,
                    "method": "label_encode",
                    "unique_values": int(len(uniques)),
                })

        return cleaned_dataframe, {
            "step": "encode_categorical_columns",
            "applied": bool(encoding_details),
            "details": encoding_details,
            "summary": f"Encoded {len(encoding_details)} categorical column(s).",
        }

    def _scale_numeric_columns(self, dataframe: pd.DataFrame, target_column: Optional[str]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Standardize numeric columns using z-score normalization."""
        cleaned_dataframe = dataframe.copy()
        excluded_columns = {target_column} if target_column else set()
        scaling_details: List[Dict[str, Any]] = []

        for column in [col for col in cleaned_dataframe.columns if col not in excluded_columns]:
            series = cleaned_dataframe[column]
            if not pd.api.types.is_numeric_dtype(series):
                continue

            mean_value = float(series.mean())
            std_value = float(series.std(ddof=0))
            if std_value == 0:
                scaling_details.append({
                    "column": column,
                    "method": "standard_scale",
                    "status": "skipped",
                    "reason": "zero variance",
                })
                continue

            cleaned_dataframe[column] = (series - mean_value) / std_value
            scaling_details.append({
                "column": column,
                "method": "standard_scale",
                "mean": mean_value,
                "std": std_value,
            })

        return cleaned_dataframe, {
            "step": "scale_numeric_columns",
            "applied": bool(scaling_details),
            "details": scaling_details,
            "summary": f"Scaled {len(scaling_details)} numeric column(s).",
        }

    def _build_summary(
        self,
        original_shape: Tuple[int, int],
        final_shape: Tuple[int, int],
        target_column: Optional[str],
        summary_steps: Sequence[Dict[str, Any]],
        transformed_columns: Sequence[str],
    ) -> Dict[str, Any]:
        """Construct a human-readable summary of the preprocessing pipeline."""
        return {
            "original_shape": {"rows": original_shape[0], "columns": original_shape[1]},
            "final_shape": {"rows": final_shape[0], "columns": final_shape[1]},
            "target_column": target_column,
            "transformed_columns": list(transformed_columns),
            "steps": list(summary_steps),
        }

    @staticmethod
    def _utc_timestamp() -> str:
        return datetime.now(timezone.utc).isoformat()


__all__ = ["DataCleaningAgent"]
=======
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
>>>>>>> origin/main
