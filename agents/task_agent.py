from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import pandas as pd

from agents.base_agent import BaseAgent
from core.context import ExperimentContext, ProblemType


class TaskDetectionAgent(BaseAgent):
    """Infer whether an experiment should be treated as classification or regression.

    The agent inspects the target column in the experiment context and chooses a
    task type using a simple heuristic:
    - classification when the target is categorical or has a small number of unique values
    - regression otherwise

    The choice and rationale are persisted in the shared experiment context so
    downstream agents can use the same task assumption.
    """

    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        self.logger = logger or logging.getLogger(__name__)

    def execute(self, context: ExperimentContext) -> ExperimentContext:
        """Determine the task type for the current experiment context."""
        started_at = self._log_execution(context, "task_agent")
        target_series = self._resolve_target(context)
        task_type, explanation = self._infer_task(target_series)

        context.problem_type = task_type
        context.experiment_metadata["task_detection"] = {
            "task_type": task_type,
            "explanation": explanation,
        }
        self._record_agent_thought(
            context,
            "task_agent",
            f"I selected {task_type} because the target column suggests that task type.",
        )
        self._log_completion("task_agent", started_at)
        self.logger.info("Detected task type '%s' for the experiment", task_type)
        return context

    def run(self, context: ExperimentContext) -> ExperimentContext:
        """Backward-compatible alias for execute."""
        return self.execute(context)

    def _resolve_target(self, context: ExperimentContext) -> Optional[pd.Series]:
        """Return the target column as a pandas Series when available."""
        dataframe = context.cleaned_dataframe if context.cleaned_dataframe is not None else context.dataset
        if dataframe is None:
            raise ValueError("ExperimentContext must contain dataset or cleaned_dataframe")

        if not isinstance(dataframe, pd.DataFrame):
            dataframe = pd.DataFrame(dataframe)

        target_column = context.target_column
        if not target_column:
            if "target" in dataframe.columns:
                target_column = "target"
            else:
                raise ValueError("No target column is available in the experiment context")

        if target_column not in dataframe.columns:
            raise ValueError(f"Target column '{target_column}' is not present in the dataset")

        return dataframe[target_column]

    def _infer_task(self, target_series: Optional[pd.Series]) -> tuple[ProblemType, str]:
        """Infer the task type and explanation from the target column."""
        if target_series is None:
            raise ValueError("Target series could not be resolved")

        non_null_values = target_series.dropna()
        if non_null_values.empty:
            return "classification", "The target column is empty, so classification was selected by default."

        unique_count = int(non_null_values.nunique(dropna=True))
        dtype = non_null_values.dtype
        is_object_like = pd.api.types.is_object_dtype(dtype) or pd.api.types.is_string_dtype(dtype) or pd.api.types.is_categorical_dtype(dtype)

        if is_object_like or unique_count <= 10:
            explanation = (
                f"The target column appears categorical because it has {unique_count} unique values "
                f"and/or object-like dtype ({dtype})."
            )
            return "classification", explanation

        if pd.api.types.is_numeric_dtype(dtype):
            explanation = (
                f"The target column appears numeric with {unique_count} unique values and dtype {dtype}, "
                "so regression was selected."
            )
            return "regression", explanation

        explanation = (
            f"The target column appears numeric with {unique_count} unique values and dtype {dtype}, "
            "so regression was selected."
        )
        return "regression", explanation


__all__ = ["TaskDetectionAgent"]
