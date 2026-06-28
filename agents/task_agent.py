from __future__ import annotations

from pandas.api.types import is_numeric_dtype

from core.context import ExperimentContext


class TaskAgent:
    """Detect whether the dataset should be modeled as classification or regression."""

    def execute(self, context: ExperimentContext) -> ExperimentContext:
        target_series = context.active_dataset[context.target_column]
        unique_values = int(target_series.nunique(dropna=True))

        if not is_numeric_dtype(target_series):
            problem_type = "classification"
        elif unique_values <= 20 or unique_values / max(len(target_series), 1) < 0.05:
            problem_type = "classification"
        else:
            problem_type = "regression"

        context.problem_type = problem_type
        context.metadata["task_detection"] = {
            "target_unique_values": unique_values,
            "target_dtype": str(target_series.dtype),
            "reason": self._reason(problem_type, unique_values, str(target_series.dtype)),
        }
        return context

    def _reason(self, problem_type: str, unique_values: int, dtype: str) -> str:
        if problem_type == "classification":
            return f"Target has {unique_values} distinct values with dtype {dtype}."
        return f"Target appears continuous with {unique_values} distinct numeric values."
