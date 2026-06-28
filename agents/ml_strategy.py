"""
ML Strategy Agent

Determines whether a prediction task is Classification or Regression
based on the target column characteristics.
"""
from typing import Dict, Any, Optional

import pandas as pd
import numpy as np


class MLStrategy:
    """
    Heuristic-based task type inference for ML experiments.

    Methods
    -------
    determine_task(df, target) -> Dict[str, Any]
        Returns a dictionary with keys: `task_type`, `confidence`, `reason`, `details`.
    """

    def determine_task(self, df: pd.DataFrame, target: str) -> Dict[str, Any]:
        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found in DataFrame")

        series = df[target]
        n_rows = int(series.shape[0])
        n_unique = int(series.nunique(dropna=True))
        unique_ratio = n_unique / max(n_rows, 1)

        dtype = str(series.dtype)
        is_numeric = pd.api.types.is_numeric_dtype(series)
        is_integer_like = False
        if is_numeric:
            # Check whether values are integer-like
            is_integer_like = np.all(np.mod(series.dropna().values, 1) == 0)

        # Heuristic scoring
        score_class = 0.5
        score_reg = 0.5
        reasons = []

        if not is_numeric:
            score_class += 0.4
            reasons.append("Target is non-numeric (categorical/strings) -> favors classification")
        else:
            if is_integer_like:
                # integer-like numeric: likely categorical if few unique values
                if n_unique <= 20:
                    score_class += 0.4
                    reasons.append("Integer-like numeric with low cardinality -> favors classification")
                else:
                    score_reg += 0.3
                    reasons.append("Integer-like numeric with higher cardinality -> leans regression")
            else:
                # float-like numeric
                if n_unique / max(n_rows, 1) > 0.05 and n_unique > 30:
                    score_reg += 0.5
                    reasons.append("Continuous numeric with many unique values -> favors regression")
                else:
                    score_class += 0.2
                    reasons.append("Numeric but low cardinality -> may be classification")

        # Additional signals
        if n_unique == 2:
            score_class += 0.3
            reasons.append("Binary target detected -> classification")

        # Final decision
        final_task = "classification" if score_class >= score_reg else "regression"
        confidence = float(max(score_class, score_reg) / (score_class + score_reg))

        # Normalize confidence to [0.5, 0.99] range for visibility
        confidence = max(0.5, min(confidence, 0.99))

        details: Dict[str, Any] = {
            "dtype": dtype,
            "n_rows": n_rows,
            "n_unique": n_unique,
            "unique_ratio": unique_ratio,
            "is_numeric": bool(is_numeric),
            "is_integer_like": bool(is_integer_like),
        }

        return {
            "task_type": final_task,
            "confidence": round(confidence, 3),
            "reason": "; ".join(reasons),
            "details": details,
        }


# Convenience function
def infer_task(df: pd.DataFrame, target: str) -> Dict[str, Any]:
    return MLStrategy().determine_task(df, target)
