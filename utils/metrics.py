from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(
    y_true: Any,
    y_pred: Any,
    y_score: Any | None = None,
) -> dict[str, float]:
    """Compute classification metrics used by the leaderboard."""
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(
            precision_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "matthews_corrcoef": float(matthews_corrcoef(y_true, y_pred)),
    }

    if y_score is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
        except ValueError:
            metrics["roc_auc"] = 0.0
    else:
        metrics["roc_auc"] = 0.0

    return metrics


def regression_metrics(y_true: Any, y_pred: Any) -> dict[str, float]:
    """Compute regression metrics used by the leaderboard."""
    mse = float(mean_squared_error(y_true, y_pred))
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(y_true, y_pred)),
        "mape": _safe_mape(y_true, y_pred),
    }


def primary_metric(problem_type: str, optimization_goal: str) -> str:
    """Choose the metric that should decide the winning model."""
    if optimization_goal == "fastest_model":
        return "training_seconds"
    if problem_type == "classification":
        return "f1" if optimization_goal == "most_explainable" else "accuracy"
    return "r2"


def metric_is_higher_better(metric_name: str) -> bool:
    return metric_name not in {"mae", "mse", "rmse", "mape", "training_seconds"}


def _safe_mape(y_true: Any, y_pred: Any) -> float:
    y_true_array = np.asarray(y_true, dtype=float)
    y_pred_array = np.asarray(y_pred, dtype=float)
    non_zero = y_true_array != 0

    if not non_zero.any():
        return 0.0

    return float(
        np.mean(np.abs((y_true_array[non_zero] - y_pred_array[non_zero]) / y_true_array[non_zero]))
    )
