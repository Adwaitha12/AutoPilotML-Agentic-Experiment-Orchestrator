"""
Experiment Agent

Trains provided models on cleaned data, evaluates performance, saves trained models,
and returns a leaderboard with metrics and paths to saved models.
"""
from typing import List, Dict, Any, Tuple
import os
import joblib

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, mean_squared_error, r2_score


class ExperimentAgent:
    def __init__(self, output_dir: str = "outputs/models") -> None:
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def _evaluate_classification(self, y_true, y_pred, y_proba=None) -> Dict[str, float]:
        res = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        }
        # roc_auc when probability estimates exist and binary classification
        try:
            if y_proba is not None and y_proba.shape[1] == 2:
                res["roc_auc"] = float(roc_auc_score(y_true, y_proba[:, 1]))
        except Exception:
            pass
        return res

    def _evaluate_regression(self, y_true, y_pred) -> Dict[str, float]:
        return {
            "mse": float(mean_squared_error(y_true, y_pred)),
            "r2": float(r2_score(y_true, y_pred)),
        }

    def train_models(self, models: List[Dict[str, Any]], X: pd.DataFrame, y: pd.Series, task_type: str, test_size: float = 0.2, random_state: int = 42) -> Dict[str, Any]:
        """
        Train and evaluate a list of models.

        Args:
            models: list of {"name": str, "estimator": sklearn estimator}
            X: preprocessed feature DataFrame
            y: target Series
            task_type: 'classification'|'regression'

        Returns:
            Dict with keys: 'leaderboard' (list sorted by primary metric), 'models' (detailed per-model info)
        """
        results = []

        # Train / test split
        if isinstance(X, pd.DataFrame):
            X_vals = X.values
        else:
            X_vals = X
        y_vals = y.values if isinstance(y, (pd.Series, pd.DataFrame)) else y

        if len(y_vals) < 2:
            raise ValueError("Not enough samples to split for training and testing")

        X_train, X_test, y_train, y_test = train_test_split(X_vals, y_vals, test_size=test_size, random_state=random_state)

        for spec in models:
            name = spec.get("name")
            est = spec.get("estimator")
            model_path = os.path.join(self.output_dir, f"{name.replace(' ', '_')}.joblib")

            try:
                est_clone = est
                est_clone.fit(X_train, y_train)

                if task_type == "classification":
                    y_pred = est_clone.predict(X_test)
                    y_proba = None
                    try:
                        y_proba = est_clone.predict_proba(X_test)
                    except Exception:
                        y_proba = None
                    metrics = self._evaluate_classification(y_test, y_pred, y_proba)
                else:
                    y_pred = est_clone.predict(X_test)
                    metrics = self._evaluate_regression(y_test, y_pred)

                # Save model
                joblib.dump(est_clone, model_path)

                results.append({
                    "name": name,
                    "model_path": model_path,
                    "metrics": metrics,
                })

            except Exception as e:
                results.append({
                    "name": name,
                    "model_path": None,
                    "metrics": {},
                    "error": str(e),
                })

        # Sort leaderboard: for classification, sort by accuracy; for regression, sort by mse (lower better)
        if task_type == "classification":
            leaderboard = sorted(results, key=lambda r: r.get("metrics", {}).get("accuracy", 0), reverse=True)
        else:
            leaderboard = sorted(results, key=lambda r: r.get("metrics", {}).get("mse", float("inf")))

        return {"leaderboard": leaderboard, "models": results}


# Convenience function

def run_experiment(models: List[Dict[str, Any]], X: pd.DataFrame, y: pd.Series, task_type: str) -> Dict[str, Any]:
    agent = ExperimentAgent()
    return agent.train_models(models, X, y, task_type)
