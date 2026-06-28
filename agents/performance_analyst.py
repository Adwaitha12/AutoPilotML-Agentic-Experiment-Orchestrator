"""
Performance Analyst Agent

Compares model results produced by the Experiment Agent and selects the
best model based on task-specific primary metrics. Provides detailed
leaderboard and recommendations.
"""
from typing import Dict, Any, List, Tuple, Optional


class PerformanceAnalyst:
    """Analyze experiment results and pick the best model."""

    def analyze(self, experiment_results: Dict[str, Any], task_type: str) -> Dict[str, Any]:
        """
        Args:
            experiment_results: output from ExperimentAgent.train_models()
            task_type: 'classification' or 'regression'

        Returns:
            Dict with keys: 'leaderboard' (sorted), 'best_model' (dict), 'recommendation' (str)
        """
        leaderboard = experiment_results.get("leaderboard", [])

        if not leaderboard:
            return {"leaderboard": [], "best_model": None, "recommendation": "No results to analyze."}

        # Define scoring/comparison logic
        def score_entry(entry: Dict[str, Any]) -> Tuple:
            metrics = entry.get("metrics", {})
            if task_type == "classification":
                # Primary: accuracy (higher), secondary: f1 (higher), tertiary: roc_auc (higher)
                return (
                    metrics.get("accuracy", 0),
                    metrics.get("f1", 0),
                    metrics.get("roc_auc", 0),
                )
            else:
                # Regression: primary mse (lower -> invert), secondary r2 (higher)
                mse = metrics.get("mse", float("inf"))
                r2 = metrics.get("r2", -float("inf"))
                # We'll return (-mse, r2) so that larger tuple is better
                return (-mse if mse is not None else float("-inf"), r2)

        # Sort entries by score tuple descending
        sorted_lb = sorted(leaderboard, key=lambda e: score_entry(e), reverse=True)

        best = sorted_lb[0]

        # Build recommendation
        recs: List[str] = []
        if task_type == "classification":
            acc = best.get("metrics", {}).get("accuracy")
            f1 = best.get("metrics", {}).get("f1")
            roc = best.get("metrics", {}).get("roc_auc")
            recs.append(f"Best model: {best.get('name')} (accuracy={acc}, f1={f1})")
            if roc is not None:
                recs.append(f"ROC AUC: {roc}")
        else:
            mse = best.get("metrics", {}).get("mse")
            r2 = best.get("metrics", {}).get("r2")
            recs.append(f"Best model: {best.get('name')} (mse={mse}, r2={r2})")

        # Check for warnings (overfitting signal: large train vs test gap not available here)
        # If any entries had errors, note them
        errors = [e for e in leaderboard if e.get("error")]
        if errors:
            recs.append(f"{len(errors)} models failed during training. See model entries for details.")

        return {"leaderboard": sorted_lb, "best_model": best, "recommendation": " | ".join(recs)}


# Convenience

def analyze_experiment(experiment_results: Dict[str, Any], task_type: str) -> Dict[str, Any]:
    return PerformanceAnalyst().analyze(experiment_results, task_type)
