"""
AI Reviewer Agent

Performs heuristic checks on experiment results and the dataset to detect
- class imbalance
- potential leakage
- suspicious accuracy
- insufficient dataset size
- recommendations for next steps
"""
from typing import Dict, Any, List, Optional

import pandas as pd
import numpy as np


class AIReviewer:
    def __init__(self) -> None:
        pass

    def review(self, df: pd.DataFrame, target: str, experiment_results: Dict[str, Any]) -> Dict[str, Any]:
        findings: List[str] = []
        recommendations: List[str] = []

        if target not in df.columns:
            findings.append(f"Target '{target}' not found in dataset.")
            return {"findings": findings, "recommendations": recommendations}

        n_rows = df.shape[0]
        n_cols = df.shape[1]

        # Dataset size
        if n_rows < 50:
            findings.append(f"Dataset is very small ({n_rows} rows). Models may overfit.")
            recommendations.append("Collect more data (recommended > 200 samples) or use simpler models / cross-validation.")
        elif n_rows < 200:
            findings.append(f"Dataset is small ({n_rows} rows). Be cautious about model variance.")
            recommendations.append("Use cross-validation, regularization, and careful feature selection.")

        # Target analysis
        series = df[target]
        is_numeric = pd.api.types.is_numeric_dtype(series)
        n_unique = int(series.nunique(dropna=True))
        unique_ratio = n_unique / max(n_rows, 1)

        # Class imbalance (for classification)
        task_hint = "classification" if not is_numeric or (is_numeric and n_unique <= 20) else "regression"
        if task_hint == "classification":
            value_counts = series.value_counts(dropna=False)
            if len(value_counts) > 0:
                top_frac = float(value_counts.iloc[0]) / max(n_rows, 1)
                if top_frac > 0.9:
                    findings.append(f"Strong class imbalance: top class is {top_frac:.1%} of samples.")
                    recommendations.append("Consider resampling, class weights, or collecting more data for minority classes.")
                elif top_frac > 0.75:
                    findings.append(f"Moderate class imbalance: top class is {top_frac:.1%} of samples.")
                    recommendations.append("Consider class weighting or SMOTE for minority classes.")

        # Potential leakage checks
        for col in df.columns:
            if col == target:
                continue
            # Exact match
            try:
                if df[col].equals(df[target]):
                    findings.append(f"Column '{col}' exactly matches the target — potential leakage.")
                    recommendations.append(f"Remove or anonymize '{col}' before modeling if it's derived from the target.")
                    continue
            except Exception:
                pass

            # Numeric correlation
            if pd.api.types.is_numeric_dtype(df[col]) and pd.api.types.is_numeric_dtype(series):
                try:
                    corr = float(df[col].corr(series))
                    if np.isnan(corr):
                        corr = 0.0
                    if abs(corr) > 0.95:
                        findings.append(f"High correlation ({corr:.2f}) between '{col}' and target — potential leakage.")
                        recommendations.append(f"Investigate '{col}': it may be derived from the target.")
                except Exception:
                    pass
            else:
                # Categorical mapping check: if each unique value maps to a single target value
                try:
                    groups = df.groupby(col)[target].nunique()
                    if groups.max() == 1 and groups.shape[0] >= min(2, n_unique):
                        findings.append(f"Categorical column '{col}' perfectly predicts the target in groups — potential leakage or strong predictor.")
                        recommendations.append(f"Verify whether '{col}' was created using future information or the target itself.")
                except Exception:
                    pass

        # Suspicious accuracy detection
        leaderboard = experiment_results.get("leaderboard", [])
        if leaderboard:
            top = leaderboard[0]
            metrics = top.get("metrics", {})
            acc = metrics.get("accuracy")
            if acc is not None:
                if acc >= 0.98:
                    findings.append(f"Top model accuracy is extremely high ({acc:.2%}) — suspicious on small/local datasets.")
                    recommendations.append("Check for label leakage, data duplication between train/test, and data leakage from preprocessing.")
                elif acc >= 0.95 and n_rows < 1000:
                    findings.append(f"Top model accuracy is very high ({acc:.2%}) given dataset size ({n_rows}).")
                    recommendations.append("Validate with cross-validation and inspect for leakage.")

        # Overfitting hint: We cannot measure train vs test gap unless train metrics provided
        # But we can warn when many features >> samples
        if n_cols - 1 > n_rows / 5:
            findings.append("Feature count is high relative to sample size — risk of overfitting.")
            recommendations.append("Reduce dimensionality, use regularization, or collect more data.")

        if not findings:
            recommendations.append("No immediate issues detected. Proceed with cross-validation and model explainability checks.")

        return {"findings": findings, "recommendations": recommendations, "task_hint": task_hint}
