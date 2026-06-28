"""
Insights Agent

Generates visualizations and model explainability artifacts:
- confusion matrix (classification)
- ROC curve data (classification)
- feature importance
- missing value chart
- model comparison chart

Saves charts to `outputs/charts/` and returns a summary dictionary with paths.
"""
from typing import Dict, Any, List, Optional
import os
import joblib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc


class InsightsAgent:
    def __init__(self, output_dir: str = "outputs/charts") -> None:
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def _save_plot(self, fig, name: str) -> str:
        path = os.path.join(self.output_dir, name)
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        return path

    def _plot_confusion_matrix(self, y_true, y_pred, labels, title: str) -> str:
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]), xticklabels=labels, yticklabels=labels,
               ylabel='True label', xlabel='Predicted label', title=title)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        fmt = 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt), ha="center", va="center", color="white" if cm[i, j] > thresh else "black")
        name = f"confusion_{title.replace(' ', '_')}.png"
        return self._save_plot(fig, name)

    def _plot_roc_curve(self, y_true, y_proba, title: str) -> str:
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
        ax.set(xlabel='False Positive Rate', ylabel='True Positive Rate', title=title)
        ax.legend(loc="lower right")
        name = f"roc_{title.replace(' ', '_')}.png"
        return self._save_plot(fig, name)

    def _plot_feature_importance(self, feature_names: List[str], importances: List[float], title: str) -> str:
        fig, ax = plt.subplots(figsize=(6, max(4, len(feature_names) * 0.3)))
        indices = np.argsort(importances)[::-1]
        sorted_names = [feature_names[i] for i in indices]
        sorted_importances = [importances[i] for i in indices]
        ax.barh(range(len(sorted_names)), sorted_importances[::-1], align='center')
        ax.set_yticks(range(len(sorted_names)))
        ax.set_yticklabels(sorted_names[::-1])
        ax.set_xlabel('Importance')
        ax.set_title(title)
        name = f"feature_importance_{title.replace(' ', '_')}.png"
        return self._save_plot(fig, name)

    def _plot_missing_values(self, df: pd.DataFrame, title: str) -> str:
        missing = df.isnull().sum()
        fig, ax = plt.subplots(figsize=(6, max(3, len(missing) * 0.2)))
        missing.sort_values(ascending=False).plot(kind='bar', ax=ax)
        ax.set_ylabel('Missing count')
        ax.set_title(title)
        name = f"missing_{title.replace(' ', '_')}.png"
        return self._save_plot(fig, name)

    def _plot_model_comparison(self, leaderboard: List[Dict[str, Any]], task_type: str, title: str) -> str:
        # Build DataFrame of primary metric
        names = []
        primary = []
        for entry in leaderboard:
            names.append(entry.get('name'))
            metrics = entry.get('metrics', {})
            if task_type == 'classification':
                primary.append(metrics.get('accuracy', 0))
            else:
                # For regression, lower mse is better; invert for plotting
                mse = metrics.get('mse', None)
                primary.append(-mse if mse is not None else 0)
        fig, ax = plt.subplots(figsize=(6, max(3, len(names) * 0.4)))
        ax.barh(names, primary)
        ax.set_xlabel('Primary metric (higher is better)')
        ax.set_title(title)
        name = f"model_comparison_{title.replace(' ', '_')}.png"
        return self._save_plot(fig, name)

    def generate_insights(self, df: pd.DataFrame, X: pd.DataFrame, y: pd.Series, experiment_results: Dict[str, Any], task_type: str, test_size: float = 0.2, random_state: int = 42) -> Dict[str, Any]:
        """
        Generate charts and insights from experiment results.

        Returns a summary dict with paths to generated charts and data used.
        """
        summary: Dict[str, Any] = {}

        # Recreate train/test split used by ExperimentAgent to evaluate
        from sklearn.model_selection import train_test_split
        X_vals = X.values if isinstance(X, pd.DataFrame) else X
        y_vals = y.values if hasattr(y, 'values') else y
        X_train, X_test, y_train, y_test = train_test_split(X_vals, y_vals, test_size=test_size, random_state=random_state)

        # Map model names to paths
        leaderboard = experiment_results.get('leaderboard', [])

        generated = {'confusion_matrices': [], 'roc_curves': [], 'feature_importances': [], 'missing_chart': None, 'model_comparison': None}

        # Missing values chart (from original df)
        missing_path = self._plot_missing_values(df, 'Missing_Values')
        generated['missing_chart'] = missing_path

        # Model comparison chart
        mc_path = self._plot_model_comparison(leaderboard, task_type, 'Model_Comparison')
        generated['model_comparison'] = mc_path

        feature_names = X.columns.tolist() if isinstance(X, pd.DataFrame) else [f'f{i}' for i in range(X.shape[1])]

        for entry in leaderboard:
            name = entry.get('name')
            model_path = entry.get('model_path')
            if not model_path:
                continue
            try:
                model = joblib.load(model_path)
            except Exception:
                continue

            try:
                y_pred = model.predict(X_test)
            except Exception:
                continue

            # Confusion matrix & ROC for classification
            if task_type == 'classification':
                # Try to get class labels from y
                labels = np.unique(y_test)
                cm_path = self._plot_confusion_matrix(y_test, y_pred, labels=labels, title=name)
                generated['confusion_matrices'].append({'model': name, 'path': cm_path})

                # ROC curve if probability available
                y_proba = None
                try:
                    y_proba = model.predict_proba(X_test)
                except Exception:
                    y_proba = None
                if y_proba is not None and y_proba.shape[1] == 2:
                    roc_path = self._plot_roc_curve(y_test, y_proba[:, 1], title=name)
                    generated['roc_curves'].append({'model': name, 'path': roc_path})

            # Feature importance
            importances = None
            if hasattr(model, 'feature_importances_'):
                importances = list(getattr(model, 'feature_importances_'))
            elif hasattr(model, 'coef_'):
                coef = getattr(model, 'coef_')
                # handle multiclass
                if coef.ndim > 1:
                    coef = np.mean(coef, axis=0)
                importances = list(np.abs(coef))

            if importances is not None:
                fi_path = self._plot_feature_importance(feature_names, importances, title=name)
                generated['feature_importances'].append({'model': name, 'path': fi_path})

        summary['generated'] = generated
        return summary
