"""
Model Architect Agent

Selects appropriate scikit-learn estimators for classification and
regression tasks. Returns a list of model spec dictionaries containing
`name` and `estimator` (unfitted) suitable for training in the
Experiment Agent.
"""
from typing import List, Dict

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor


class ModelArchitect:
    """Provides model selection based on task type."""

    CLASSIFICATION_MODELS = [
        ("Logistic Regression", LogisticRegression(max_iter=1000)),
        ("Decision Tree", DecisionTreeClassifier(random_state=42)),
        ("Random Forest", RandomForestClassifier(n_estimators=100, random_state=42)),
        ("Gradient Boosting", GradientBoostingClassifier(n_estimators=100, random_state=42)),
    ]

    REGRESSION_MODELS = [
        ("Linear Regression", LinearRegression()),
        ("Decision Tree Regressor", DecisionTreeRegressor(random_state=42)),
        ("Random Forest Regressor", RandomForestRegressor(n_estimators=100, random_state=42)),
        ("Gradient Boosting Regressor", GradientBoostingRegressor(n_estimators=100, random_state=42)),
    ]

    def select_models(self, task_type: str) -> List[Dict[str, object]]:
        """
        Return a list of model specifications for the given task type.

        Args:
            task_type: 'classification' or 'regression'

        Returns:
            List of dicts: [{"name": str, "estimator": sklearn estimator}, ...]
        """
        task = task_type.lower().strip()
        if task == "classification":
            return [{"name": name, "estimator": est} for name, est in self.CLASSIFICATION_MODELS]
        elif task == "regression":
            return [{"name": name, "estimator": est} for name, est in self.REGRESSION_MODELS]
        else:
            raise ValueError("Unknown task_type. Expected 'classification' or 'regression'.")


# convenience factory
def get_models_for_task(task_type: str) -> List[Dict[str, object]]:
    return ModelArchitect().select_models(task_type)
