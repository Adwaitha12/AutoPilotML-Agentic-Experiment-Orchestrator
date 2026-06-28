from __future__ import annotations

from core.context import ExperimentContext


class ModelAgent:
    """Select a compact model portfolio for the detected task."""

    CLASSIFICATION_MODELS = [
        "logistic_regression",
        "decision_tree",
        "random_forest",
        "gradient_boosting",
    ]
    REGRESSION_MODELS = [
        "linear_regression",
        "decision_tree_regressor",
        "random_forest_regressor",
        "gradient_boosting_regressor",
    ]
    EXPLAINABLE_MODELS = {
        "classification": ["logistic_regression", "decision_tree"],
        "regression": ["linear_regression", "decision_tree_regressor"],
    }

    def execute(self, context: ExperimentContext) -> ExperimentContext:
        if context.problem_type is None:
            raise ValueError("Problem type must be detected before model selection.")

        if context.optimization_goal == "most_explainable":
            models = self.EXPLAINABLE_MODELS[context.problem_type]
        elif context.problem_type == "classification":
            models = self.CLASSIFICATION_MODELS
        else:
            models = self.REGRESSION_MODELS

        context.selected_models = models
        context.metadata["model_selection"] = {
            "optimization_goal": context.optimization_goal,
            "strategy": "explainable subset"
            if context.optimization_goal == "most_explainable"
            else "balanced benchmark portfolio",
        }
        return context
