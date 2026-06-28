from __future__ import annotations

<<<<<<< HEAD
import logging
from typing import Any, Dict, List, Optional

from sklearn.base import BaseEstimator
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from agents.base_agent import BaseAgent
from core.context import ExperimentContext


class ModelSelectionAgent(BaseAgent):
    """Select and initialize a set of baseline models for the detected task.

    The agent uses the problem type stored in the experiment context to choose a
    suite of candidate models, instantiates them, and records both the models and
    a human-readable explanation of the selection rationale in the shared context.
    """

    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        self.logger = logger or logging.getLogger(__name__)

    def execute(self, context: ExperimentContext) -> ExperimentContext:
        """Initialize models for the problem type in the current experiment context."""
        started_at = self._log_execution(context, "model_agent")
        problem_type = self._resolve_problem_type(context)
        models = self._select_models(problem_type)

        context.selected_models = [model.__class__.__name__ for model in models]
        context.trained_models = {}
        context.experiment_metadata["model_selection"] = {
            "problem_type": problem_type,
            "selected_models": context.selected_models,
            "reasoning": self._build_reasoning(problem_type),
        }

        self._record_agent_thought(
            context,
            "model_agent",
            f"I selected {', '.join(context.selected_models)} because the task is {problem_type}.",
        )
        self._log_completion("model_agent", started_at)
        self.logger.info("Initialized %s models for %s", len(models), problem_type)
        return context

    def run(self, context: ExperimentContext) -> ExperimentContext:
        """Backward-compatible alias for execute."""
        return self.execute(context)

    def _resolve_problem_type(self, context: ExperimentContext) -> str:
        """Resolve the problem type from the experiment context."""
        problem_type = context.problem_type
        if problem_type:
            return problem_type
        raise ValueError("ExperimentContext must define problem_type before model selection")

    def _select_models(self, problem_type: str) -> List[BaseEstimator]:
        """Instantiate the appropriate baseline models for the detected task."""
        if problem_type == "classification":
            return [
                LogisticRegression(max_iter=1000),
                DecisionTreeClassifier(random_state=42),
                RandomForestClassifier(random_state=42),
                GradientBoostingClassifier(random_state=42),
            ]

        if problem_type == "regression":
            return [
                LinearRegression(),
                DecisionTreeRegressor(random_state=42),
                RandomForestRegressor(random_state=42),
                GradientBoostingRegressor(random_state=42),
            ]

        raise ValueError(f"Unsupported problem type: {problem_type}")

    def _build_reasoning(self, problem_type: str) -> str:
        """Create a human-readable explanation for the selected model suite."""
        if problem_type == "classification":
            return (
                "Classification was selected, so the agent initialized a mix of linear, tree-based, "
                "and boosting models to cover simple and more complex decision boundaries."
            )

        if problem_type == "regression":
            return (
                "Regression was selected, so the agent initialized linear, tree-based, and boosting "
                "models to capture both linear and nonlinear relationships."
            )

        return "No model selection rationale is available for the requested problem type."


__all__ = ["ModelSelectionAgent"]
=======
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
>>>>>>> origin/main
