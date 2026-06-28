from __future__ import annotations

from core.context import ExperimentContext


class CriticAgent:
    """Review experiment quality and produce practical recommendations."""

    def execute(self, context: ExperimentContext) -> ExperimentContext:
        comments: list[str] = []
        dataframe = context.active_dataset

        if len(dataframe) < 200:
            comments.append(
                "Dataset is small; use cross-validation before trusting the ranking."
            )

        if context.problem_type == "classification":
            distribution = dataframe[context.target_column].value_counts(normalize=True)
            if not distribution.empty and float(distribution.max()) > 0.75:
                comments.append(
                    "Target classes are imbalanced; prioritize F1 score over accuracy."
                )

        if context.best_model_score is not None:
            if context.problem_type == "classification" and context.best_model_score > 0.98:
                comments.append(
                    "Very high score detected; verify there is no target leakage."
                )
            if context.problem_type == "regression" and context.best_model_score < 0.2:
                comments.append(
                    "Low R2 score; consider feature engineering or more predictive data."
                )

        if context.optimization_goal == "most_explainable":
            comments.append(
                "Explainability was prioritized, so simpler models may be preferred over peak performance."
            )

        if not comments:
            comments.append(
                "No major experiment risks detected in this first-pass automated review."
            )

        context.critic_comments = comments
        context.critic_analysis = {
            "checks": [
                "small_dataset",
                "class_imbalance",
                "data_leakage_signal",
                "metric_recommendation",
            ],
            "recommendations": comments,
            "business_recommendation": self._business_recommendation(context),
        }
        return context

    def _business_recommendation(self, context: ExperimentContext) -> str:
        goal = context.business_goal or "the stated business outcome"
        if context.best_model is None:
            return f"Run the full experiment before using the model for {goal}."
        return (
            f"Use {context.best_model.replace('_', ' ')} as the baseline model for "
            f"{goal}, then validate it with domain-specific acceptance criteria."
        )
