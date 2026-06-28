from __future__ import annotations

from core.context import ExperimentContext
from utils.metrics import metric_is_higher_better, primary_metric


class EvaluationAgent:
    """Compare trained models and select the best candidate."""

    def execute(self, context: ExperimentContext) -> ExperimentContext:
        if context.problem_type is None:
            raise ValueError("Problem type must exist before evaluation.")
        if not context.metrics:
            raise ValueError("No metrics found for evaluation.")

        metric_name = primary_metric(context.problem_type, context.optimization_goal)
        higher_is_better = metric_is_higher_better(metric_name)

        leaderboard = sorted(
            context.metrics.items(),
            key=lambda item: item[1][metric_name],
            reverse=higher_is_better,
        )

        best_model_name, best_metrics = leaderboard[0]
        context.best_model = best_model_name
        context.best_model_score = float(best_metrics[metric_name])
        context.metadata["leaderboard"] = [
            {"model": model_name, **metrics} for model_name, metrics in leaderboard
        ]
        context.metadata["primary_metric"] = metric_name
        context.metadata["higher_is_better"] = higher_is_better
        return context
