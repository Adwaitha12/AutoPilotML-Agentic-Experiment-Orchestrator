from __future__ import annotations

from pathlib import Path
import tempfile
from typing import Any, Dict

import pandas as pd

from agents.cleaning_agent import DataCleaningAgent
from agents.critic_agent import CriticAgent
from agents.evaluation_agent import EvaluationAgent
from agents.memory_agent import MemoryAgent
from agents.model_agent import ModelSelectionAgent
from agents.planner import PlannerAgent
from agents.qa_agent import DataQualityAgent
from agents.report_agent import ReportAgent
from agents.task_agent import TaskDetectionAgent
from agents.training_agent import TrainingAgent
from agents.visualization_agent import VisualizationAgent
from core.context import ExperimentContext


def build_sample_dataset() -> pd.DataFrame:
    """Create a small classification dataset and save it to disk as CSV."""
    rows = []
    for i in range(80):
        feature1 = i % 10
        feature2 = (i * 3) % 7
        target = 1 if (feature1 + feature2 + (i % 3)) % 2 == 0 else 0
        rows.append({"feature1": feature1, "feature2": feature2, "target": target})

    dataframe = pd.DataFrame(rows)
    with tempfile.TemporaryDirectory() as temp_dir:
        csv_path = Path(temp_dir) / "sample_dataset.csv"
        dataframe.to_csv(csv_path, index=False)
        return pd.read_csv(csv_path)


def build_agents() -> Dict[str, Any]:
    """Assemble the agent registry for the full pipeline."""
    return {
        "quality": DataQualityAgent(),
        "cleaning": DataCleaningAgent(),
        "task": TaskDetectionAgent(),
        "models": ModelSelectionAgent(),
        "training": TrainingAgent(),
        "evaluation": EvaluationAgent(),
        "critic": CriticAgent(),
        "visualization": VisualizationAgent(output_dir="outputs/charts"),
        "report": ReportAgent(output_dir="outputs/reports"),
        "memory": MemoryAgent(db_path="outputs/experiments.db"),
    }


def main() -> None:
    """Load a sample CSV, run the planner-driven pipeline, and print key results."""
    dataframe = build_sample_dataset()
    context = ExperimentContext(dataset=dataframe, target_column="target")
    planner = PlannerAgent(context=context, agents=build_agents(), stop_on_error=True)

    print("Running complete pipeline...")
    planner.run(
        agent_order=[
            "quality",
            "cleaning",
            "task",
            "models",
            "training",
            "evaluation",
            "critic",
            "visualization",
            "report",
            "memory",
        ]
    )

    print("\nExecution history")
    print(context.execution_history)

    print("\nBest model")
    print(context.best_model.__class__.__name__ if context.best_model is not None else None)

    print("\nMetrics")
    print(context.metrics)

    print("\nCritic recommendations")
    critic_analysis = context.critic_analysis or {}
    for recommendation in critic_analysis.get("recommendations", []):
        print(f"- {recommendation}")

    print("\nReport path")
    print(context.report_path)


if __name__ == "__main__":
    main()
