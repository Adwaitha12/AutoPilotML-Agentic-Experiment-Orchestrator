"""
Orchestrator

Coordinates all agents to run a full experiment pipeline given a DataFrame
and a target column. Returns a consolidated result object.
"""
from typing import Dict, Any
import os
import uuid

from agents.data_engineer import DataEngineer
from agents.ml_strategy import MLStrategy
from agents.model_architect import ModelArchitect
from agents.experiment_agent import ExperimentAgent
from agents.performance_analyst import PerformanceAnalyst
from agents.ai_reviewer import AIReviewer
from agents.insights_agent import InsightsAgent
from agents.documentation_agent import DocumentationAgent
from agents.knowledge_base import KnowledgeBase


class Orchestrator:
    def __init__(self, workspace_dir: str = ".") -> None:
        self.workspace_dir = workspace_dir
        self.data_engineer = DataEngineer()
        self.strategy = MLStrategy()
        self.architect = ModelArchitect()
        self.experiment_agent = ExperimentAgent(output_dir=os.path.join("outputs", "models"))
        self.performance = PerformanceAnalyst()
        self.reviewer = AIReviewer()
        self.insights = InsightsAgent(output_dir=os.path.join("outputs", "charts"))
        self.docs = DocumentationAgent(output_dir=os.path.join("outputs", "reports"))
        self.kb = KnowledgeBase()

    def run_experiment(self, df, target: str, experiment_name: str = None) -> Dict[str, Any]:
        if experiment_name is None:
            experiment_name = f"experiment_{uuid.uuid4().hex[:8]}"

        # 1. Data engineering
        X_clean, y, preproc = self.data_engineer.clean(df, target=target)

        # 2. Strategy
        strat = self.strategy.determine_task(df, target)
        task_type = strat.get("task_type")

        # 3. Model selection
        models = self.architect.select_models(task_type)

        # 4. Experiment / Train
        experiment_results = self.experiment_agent.train_models(models, X_clean, y, task_type)

        # 5. Performance analysis
        perf = self.performance.analyze(experiment_results, task_type)

        # 6. AI review
        review = self.reviewer.review(df, target, experiment_results)

        # 7. Insights
        insights_summary = self.insights.generate_insights(df, X_clean, y, experiment_results, task_type)

        # 8. Documentation
        report_paths = self.docs.generate_report(review, experiment_results, perf, insights_summary)

        # 9. Persist experiment
        kb_id = self.kb.save_experiment(
            name=experiment_name,
            dataset_summary=perf.get('leaderboard', []),
            target=target,
            task_type=task_type,
            models=[{"name": r.get('name'), "metrics": r.get('metrics')} for r in experiment_results.get('models', [])],
            best_model=perf.get('best_model', {}),
            metrics=perf.get('best_model', {}).get('metrics', {}),
            report_paths=report_paths,
        )

        return {
            "experiment_name": experiment_name,
            "kb_id": kb_id,
            "strategy": strat,
            "experiment_results": experiment_results,
            "performance": perf,
            "review": review,
            "insights": insights_summary,
            "report_paths": report_paths,
        }


# convenience

def run_full_experiment(df, target: str, name: str = None) -> Dict[str, Any]:
    return Orchestrator().run_experiment(df, target, experiment_name=name)
