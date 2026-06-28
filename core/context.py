from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

import pandas as pd


@dataclass
class AgentRun:
    """Execution record for one agent in the experiment pipeline."""

    name: str
    status: str
    message: str = ""
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    finished_at: datetime | None = None

    def complete(self, message: str = "") -> None:
        self.status = "completed"
        self.message = message or self.message
        self.finished_at = datetime.now(timezone.utc)

    def fail(self, message: str) -> None:
        self.status = "failed"
        self.message = message
        self.finished_at = datetime.now(timezone.utc)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "message": self.message,
            "started_at": self.started_at.isoformat(),
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
        }


@dataclass
class ExperimentContext:
    """
    Shared state passed through every AutoPilot ML agent.

    Agents should communicate only by reading from and writing to this object.
    The planner owns the execution order and passes the same context to each
    agent's public execute(context) method.
    """

    dataset: pd.DataFrame
    target_column: str
    optimization_goal: str = "highest_accuracy"
    test_size: float = 0.2
    business_goal: str | None = None
    dataset_name: str = "uploaded_dataset.csv"
    experiment_id: str = field(default_factory=lambda: str(uuid4()))
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    quality_report: dict[str, Any] | None = None
    cleaning_summary: dict[str, Any] | None = None
    cleaning_changes: list[dict[str, Any]] = field(default_factory=list)
    clean_dataframe: pd.DataFrame | None = None
    feature_columns: list[str] = field(default_factory=list)
    problem_type: str | None = None
    selected_models: list[str] = field(default_factory=list)
    trained_models: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    train_shape: tuple[int, int] | None = None
    test_shape: tuple[int, int] | None = None
    predictions: dict[str, list[Any]] = field(default_factory=dict)
    probabilities: dict[str, list[float]] = field(default_factory=dict)
    classification_reports: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    feature_importances: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    best_model: str | None = None
    best_model_score: float | None = None
    critic_comments: list[str] = field(default_factory=list)
    critic_analysis: dict[str, Any] | None = None
    charts: dict[str, str] = field(default_factory=dict)
    report_path: str | None = None
    html_report_path: str | None = None
    pdf_report_path: str | None = None
    memory_record_id: str | None = None

    agent_runs: list[AgentRun] = field(default_factory=list)
    artifacts: dict[str, str] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._validate_dataset()
        self.feature_columns = [
            column for column in self.dataset.columns if column != self.target_column
        ]

    @property
    def active_dataset(self) -> pd.DataFrame:
        """Return the cleaned dataset when available, otherwise the original."""
        return self.clean_dataframe if self.clean_dataframe is not None else self.dataset

    @property
    def clean_dataset(self) -> pd.DataFrame | None:
        """Backward-compatible alias for earlier teammate task wording."""
        return self.clean_dataframe

    @clean_dataset.setter
    def clean_dataset(self, value: pd.DataFrame | None) -> None:
        self.clean_dataframe = value

    def start_agent(self, name: str, message: str = "") -> AgentRun:
        run = AgentRun(name=name, status="running", message=message)
        self.agent_runs.append(run)
        return run

    def complete_agent(self, name: str, message: str = "") -> None:
        run = self._latest_agent_run(name)
        if run is None:
            run = self.start_agent(name)
        run.complete(message)

    def fail_agent(self, name: str, message: str) -> None:
        run = self._latest_agent_run(name)
        if run is None:
            run = self.start_agent(name)
        run.fail(message)
        self.errors.append(f"{name}: {message}")

    def add_chart(self, name: str, path: str) -> None:
        self.charts[name] = path
        self.artifacts[f"chart:{name}"] = path

    def add_artifact(self, name: str, path: str) -> None:
        self.artifacts[name] = path

    def add_critic_comment(self, comment: str) -> None:
        if comment:
            self.critic_comments.append(comment)

    def summary(self) -> dict[str, Any]:
        """Return lightweight experiment details for dashboards and reports."""
        return {
            "experiment_id": self.experiment_id,
            "dataset_name": self.dataset_name,
            "target_column": self.target_column,
            "optimization_goal": self.optimization_goal,
            "test_size": self.test_size,
            "business_goal": self.business_goal,
            "rows": int(self.dataset.shape[0]),
            "columns": int(self.dataset.shape[1]),
            "problem_type": self.problem_type,
            "selected_models": self.selected_models,
            "best_model": self.best_model,
            "best_model_score": self.best_model_score,
            "report_path": self.report_path,
            "html_report_path": self.html_report_path,
            "pdf_report_path": self.pdf_report_path,
            "created_at": self.created_at.isoformat(),
            "agent_runs": [run.to_dict() for run in self.agent_runs],
            "errors": self.errors,
        }

    def _validate_dataset(self) -> None:
        if self.dataset.empty:
            raise ValueError("Dataset cannot be empty.")
        if self.target_column not in self.dataset.columns:
            raise ValueError(
                f"Target column '{self.target_column}' was not found in the dataset."
            )

    def _latest_agent_run(self, name: str) -> AgentRun | None:
        for run in reversed(self.agent_runs):
            if run.name == name:
                return run
        return None
