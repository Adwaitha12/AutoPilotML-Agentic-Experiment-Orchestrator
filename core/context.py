from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional

if TYPE_CHECKING:
    import pandas as pd

ProblemType = Literal["classification", "regression", "clustering", "time_series", "other"]


@dataclass
class ExperimentContext:
    """Container for the shared state of an ML experiment workflow."""

    dataset: Optional[Any] = None
    clean_dataframe: Optional["pd.DataFrame"] = None
    target_column: Optional[str] = None
    problem_type: Optional[ProblemType] = None
    quality_report: Optional[Dict[str, Any]] = None
    preprocessing_summary: Optional[Dict[str, Any]] = None
    selected_models: List[str] = field(default_factory=list)
    trained_models: Dict[str, Any] = field(default_factory=dict)
    predictions: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    leaderboard: List[Dict[str, Any]] = field(default_factory=list)
    best_model: Optional[Any] = None
    critic_analysis: Optional[Dict[str, Any]] = None
    agent_thoughts: Dict[str, str] = field(default_factory=dict)
    visualizations: Dict[str, str] = field(default_factory=dict)
    plotly_visualizations: Dict[str, Any] = field(default_factory=dict)
    report_path: Optional[Path] = None
    execution_history: List[Dict[str, Any]] = field(default_factory=list)
    experiment_metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.report_path is not None and not isinstance(self.report_path, Path):
            self.report_path = Path(self.report_path)

    @property
    def cleaned_dataframe(self) -> Optional["pd.DataFrame"]:
        """Backward-compatible alias for the cleaned dataframe field."""
        return self.clean_dataframe

    @cleaned_dataframe.setter
    def cleaned_dataframe(self, value: Optional["pd.DataFrame"]) -> None:
        self.clean_dataframe = value

    def update_metadata(self, **metadata: Any) -> None:
        self.experiment_metadata.update(metadata)

    def sanitize_visualization_artifacts(self, artifacts: Optional[Dict[str, Any]]) -> Dict[str, str]:
        """Return only valid image-path artifacts that can be rendered by the UI."""
        if not artifacts:
            return {}

        sanitized: Dict[str, str] = {}
        for name, artifact in artifacts.items():
            if not isinstance(artifact, (str, Path)):
                continue

            try:
                candidate = Path(artifact)
            except TypeError:
                continue

            if not candidate.exists() or not candidate.is_file():
                continue
            if candidate.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                continue
            try:
                if candidate.stat().st_size <= 0:
                    continue
            except OSError:
                continue
            sanitized[name] = str(candidate)

        return sanitized

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset": self.dataset,
            "clean_dataframe": self.clean_dataframe,
            "cleaned_dataframe": self.clean_dataframe,
            "target_column": self.target_column,
            "problem_type": self.problem_type,
            "quality_report": self.quality_report,
            "preprocessing_summary": self.preprocessing_summary,
            "selected_models": self.selected_models,
            "trained_models": list(self.trained_models.keys()),
            "predictions": self.predictions,
            "metrics": self.metrics,
            "leaderboard": self.leaderboard,
            "best_model": self.best_model,
            "critic_analysis": self.critic_analysis,
            "agent_thoughts": self.agent_thoughts,
            "visualizations": self.sanitize_visualization_artifacts(self.visualizations),
            "plotly_visualizations": self._serialize_artifacts(self.plotly_visualizations),
            "report_path": str(self.report_path) if self.report_path else None,
            "execution_history": self.execution_history,
            "experiment_metadata": self.experiment_metadata,
        }

    @staticmethod
    def _serialize_artifacts(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {key: ExperimentContext._serialize_artifacts(item) for key, item in value.items()}
        if isinstance(value, list):
            return [ExperimentContext._serialize_artifacts(item) for item in value]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)


__all__ = ["ExperimentContext", "ProblemType"]
