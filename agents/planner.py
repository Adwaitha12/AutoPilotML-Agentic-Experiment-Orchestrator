from __future__ import annotations

import logging
from datetime import datetime, timezone
from time import perf_counter
from typing import Any, Callable, Dict, Mapping, Optional, Protocol, Sequence

from core.context import ExperimentContext
from utils.helpers import configure_logger


class SupportsExecute(Protocol):
    """Minimal protocol for agents that the planner can execute."""

    def execute(self, context: ExperimentContext) -> Any:
        """Execute the agent against the provided experiment context."""


class PlannerAgent:
    """Coordinate the execution of multiple agents in a controlled sequence.

    The planner is intentionally lightweight: it accepts an experiment context,
    invokes registered agents sequentially, records execution status in the
    shared metadata, and surfaces failures without implementing the underlying
    agent logic itself.
    """

    DISPLAY_NAMES: Dict[str, str] = {
        "quality": "QA Agent",
        "cleaning": "Cleaning Agent",
        "task": "Task Agent",
        "models": "Model Agent",
        "training": "Training Agent",
        "evaluation": "Evaluation Agent",
        "critic": "Critic Agent",
        "visualization": "Visualization Agent",
        "report": "Report Agent",
        "memory": "Memory Agent",
    }

    def __init__(
        self,
        context: ExperimentContext,
        agents: Optional[Mapping[str, Any]] = None,
        logger: Optional[logging.Logger] = None,
        stop_on_error: bool = False,
        status_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ) -> None:
        self.context = context
        self.agents: Dict[str, Any] = dict(agents or {})
        self.logger = logger or configure_logger(__name__)
        self.stop_on_error = stop_on_error
        self.status_callback = status_callback

    def register_agent(self, name: str, agent: Any) -> None:
        """Register an agent implementation under a logical name."""
        self.agents[name] = agent

    def run(self, agent_order: Optional[Sequence[str]] = None) -> ExperimentContext:
        """Execute the registered agents in order and update context metadata."""
        execution_order = list(agent_order or self.agents.keys())
        self.context.experiment_metadata["execution_status"] = {}
        self.context.execution_history = []
        self.context.experiment_metadata["execution_history"] = self.context.execution_history
        self.context.experiment_metadata["execution_summaries"] = {}
        self.context.experiment_metadata["planner_started_at"] = self._utc_timestamp()
        self.context.experiment_metadata["planner_status"] = "running"
        self.logger.info("Starting planner orchestration for %s agents", len(execution_order))
        had_failures = False

        for name in execution_order:
            if name not in self.agents:
                self.logger.warning("Agent '%s' is not registered; skipping.", name)
                self._record_status(
                    name,
                    {
                        "status": "skipped",
                        "message": "agent not registered",
                    },
                )
                continue

            self._run_single_agent(name, self.agents[name])
            had_failures = had_failures or self.context.experiment_metadata["execution_status"].get(name, {}).get("status") == "failed"

        self.context.experiment_metadata["planner_finished_at"] = self._utc_timestamp()
        self.context.experiment_metadata["planner_status"] = "completed_with_errors" if had_failures else "completed"
        self.logger.info("Planner orchestration finished with status %s", self.context.experiment_metadata["planner_status"])
        return self.context

    def _run_single_agent(self, name: str, agent: Any) -> None:
        started_at = self._utc_timestamp()
        status_entry: Dict[str, Any] = {"status": "running", "started_at": started_at}
        self._record_status(name, status_entry)

        try:
            self.logger.info("Executing agent '%s'", name)
            start_time = perf_counter()
            result = self._invoke_agent(agent)
            elapsed = perf_counter() - start_time

            if isinstance(result, ExperimentContext):
                self.context = result

            summary = self._build_execution_summary(name, result)
            self.context.experiment_metadata["execution_summaries"][name] = summary
            status_entry.update({
                "status": "completed",
                "message": "success",
                "duration_seconds": round(elapsed, 3),
                "summary": summary,
                "finished_at": self._utc_timestamp(),
            })
            self._record_status(name, status_entry)
        except Exception as exc:  # pragma: no cover - defensive orchestration
            status_entry.update({
                "status": "failed",
                "message": str(exc),
                "summary": f"Failed: {exc}",
                "finished_at": self._utc_timestamp(),
            })
            self._record_status(name, status_entry)
            self.logger.exception("Agent '%s' failed during execution", name)
            if self.stop_on_error:
                raise

    def _invoke_agent(self, agent: Any) -> Any:
        if hasattr(agent, "execute") and callable(getattr(agent, "execute")):
            return agent.execute(self.context)

        if hasattr(agent, "run") and callable(getattr(agent, "run")):
            return agent.run(self.context)

        if hasattr(agent, "process") and callable(getattr(agent, "process")):
            return agent.process(self.context)

        if callable(agent):
            return agent(self.context)

        raise TypeError(
            "Agent must expose a callable interface such as execute(), run(), process(), or be callable."
        )

    def _record_status(self, name: str, status_entry: Dict[str, Any]) -> None:
        self.context.experiment_metadata["execution_status"][name] = status_entry
        if str(status_entry.get("status", "")).lower() not in {"running"}:
            self.context.execution_history.append(self._to_history_entry(name, status_entry))
            self.context.experiment_metadata["execution_history"] = self.context.execution_history
        if self.status_callback is not None:
            self.status_callback(name, status_entry)

    def _to_history_entry(self, name: str, status_entry: Dict[str, Any]) -> Dict[str, Any]:
        status_value = self._normalize_status(status_entry.get("status", ""))
        summary = status_entry.get("summary") or status_entry.get("message") or "Completed"
        return {
            "agent": self.DISPLAY_NAMES.get(name, name.replace("_", " ").title()),
            "status": status_value,
            "summary": summary,
        }

    @staticmethod
    def _normalize_status(status: str) -> str:
        normalized = str(status or "").strip().upper()
        if normalized in {"COMPLETED", "SUCCESS"}:
            return "SUCCESS"
        if normalized in {"FAILED", "ERROR"}:
            return "FAILED"
        if normalized in {"SKIPPED", "RUNNING"}:
            return normalized
        return normalized or "UNKNOWN"

    def _build_execution_summary(self, name: str, result: Any) -> str:
        if name == "quality" and isinstance(self.context.quality_report, dict):
            missing_values = self.context.quality_report.get("missing_values", {})
            duplicate_rows = self.context.quality_report.get("duplicate_rows", 0)
            return (
                f"Detected {missing_values.get('total', 0)} missing values "
                f"and {duplicate_rows} duplicate rows"
            )

        if name == "cleaning" and isinstance(self.context.preprocessing_summary, dict):
            return "Prepared a cleaned dataset"

        if name == "task" and self.context.problem_type:
            return f"Detected problem type '{self.context.problem_type}'"

        if name == "models" and self.context.selected_models:
            return f"Selected {len(self.context.selected_models)} model(s)"

        if name == "training" and self.context.trained_models:
            return f"Trained {len(self.context.trained_models)} model(s)"

        if name == "evaluation" and self.context.leaderboard:
            return f"Evaluated {len(self.context.leaderboard)} model(s)"

        if name == "critic" and self.context.critic_analysis:
            return "Generated critic recommendations"

        if name == "visualization" and self.context.visualizations:
            return f"Generated {len(self.context.visualizations)} visualization(s)"

        if name == "report" and self.context.report_path:
            return "Generated experiment report"

        if name == "memory":
            return "Persisted experiment state"

        if isinstance(result, ExperimentContext):
            return "Completed successfully"

        if isinstance(result, dict):
            return f"Processed {len(result)} field(s)"

        if isinstance(result, (list, tuple)):
            return f"Processed {len(result)} item(s)"

        return "Completed successfully"

    @staticmethod
    def _summarize_result(result: Any) -> Dict[str, Any]:
        if isinstance(result, ExperimentContext):
            return {
                "type": "ExperimentContext",
                "problem_type": result.problem_type,
                "target_column": result.target_column,
                "best_model": result.best_model.__class__.__name__ if result.best_model is not None else None,
                "report_path": str(result.report_path) if result.report_path is not None else None,
            }

        if isinstance(result, dict):
            return {"type": "dict", "keys": list(result.keys())}

        if isinstance(result, (list, tuple)):
            return {"type": type(result).__name__, "length": len(result)}

        return {"type": type(result).__name__}

    @staticmethod
    def _utc_timestamp() -> str:
        return datetime.now(timezone.utc).isoformat()


__all__ = ["PlannerAgent", "SupportsExecute"]
