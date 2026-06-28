from __future__ import annotations

import logging
from datetime import datetime, timezone
from time import perf_counter
from typing import Optional

import pandas as pd

from core.context import ExperimentContext
from utils.helpers import configure_logger


class BaseAgent:
    """Shared foundation for all experiment agents."""

    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        self.logger = logger or configure_logger(self.__class__.__name__)

    def execute(self, context: ExperimentContext) -> ExperimentContext:
        """Execute the agent against the provided experiment context."""
        raise NotImplementedError

    def run(self, context: ExperimentContext) -> ExperimentContext:
        """Backward-compatible alias for execute."""
        return self.execute(context)

    def _resolve_dataframe(self, context: ExperimentContext) -> pd.DataFrame:
        """Resolve the dataframe used by the agent from the experiment context."""
        if context.cleaned_dataframe is not None:
            return context.cleaned_dataframe.copy()
        if context.dataset is None:
            raise ValueError("ExperimentContext must contain dataset or cleaned_dataframe")
        if isinstance(context.dataset, pd.DataFrame):
            return context.dataset.copy()
        return pd.DataFrame(context.dataset)

    def _record_agent_thought(self, context: ExperimentContext, agent_name: str, thought: str) -> str:
        """Persist a short thought summary for the agent in the experiment context."""
        context.agent_thoughts[agent_name] = thought
        context.experiment_metadata["agent_thoughts"] = dict(context.agent_thoughts)
        self.logger.info("%s thought: %s", agent_name, thought)
        return thought

    def _log_execution(self, context: ExperimentContext, operation: str) -> float:
        """Log the start and end of an agent execution and return elapsed seconds."""
        self.logger.info("Starting execution for %s", operation)
        start_time = perf_counter()
        return start_time

    def _log_completion(self, operation: str, started_at: float) -> None:
        """Log completion time for an agent execution."""
        elapsed = perf_counter() - started_at
        self.logger.info("Completed execution for %s in %.3fs", operation, elapsed)

    @staticmethod
    def _utc_timestamp() -> str:
        return datetime.now(timezone.utc).isoformat()
