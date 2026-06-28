from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass
from typing import Callable, Protocol

from core.context import ExperimentContext


logger = logging.getLogger(__name__)
PlannerCallback = Callable[[str, str, ExperimentContext], None]


class Agent(Protocol):
    """Contract every pipeline agent must follow."""

    def execute(self, context: ExperimentContext) -> ExperimentContext:
        """Run the agent and return the updated experiment context."""


@dataclass(frozen=True)
class AgentStep:
    """Import information for one agent in the AutoPilot ML workflow."""

    name: str
    module_path: str
    class_name: str
    required: bool = True


class PlannerAgent:
    """
    Orchestrates the AutoPilot ML agent pipeline.

    The planner intentionally contains no ML logic. It loads each specialized
    agent, calls execute(context), and records execution progress in the shared
    ExperimentContext.
    """

    DEFAULT_STEPS: tuple[AgentStep, ...] = (
        AgentStep("QA Agent", "agents.qa_agent", "QAAgent"),
        AgentStep("Cleaning Agent", "agents.cleaning_agent", "CleaningAgent"),
        AgentStep("Task Detection Agent", "agents.task_agent", "TaskAgent"),
        AgentStep("Model Selection Agent", "agents.model_agent", "ModelAgent"),
        AgentStep("Training Agent", "agents.training_agent", "TrainingAgent"),
        AgentStep("Evaluation Agent", "agents.evaluation_agent", "EvaluationAgent"),
        AgentStep("Critic Agent", "agents.critic_agent", "CriticAgent"),
        AgentStep(
            "Visualization Agent",
            "agents.visualization_agent",
            "VisualizationAgent",
            required=False,
        ),
        AgentStep("Report Agent", "agents.report_agent", "ReportAgent"),
        AgentStep("Memory Agent", "agents.memory_agent", "MemoryAgent", required=False),
    )

    def __init__(
        self,
        steps: tuple[AgentStep, ...] | None = None,
        progress_callback: PlannerCallback | None = None,
    ) -> None:
        self.steps = steps or self.DEFAULT_STEPS
        self.progress_callback = progress_callback

    def execute(self, context: ExperimentContext) -> ExperimentContext:
        """Run every configured agent in order."""
        logger.info("Starting experiment %s", context.experiment_id)

        for step in self.steps:
            self._run_step(step, context)

        logger.info("Finished experiment %s", context.experiment_id)
        return context

    def execution_plan(self) -> list[str]:
        """Return the ordered agent names for UI previews and logs."""
        return [step.name for step in self.steps]

    def _run_step(self, step: AgentStep, context: ExperimentContext) -> None:
        context.start_agent(step.name, "Starting")
        self._notify(step.name, "running", context)
        logger.info("Running %s", step.name)

        try:
            agent = self._load_agent(step)
        except (ImportError, AttributeError) as exc:
            self._handle_agent_setup_error(step, context, exc)
            return

        try:
            result = agent.execute(context)

            if result is not context:
                logger.warning(
                    "%s returned a different context object; keeping returned context data is unsupported.",
                    step.name,
                )

            context.complete_agent(step.name, "Completed")
            self._notify(step.name, "completed", context)
            logger.info("Completed %s", step.name)
        except Exception as exc:
            context.fail_agent(step.name, str(exc))
            self._notify(step.name, "failed", context)
            logger.exception("%s failed", step.name)
            raise

    def _load_agent(self, step: AgentStep) -> Agent:
        module = importlib.import_module(step.module_path)
        agent_class = getattr(module, step.class_name)
        agent = agent_class()

        if not hasattr(agent, "execute"):
            raise AttributeError(f"{step.class_name} must define execute(context).")

        return agent

    def _handle_agent_setup_error(
        self,
        step: AgentStep,
        context: ExperimentContext,
        exc: ImportError | AttributeError,
    ) -> None:
        message = f"{step.class_name} is not ready: {exc}"

        if step.required:
            context.fail_agent(step.name, message)
            self._notify(step.name, "failed", context)
            logger.exception("Required agent setup failed: %s", step.name)
            raise exc

        context.complete_agent(step.name, f"Skipped optional agent. {message}")
        self._notify(step.name, "completed", context)
        logger.warning("Skipped optional agent %s: %s", step.name, exc)

    def _notify(self, agent_name: str, status: str, context: ExperimentContext) -> None:
        if self.progress_callback is not None:
            self.progress_callback(agent_name, status, context)
