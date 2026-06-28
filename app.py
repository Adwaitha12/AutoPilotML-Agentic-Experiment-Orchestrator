from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pandas as pd
import streamlit as st
from PIL import Image, UnidentifiedImageError

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
from utils.helpers import configure_logger


st.set_page_config(page_title="AutoPilotML", page_icon="🤖", layout="wide")


def main() -> None:
    """Render the Streamlit dashboard for the experiment workflow."""
    st.title("AutoPilotML Agentic Experiment Orchestrator")
    st.caption("A professional, agent-driven workspace for dataset QA, modeling, evaluation, and reporting.")

    initialize_state()

    with st.sidebar:
        st.header("Experiment Setup")
        st.markdown("Upload a CSV, choose the target, and launch the workflow.")
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

        target_column = None
        if uploaded_file is not None:
            dataframe = pd.read_csv(uploaded_file)
            target_options = list(dataframe.columns)
            target_column = st.selectbox("Target column", options=target_options, index=0 if target_options else None)
            st.session_state["dataframe"] = dataframe
            st.session_state["target_column"] = target_column

        st.divider()
        run_disabled = "dataframe" not in st.session_state or st.session_state["dataframe"] is None
        if st.button("Run Experiment", use_container_width=True, disabled=run_disabled):
            if st.session_state.get("dataframe") is not None:
                run_experiment(st.session_state["dataframe"], st.session_state.get("target_column"))

        st.divider()
        st.subheader("Status")
        if st.session_state.get("context") is not None:
            st.success("Experiment ready for review")
        else:
            st.info("Awaiting your first run")

    if st.session_state.get("dataframe") is None:
        st.info("Upload a CSV file to begin an experiment.")
        return

    dataframe = st.session_state["dataframe"]
    with st.container():
        metric_cols = st.columns(3)
        with metric_cols[0]:
            st.metric("Rows", f"{dataframe.shape[0]:,}")
        with metric_cols[1]:
            st.metric("Columns", f"{dataframe.shape[1]:,}")
        with metric_cols[2]:
            st.metric("Target", st.session_state.get("target_column") or "—")

    with st.expander("Dataset Preview", expanded=True):
        st.dataframe(dataframe.head(20), use_container_width=True)

    if st.session_state.get("context") is not None:
        render_results(st.session_state["context"])


def initialize_state() -> None:
    """Initialize Streamlit session state for the app."""
    for key in ["dataframe", "target_column", "context", "console_lines"]:
        st.session_state.setdefault(key, None)
    if st.session_state["console_lines"] is None:
        st.session_state["console_lines"] = []


def run_experiment(dataframe: pd.DataFrame, target_column: str | None) -> None:
    """Create an ExperimentContext, run the planner, and expose the results in the UI."""
    logger = configure_logger("app")
    context = ExperimentContext(dataset=dataframe, target_column=target_column)
    st.session_state["context"] = context
    st.session_state["console_lines"] = []

    agent_order = [
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

    def status_callback(agent_name: str, status_entry: Dict[str, Any]) -> None:
        status = str(status_entry.get("status", "")).upper()
        summary = status_entry.get("summary") or status_entry.get("message") or "Processing"
        st.session_state["console_lines"].append(f"{agent_name}: {status} - {summary}")
        progress_value = min((len(st.session_state["console_lines"]) / len(agent_order)), 1.0)
        progress_bar.progress(progress_value, text=f"Executing {agent_name}")

    agents: Dict[str, Any] = build_agents(logger=logger)
    planner = PlannerAgent(context=context, agents=agents, logger=logger, status_callback=status_callback)

    progress_bar = st.progress(0.0, text="Preparing the workflow")
    with st.spinner("Running the experiment workflow..."):
        try:
            planner.run(agent_order=agent_order)
            progress_bar.progress(1.0, text="Workflow completed")
            st.session_state["context"] = context
            st.success("Experiment completed successfully.")
        except Exception as exc:
            logger.exception("Experiment workflow failed")
            st.session_state["context"] = context
            st.error(f"Experiment failed: {exc}")


def build_agents(logger: Any) -> Dict[str, Any]:
    """Assemble the agent registry used by the planner."""
    return {
        "quality": DataQualityAgent(logger=logger),
        "cleaning": DataCleaningAgent(logger=logger),
        "task": TaskDetectionAgent(logger=logger),
        "models": ModelSelectionAgent(logger=logger),
        "training": TrainingAgent(logger=logger),
        "evaluation": EvaluationAgent(logger=logger),
        "critic": CriticAgent(logger=logger),
        "visualization": VisualizationAgent(output_dir="outputs/charts", logger=logger),
        "report": ReportAgent(output_dir="outputs/reports", logger=logger),
        "memory": MemoryAgent(db_path="outputs/experiments.db", logger=logger),
    }


def render_results(context: ExperimentContext) -> None:
    """Render the experiment results in a structured dashboard."""
    st.divider()
    st.subheader("Dashboard")
    metric_cols = st.columns(4)
    with metric_cols[0]:
        st.metric("Problem Type", context.problem_type or "—")
    with metric_cols[1]:
        st.metric("Best Model", context.best_model.__class__.__name__ if context.best_model is not None else "—")
    with metric_cols[2]:
        st.metric("Agent Thoughts", len(context.agent_thoughts))
    with metric_cols[3]:
        st.metric("Report", "Ready" if context.report_path else "Pending")

    tabs = st.tabs(["Agent Console", "Results", "Visualizations", "Report"])

    with tabs[0]:
        history = context.execution_history or context.experiment_metadata.get("execution_history", [])
        if history:
            st.dataframe(pd.DataFrame(history), use_container_width=True)
        else:
            st.info("The planner has not recorded any execution history yet.")

        console_lines = st.session_state.get("console_lines", [])
        if console_lines:
            with st.expander("Execution Console", expanded=True):
                st.code("\n".join(console_lines), language="text")
        else:
            st.info("No execution events were recorded yet.")

    with tabs[1]:
        left_col, right_col = st.columns(2, gap="large")
        with left_col:
            with st.expander("Agent Thoughts", expanded=True):
                if context.agent_thoughts:
                    for name, thought in context.agent_thoughts.items():
                        st.markdown(f"- **{name}**: {thought}")
                else:
                    st.info("No agent thoughts available yet.")

            with st.expander("QA Report", expanded=True):
                if context.quality_report:
                    st.json(context.quality_report)
                else:
                    st.info("No QA report is available.")

            with st.expander("Cleaning Summary", expanded=True):
                preprocessing_summary = context.experiment_metadata.get("preprocessing_summary", {})
                if preprocessing_summary:
                    st.json(preprocessing_summary)
                else:
                    st.info("No cleaning summary is available.")

        with right_col:
            with st.expander("Leaderboard", expanded=True):
                evaluation_results = context.experiment_metadata.get("evaluation_results", {})
                leaderboard = evaluation_results.get("leaderboard", [])
                if leaderboard:
                    st.dataframe(pd.DataFrame(leaderboard), use_container_width=True)
                else:
                    st.info("No leaderboard is available yet.")

            with st.expander("Critic Analysis", expanded=True):
                critic_analysis = context.critic_analysis or {}
                if critic_analysis:
                    st.write(critic_analysis.get("summary", ""))
                    for recommendation in critic_analysis.get("recommendations", []):
                        st.markdown(f"- {recommendation}")
                else:
                    st.info("No critic analysis is available yet.")

    with tabs[2]:
        render_visualizations(context)

    with tabs[3]:
        if context.report_path is not None:
            with open(context.report_path, "rb") as report_file:
                st.download_button(
                    label="Download HTML Report",
                    data=report_file.read(),
                    file_name=context.report_path.name,
                    mime="text/html",
                )
            st.markdown("Preview is available in the generated HTML report.")
        else:
            st.info("No report has been generated yet.")


def render_visualizations(context: ExperimentContext) -> None:
    """Safely render saved image charts and Plotly figures without crashing the app."""
    logger = configure_logger("app")
    visualizations = context.sanitize_visualization_artifacts(context.visualizations or {})
    plotly_visualizations = context.plotly_visualizations or {}

    if visualizations:
        chart_cols = st.columns(min(2, len(visualizations)))
        for index, (name, path) in enumerate(visualizations.items()):
            with chart_cols[index % len(chart_cols)]:
                st.caption(name)
                render_image(path, logger)
    else:
        st.info("No visualizations were generated.")

    if plotly_visualizations:
        st.divider()
        st.subheader("Interactive Charts")
        for name, figure in plotly_visualizations.items():
            st.caption(name)
            st.plotly_chart(figure, use_container_width=True)


def render_image(path: str | Path, logger: Any) -> None:
    """Render a visualization image only when it is a valid file."""
    if not isinstance(path, (str, Path)):
        logger.warning("Visualization artifact is not a valid path: %r", path)
        st.warning("Visualization unavailable")
        return

    file_path = Path(path)
    if not file_path.exists() or not file_path.is_file():
        logger.warning("Visualization file not found: %s", path)
        st.warning("Visualization unavailable")
        return

    if file_path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
        logger.warning("Unsupported visualization format: %s", path)
        st.warning("Visualization unavailable")
        return

    if file_path.stat().st_size <= 0:
        logger.warning("Visualization file is empty: %s", path)
        st.warning("Visualization unavailable")
        return

    try:
        with Image.open(file_path) as image:
            st.image(image)
    except (FileNotFoundError, UnidentifiedImageError, OSError, ValueError) as exc:
        logger.warning("Could not load visualization %s: %s", path, exc)
        st.warning("Visualization unavailable")


if __name__ == "__main__":
    main()
