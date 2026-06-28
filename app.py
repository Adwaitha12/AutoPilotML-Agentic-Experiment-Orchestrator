from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from agents.planner import PlannerAgent
from core.context import ExperimentContext


OPTIMIZATION_GOALS = {
    "Best Accuracy": "highest_accuracy",
    "Fastest Model": "fastest_model",
    "Most Explainable": "most_explainable",
}
GOAL_LABELS = {value: label for label, value in OPTIMIZATION_GOALS.items()}

AGENT_LABELS = {
    "QA Agent": ("Data Inspector Agent", "Checking dataset..."),
    "Cleaning Agent": ("Cleaning Agent", "Cleaning data..."),
    "Task Detection Agent": ("ML Strategy Agent", "Detecting problem type..."),
    "Model Selection Agent": ("Model Architect Agent", "Selecting algorithms..."),
    "Training Agent": ("Experiment Agent", "Training models..."),
    "Evaluation Agent": ("Performance Analyst Agent", "Comparing models..."),
    "Critic Agent": ("AI Reviewer Agent", "Reviewing results..."),
    "Visualization Agent": ("Insights Agent", "Creating charts..."),
    "Report Agent": ("Documentation Agent", "Generating report..."),
    "Memory Agent": ("Knowledge Base Agent", "Saving experiment..."),
}


def main() -> None:
    st.set_page_config(
        page_title="AgentLab AI",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    initialize_state()

    uploaded_file = st.sidebar.file_uploader("Upload Dataset", type=["csv"])
    dataframe = load_dataset(uploaded_file) if uploaded_file is not None else None

    target_column = None
    if dataframe is not None:
        target_column = st.sidebar.selectbox("Target Column", dataframe.columns)

    with st.sidebar.expander("Experiment Settings", expanded=False):
        optimization_label = st.selectbox(
            "Optimization Goal",
            list(OPTIMIZATION_GOALS.keys()),
        )
        test_size_percent = st.slider("Test Size", 10, 50, 20, step=10)
        business_goal = st.text_input(
            "Business Goal",
            placeholder="Predict Customer Churn",
        )

    run_clicked = st.sidebar.button(
        "Run Experiment",
        type="primary",
        use_container_width=True,
        disabled=dataframe is None or target_column is None,
    )

    render_header(uploaded_file.name if uploaded_file else None, st.session_state.current_context)

    if dataframe is None or uploaded_file is None or target_column is None:
        st.info("Upload a CSV dataset to begin.")
        return

    render_dataset_summary(dataframe, target_column, st.session_state.current_context)
    render_dataset_preview(dataframe)

    if run_clicked:
        context = create_context(
            dataframe=dataframe,
            dataset_name=uploaded_file.name,
            target_column=target_column,
            optimization_goal=OPTIMIZATION_GOALS[optimization_label],
            test_size=test_size_percent / 100,
            business_goal=business_goal.strip() or None,
        )
        run_experiment(context)

    context = st.session_state.current_context
    if context is not None:
        with st.expander("Agent Execution", expanded=False):
            render_agent_execution(context)
        render_results(context)


def initialize_state() -> None:
    st.session_state.setdefault("current_context", None)
    st.session_state.setdefault("experiment_history", [])


@st.cache_data(show_spinner=False)
def load_dataset(uploaded_file: Any) -> pd.DataFrame:
    return pd.read_csv(uploaded_file)


def render_header(dataset_name: str | None, context: ExperimentContext | None) -> None:
    st.title("AgentLab AI")
    st.caption("Your AI Data Science Team")

    if dataset_name is None:
        return

    columns = st.columns(3)
    columns[0].metric("Dataset Name", dataset_name)
    columns[1].metric("Problem Type", title_case(context.problem_type) if context else "Pending")
    columns[2].metric("Status", "Complete" if context else "Ready")


def render_dataset_summary(
    dataframe: pd.DataFrame,
    target_column: str,
    context: ExperimentContext | None,
) -> None:
    st.subheader("Dataset Summary")

    metrics = [
        ("Rows", f"{dataframe.shape[0]:,}"),
        ("Columns", f"{dataframe.shape[1]:,}"),
        ("Missing Values", f"{int(dataframe.isna().sum().sum()):,}"),
        ("Duplicate Rows", f"{int(dataframe.duplicated().sum()):,}"),
        ("Target Column", target_column),
        ("Problem Type", title_case(context.problem_type) if context else "Pending"),
        ("Memory Usage", format_bytes(dataframe.memory_usage(deep=True).sum())),
    ]

    for row_start in range(0, len(metrics), 4):
        columns = st.columns(4)
        for column, (label, value) in zip(columns, metrics[row_start : row_start + 4]):
            column.metric(label, value)


def render_dataset_preview(dataframe: pd.DataFrame) -> None:
    st.subheader("Dataset Preview")

    visible = dataframe
    if st.checkbox("Show Column Filter", value=False):
        selected_columns = st.multiselect("Columns", list(dataframe.columns))
        if selected_columns:
            visible = dataframe[selected_columns]

    st.dataframe(visible, use_container_width=True, height=420)


def create_context(
    dataframe: pd.DataFrame,
    dataset_name: str,
    target_column: str,
    optimization_goal: str,
    test_size: float,
    business_goal: str | None,
) -> ExperimentContext:
    return ExperimentContext(
        dataset=dataframe,
        target_column=target_column,
        optimization_goal=optimization_goal,
        test_size=test_size,
        business_goal=business_goal,
        dataset_name=dataset_name,
    )


def run_experiment(context: ExperimentContext) -> None:
    execution_placeholder = st.empty()

    def on_progress(agent_name: str, status: str, current_context: ExperimentContext) -> None:
        with execution_placeholder.container():
            st.subheader("Agent Execution")
            render_agent_execution(current_context, active_agent=agent_name, active_status=status)

    with execution_placeholder.container():
        st.subheader("Agent Execution")
        render_agent_execution(context)

    try:
        PlannerAgent(progress_callback=on_progress).execute(context)
    except Exception as exc:
        st.error(f"Experiment stopped: {exc}")

    st.session_state.current_context = context
    st.session_state.experiment_history.insert(0, context.summary())
    execution_placeholder.empty()


def render_agent_execution(
    context: ExperimentContext,
    active_agent: str | None = None,
    active_status: str | None = None,
) -> None:
    if not context.agent_runs:
        st.write("Starting experiment...")
        return

    for index, run in enumerate(context.agent_runs):
        label, default_message = AGENT_LABELS.get(run.name, (run.name, "Working..."))
        status = active_status if active_agent == run.name and active_status else run.status
        icon = "✓" if status == "completed" else "●"
        status_text = "Completed" if status == "completed" else default_message

        st.markdown(f"**{icon} {label}**")
        st.caption(status_text)
        if status == "completed":
            st.caption(f"Execution Time: {execution_duration(run.to_dict())}")
        if index < len(context.agent_runs) - 1:
            st.caption("↓")


def render_results(context: ExperimentContext) -> None:
    overview_tab, metrics_tab, charts_tab, report_tab = st.tabs(
        ["Overview", "Metrics", "Charts", "Report"]
    )

    with overview_tab:
        render_cleaning_summary(context)
        render_leaderboard(context)
        render_best_model(context)
        render_ai_review(context)

    with metrics_tab:
        render_metric_cards(context)
        render_classification_report(context)
        render_feature_importance(context)

    with charts_tab:
        render_charts(context)

    with report_tab:
        render_report_download(context)
        render_history(context)


def render_cleaning_summary(context: ExperimentContext) -> None:
    st.subheader("Cleaning Summary")

    changes = context.cleaning_changes
    if not changes or _only_noop_cleaning(changes):
        st.write("No Changes Required")
        return

    render_change_table(changes, "Removed Duplicate Rows", "Removed Duplicate Row")
    render_change_table(changes, "Filled Missing Values", "Filled Missing Value")
    render_change_table(changes, "Corrected Invalid Values", "Corrected Invalid Value")
    render_change_table(changes, "Removed Empty Rows", "Removed Missing Target Row")
    render_change_table(changes, "Removed Empty Columns", "Removed Empty Column")

    with st.expander("Before Vs After Cleaning", expanded=False):
        before, after = st.columns(2)
        before.caption("Before Cleaning")
        before.dataframe(context.dataset, use_container_width=True, height=260)
        after.caption("After Cleaning")
        after.dataframe(context.active_dataset, use_container_width=True, height=260)


def render_change_table(
    changes: list[dict[str, Any]],
    heading: str,
    change_type: str,
) -> None:
    rows = [change for change in changes if change.get("change") == change_type]
    if not rows:
        return

    st.write(heading)
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def render_leaderboard(context: ExperimentContext) -> None:
    st.subheader("Model Leaderboard")

    leaderboard = context.metadata.get("leaderboard", [])
    if not leaderboard:
        st.info("No leaderboard available yet.")
        return

    frame = pd.DataFrame(leaderboard).reset_index(drop=True)
    frame.insert(0, "Rank", range(1, len(frame) + 1))
    frame["Model"] = frame["model"].apply(lambda value: title_case(str(value).replace("_", " ")))
    frame = frame.drop(columns=["model"])
    frame = frame.rename(columns={column: title_case(column.replace("_", " ")) for column in frame.columns})
    st.dataframe(frame, use_container_width=True, hide_index=True)


def render_best_model(context: ExperimentContext) -> None:
    st.subheader("Best Model")

    if not context.best_model:
        st.info("No best model selected yet.")
        return

    metrics = context.metrics.get(context.best_model, {})
    primary_metric = context.metadata.get("primary_metric", "score")

    columns = st.columns(4)
    columns[0].metric("Model", title_case(context.best_model.replace("_", " ")))
    columns[1].metric("Selected Metric", title_case(primary_metric.replace("_", " ")))
    columns[2].metric("Score", format_metric_value(metrics.get(primary_metric)))
    columns[3].metric("Training Time", f"{metrics.get('training_seconds', 0):.3f} Sec")
    st.write(best_model_reason(context, metrics, primary_metric))


def render_ai_review(context: ExperimentContext) -> None:
    st.subheader("AI Review")

    if not context.critic_analysis:
        st.info("No AI review available yet.")
        return

    for recommendation in context.critic_analysis.get("recommendations", []):
        st.write(f"- {recommendation}")

    business_recommendation = context.critic_analysis.get("business_recommendation")
    if business_recommendation:
        st.subheader("Business Recommendation")
        st.write(business_recommendation)


def render_metric_cards(context: ExperimentContext) -> None:
    st.subheader("Evaluation Metrics")

    if not context.best_model:
        st.info("No metrics available yet.")
        return

    metrics = context.metrics.get(context.best_model, {})
    metric_names = (
        ["accuracy", "precision", "recall", "f1", "roc_auc", "matthews_corrcoef"]
        if context.problem_type == "classification"
        else ["mae", "mse", "rmse", "r2", "mape"]
    )

    columns = st.columns(3)
    for index, metric_name in enumerate(metric_names):
        columns[index % 3].metric(
            title_case(metric_name.replace("_", " ")),
            format_metric_value(metrics.get(metric_name)),
        )


def render_classification_report(context: ExperimentContext) -> None:
    if context.problem_type != "classification" or not context.best_model:
        return

    report = context.classification_reports.get(context.best_model, [])
    if not report:
        return

    st.subheader("Classification Report")
    st.dataframe(pd.DataFrame(report), use_container_width=True, hide_index=True)


def render_feature_importance(context: ExperimentContext) -> None:
    if not context.best_model:
        return

    rows = context.feature_importances.get(context.best_model, [])
    if not rows:
        return

    st.subheader("Feature Importance")
    frame = pd.DataFrame(rows)
    frame["Contribution"] = frame["contribution"].apply(lambda value: f"{value:.1%}")
    frame = frame.rename(columns={"feature": "Feature", "importance": "Importance"})
    st.dataframe(frame[["Feature", "Contribution", "Importance"]], use_container_width=True, hide_index=True)


def render_charts(context: ExperimentContext) -> None:
    if not context.charts:
        st.info("No charts available yet.")
        return

    chart_items = list(context.charts.items())
    for index in range(0, len(chart_items), 2):
        columns = st.columns(2)
        for column, (title, chart_path) in zip(columns, chart_items[index : index + 2]):
            with column:
                st.caption(title_case(title.replace("_", " ")))
                path = Path(chart_path)
                if path.exists():
                    st.image(str(path), use_container_width=True)
                else:
                    st.warning(f"Chart file not found: {chart_path}")


def render_report_download(context: ExperimentContext) -> None:
    st.subheader("Download Report")

    if not context.html_report_path and not context.pdf_report_path:
        st.info("No report available yet.")
        return

    columns = st.columns(2)
    for column, (label, path_value, mime) in zip(
        columns,
        [
            ("Download HTML Report", context.html_report_path, "text/html"),
            ("Download PDF Report", context.pdf_report_path, "application/pdf"),
        ],
    ):
        if not path_value:
            continue

        report_path = Path(path_value)
        if not report_path.exists():
            column.warning(f"Report file not found: {path_value}")
            continue

        column.download_button(
            label=label,
            data=report_path.read_bytes(),
            file_name=report_path.name,
            mime=mime,
            use_container_width=True,
        )


def render_history(context: ExperimentContext) -> None:
    history = load_persisted_history() or st.session_state.experiment_history
    if not history:
        return

    with st.expander("Experiment History", expanded=False):
        st.dataframe(history_frame(history), use_container_width=True, hide_index=True)


def load_persisted_history() -> list[dict[str, Any]]:
    database_path = Path("database/experiments.sqlite")
    if not database_path.exists():
        return []

    with sqlite3.connect(database_path) as connection:
        rows = connection.execute(
            """
            SELECT payload_json
            FROM experiments
            ORDER BY created_at DESC
            LIMIT 25
            """
        ).fetchall()

    return [json.loads(row[0]) for row in rows]


def history_frame(history: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Created At": item.get("created_at"),
                "Dataset": item.get("dataset_name"),
                "Target": item.get("target_column"),
                "Problem Type": title_case(item.get("problem_type") or "Pending"),
                "Best Model": title_case(str(item.get("best_model") or "Pending").replace("_", " ")),
                "Score": format_metric_value(item.get("best_model_score")),
                "Report": item.get("html_report_path") or item.get("report_path"),
            }
            for item in history
        ]
    )


def best_model_reason(
    context: ExperimentContext,
    metrics: dict[str, float],
    primary_metric: str,
) -> str:
    model = title_case((context.best_model or "selected model").replace("_", " "))
    metric_label = title_case(primary_metric.replace("_", " "))
    score = format_metric_value(metrics.get(primary_metric))

    if context.optimization_goal == "fastest_model":
        return f"{model} was selected because it produced the fastest training time while completing the experiment successfully."
    if context.optimization_goal == "most_explainable":
        return f"{model} was selected because it balances explainability with competitive {metric_label} performance."
    return f"{model} was selected because it achieved the strongest {metric_label} score of {score} among the tested models."


def execution_duration(run: dict[str, Any]) -> str:
    started_at = pd.to_datetime(run.get("started_at"))
    finished_at = pd.to_datetime(run.get("finished_at"))
    if pd.isna(started_at) or pd.isna(finished_at):
        return "Pending"
    return f"{(finished_at - started_at).total_seconds():.2f} Sec"


def format_metric_value(value: Any) -> str:
    if value is None:
        return "Pending"
    if isinstance(value, (int, float)):
        if -1 <= value <= 1:
            return f"{value:.2%}"
        return f"{value:.4f}"
    return str(value)


def format_bytes(size: int) -> str:
    units = ["B", "KB", "MB", "GB"]
    value = float(size)
    for unit in units:
        if value < 1024:
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TB"


def title_case(value: Any) -> str:
    return str(value or "Pending").title()


def _only_noop_cleaning(changes: list[dict[str, Any]]) -> bool:
    return all(change.get("change") == "No Destructive Cleaning Required" for change in changes)


if __name__ == "__main__":
    main()
