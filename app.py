from __future__ import annotations

import json
import html
import sqlite3
from datetime import datetime
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

AGENT_ICONS = {
    "QA Agent": "◉",
    "Cleaning Agent": "◇",
    "Task Detection Agent": "⌁",
    "Model Selection Agent": "▦",
    "Training Agent": "△",
    "Evaluation Agent": "◎",
    "Critic Agent": "◈",
    "Visualization Agent": "▤",
    "Report Agent": "▧",
    "Memory Agent": "◫",
}

CHART_DESCRIPTIONS = {
    "model_comparison": "Side-by-side performance across every evaluated candidate.",
    "confusion": "Actual and predicted classes reveal where the model succeeds or confuses outcomes.",
    "roc": "Discrimination performance across classification thresholds.",
    "feature_importance": "Relative contribution of the most influential model inputs.",
    "missing": "Distribution of missing observations before model preprocessing.",
}


def main() -> None:
    st.set_page_config(
        page_title="AgentLab AI",
        page_icon="◈",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    initialize_state()
    inject_theme()

    st.sidebar.markdown("<div class='brand-mark'>AL</div>", unsafe_allow_html=True)
    st.sidebar.markdown("### AgentLab AI")
    st.sidebar.caption("MULTI-AGENT ML ORCHESTRATOR")
    st.sidebar.divider()

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
        width="stretch",
        disabled=dataframe is None or target_column is None,
    )

    st.sidebar.caption("Secure local execution · Artifacts stay in your workspace")

    render_header(uploaded_file.name if uploaded_file else None, st.session_state.current_context)

    if dataframe is None or uploaded_file is None or target_column is None:
        render_empty_state()
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
        with st.expander("Agent execution timeline", expanded=False):
            render_agent_execution(context)
        render_results(context)


def initialize_state() -> None:
    st.session_state.setdefault("current_context", None)
    st.session_state.setdefault("experiment_history", [])


def inject_theme() -> None:
    st.markdown(
        """
        <style>
        :root { --ink:#111827; --muted:#64748b; --line:#e5e7eb; --soft:#f8fafc; --accent:#4f46e5; }
        .stApp { background: #fbfcfe; color: var(--ink); }
        .block-container { max-width: 1440px; padding-top: 2.25rem; padding-bottom: 4rem; }
        [data-testid="stSidebar"] { border-right: 1px solid var(--line); background: #f8fafc; }
        [data-testid="stSidebar"] .block-container { padding-top: 1.5rem; }
        h1, h2, h3 { letter-spacing: -0.025em; color: var(--ink); }
        h1 { font-size: clamp(2rem, 4vw, 3.2rem) !important; }
        [data-testid="stMetric"] { background:#fff; border:1px solid var(--line); border-radius:12px; padding:1rem 1.1rem; }
        [data-testid="stMetricLabel"] { color:var(--muted); }
        [data-testid="stMetricValue"] { font-size:1.55rem; letter-spacing:-.03em; }
        [data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:12px; overflow:hidden; }
        .stButton > button, .stDownloadButton > button { border-radius:9px; min-height:2.7rem; font-weight:600; }
        .stButton > button[kind="primary"] { background:var(--ink); border-color:var(--ink); }
        .brand-mark { width:36px; height:36px; display:grid; place-items:center; border-radius:10px; background:#111827; color:white; font-weight:750; letter-spacing:-.04em; }
        .eyebrow { color:var(--accent); font-size:.72rem; font-weight:750; letter-spacing:.12em; text-transform:uppercase; margin-bottom:.5rem; }
        .hero-copy { max-width:720px; color:var(--muted); font-size:1.04rem; line-height:1.65; margin-top:-.25rem; }
        .section-copy { color:var(--muted); margin-top:-.55rem; margin-bottom:1.2rem; }
        .metric-card { min-height:132px; background:#fff; border:1px solid var(--line); border-radius:12px; padding:1rem 1.05rem; }
        .metric-icon { color:var(--accent); font-size:1rem; margin-bottom:.75rem; }
        .metric-label { color:var(--muted); font-size:.76rem; font-weight:650; text-transform:uppercase; letter-spacing:.05em; }
        .metric-value { color:var(--ink); font-size:1.35rem; font-weight:720; letter-spacing:-.025em; margin:.2rem 0; overflow-wrap:anywhere; }
        .metric-note { color:#94a3b8; font-size:.76rem; }
        .timeline { border-left:1px solid #cbd5e1; padding-left:1.2rem; margin:.25rem 0 1.1rem .45rem; }
        .timeline-card { position:relative; background:#fff; border:1px solid var(--line); border-radius:12px; padding:1rem 1.1rem; margin:0 0 .85rem; }
        .timeline-card:before { content:""; position:absolute; width:9px; height:9px; border-radius:50%; left:-1.55rem; top:1.35rem; background:#94a3b8; box-shadow:0 0 0 4px #fbfcfe; }
        .timeline-card.completed:before { background:#16a34a; }
        .timeline-card.failed:before { background:#dc2626; }
        .timeline-title { font-weight:700; }
        .status-pill { float:right; border-radius:999px; background:#f1f5f9; color:#475569; padding:.18rem .55rem; font-size:.7rem; font-weight:700; text-transform:uppercase; letter-spacing:.04em; }
        .timeline-meta { color:var(--muted); font-size:.78rem; margin:.45rem 0; }
        .timeline-output { color:#334155; font-size:.88rem; line-height:1.55; }
        .feature-card { background:linear-gradient(135deg,#111827,#1f2937); color:#fff; border-radius:16px; padding:1.5rem; margin:.5rem 0 1.5rem; }
        .feature-card .eyebrow { color:#a5b4fc; }
        .feature-card h3 { color:#fff; font-size:1.75rem; margin:.15rem 0 1.1rem; }
        .feature-grid { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:1rem; }
        .feature-label { color:#94a3b8; font-size:.72rem; text-transform:uppercase; letter-spacing:.06em; }
        .feature-value { font-weight:700; font-size:1.05rem; margin-top:.25rem; }
        .feature-reason { border-top:1px solid #374151; color:#d1d5db; margin-top:1.2rem; padding-top:1rem; line-height:1.55; }
        .insight-card { background:#fff; border:1px solid var(--line); border-radius:12px; padding:1.05rem; min-height:150px; }
        .insight-card h4 { margin:0 0 .75rem; color:var(--ink); }
        .insight-card p { color:#475569; font-size:.88rem; line-height:1.55; margin:.35rem 0; }
        .chart-copy { color:var(--muted); font-size:.84rem; min-height:2.7rem; }
        .empty-state { background:#fff; border:1px dashed #cbd5e1; border-radius:16px; padding:4rem 2rem; text-align:center; margin-top:2rem; }
        .empty-state h3 { margin:.4rem 0; }
        .empty-state p { color:var(--muted); }
        @media(max-width:900px) { .feature-grid { grid-template-columns:repeat(2,minmax(0,1fr)); } }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_empty_state() -> None:
    st.markdown(
        """<div class="empty-state"><div class="eyebrow">Ready when you are</div>
        <h3>Turn a CSV into a complete ML experiment</h3>
        <p>Upload a dataset from the sidebar, choose its target, and let the agent team take it from there.</p></div>""",
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_dataset(uploaded_file: Any) -> pd.DataFrame:
    return pd.read_csv(uploaded_file)


def render_header(dataset_name: str | None, context: ExperimentContext | None) -> None:
    st.markdown('<div class="eyebrow">Autonomous experiment workspace</div>', unsafe_allow_html=True)
    st.title("AgentLab AI")
    st.markdown(
        '<p class="hero-copy">A coordinated AI data science team that inspects, prepares, trains, evaluates, and documents your machine learning experiment.</p>',
        unsafe_allow_html=True,
    )

    if dataset_name is None:
        return

    st.divider()
    columns = st.columns([2, 1, 1])
    columns[0].metric("Dataset", dataset_name, help="Currently loaded source file")
    columns[1].metric("Problem Type", title_case(context.problem_type) if context else "Pending")
    columns[2].metric("Experiment Status", "Complete" if context else "Ready")


def render_dataset_summary(
    dataframe: pd.DataFrame,
    target_column: str,
    context: ExperimentContext | None,
) -> None:
    st.subheader("Dataset Summary")
    st.markdown('<p class="section-copy">A quick profile of the source data and experiment configuration.</p>', unsafe_allow_html=True)

    metrics = [
        ("▤", "Rows", f"{dataframe.shape[0]:,}", "Observations in source"),
        ("▦", "Columns", f"{dataframe.shape[1]:,}", "Available variables"),
        ("○", "Missing Values", f"{int(dataframe.isna().sum().sum()):,}", "Cells requiring attention"),
        ("≋", "Duplicate Rows", f"{int(dataframe.duplicated().sum()):,}", "Repeated observations"),
        ("◎", "Target Column", target_column, "Prediction objective"),
        ("⌁", "Problem Type", title_case(context.problem_type) if context else "Pending", "Detected by the strategy agent"),
        ("◫", "Memory Usage", format_bytes(dataframe.memory_usage(deep=True).sum()), "In-memory footprint"),
        ("✓", "Status", "Complete" if context else "Ready", "Experiment lifecycle"),
    ]

    for row_start in range(0, len(metrics), 4):
        columns = st.columns(4)
        for column, (icon, label, value, note) in zip(columns, metrics[row_start : row_start + 4]):
            with column:
                render_html_metric(icon, label, value, note)


def render_dataset_preview(dataframe: pd.DataFrame) -> None:
    st.subheader("Dataset Preview")
    st.markdown('<p class="section-copy">Inspect source rows before launching the experiment.</p>', unsafe_allow_html=True)

    visible = dataframe
    if st.checkbox("Show Column Filter", value=False):
        selected_columns = st.multiselect("Columns", list(dataframe.columns))
        if selected_columns:
            visible = dataframe[selected_columns]

    st.dataframe(visible, width="stretch", height=390)


def render_html_metric(icon: str, label: str, value: Any, note: str) -> None:
    st.markdown(
        f"""<div class="metric-card"><div class="metric-icon">{html.escape(icon)}</div>
        <div class="metric-label">{html.escape(str(label))}</div>
        <div class="metric-value">{html.escape(str(value))}</div>
        <div class="metric-note">{html.escape(note)}</div></div>""",
        unsafe_allow_html=True,
    )


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
            st.caption("Live orchestration · Each specialist passes its work to the next agent")
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
        st.info("Preparing the agent workspace…")
        return

    st.markdown('<div class="timeline">', unsafe_allow_html=True)
    for run in context.agent_runs:
        label, default_message = AGENT_LABELS.get(run.name, (run.name, "Working..."))
        status = active_status if active_agent == run.name and active_status else run.status
        status_text = "Completed" if status == "completed" else "Failed" if status == "failed" else "Running"
        duration = execution_duration(run.to_dict()) if status in {"completed", "failed"} else "In progress"
        output = agent_output(context, run.name, default_message if status == "running" else run.message)
        icon = AGENT_ICONS.get(run.name, "◇")
        st.markdown(
            f"""<div class="timeline-card {html.escape(status)}">
            <span class="status-pill">{html.escape(status_text)}</span>
            <div class="timeline-title">{html.escape(icon)} &nbsp;{html.escape(label)}</div>
            <div class="timeline-meta">Execution time · {html.escape(duration)}</div>
            <div class="timeline-output"><strong>Output</strong><br>{html.escape(output)}</div>
            </div>""",
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)


def agent_output(context: ExperimentContext, agent_name: str, fallback: str) -> str:
    report = context.quality_report or {}
    summary = context.cleaning_summary or {}
    mapping = {
        "QA Agent": (
            f"Inspected {report.get('shape', {}).get('rows', context.dataset.shape[0]):,} rows and "
            f"{report.get('shape', {}).get('columns', context.dataset.shape[1]):,} columns; "
            f"found {report.get('missing_values_total', int(context.dataset.isna().sum().sum())):,} missing values "
            f"and {report.get('duplicate_rows', int(context.dataset.duplicated().sum())):,} duplicate rows."
        ),
        "Cleaning Agent": (
            f"Prepared {summary.get('remaining_rows', context.active_dataset.shape[0]):,} rows and "
            f"{summary.get('remaining_columns', context.active_dataset.shape[1]):,} columns; "
            f"removed {summary.get('rows_removed', 0):,} duplicate rows and {summary.get('columns_removed', 0):,} empty columns."
        ),
        "Task Detection Agent": (context.metadata.get("task_detection") or {}).get("reason", fallback),
        "Model Selection Agent": (
            f"Selected {len(context.selected_models)} candidate models using the "
            f"{(context.metadata.get('model_selection') or {}).get('strategy', 'configured')} strategy."
        ),
        "Training Agent": (
            f"Trained {len(context.metrics)} models on {context.train_shape[0]:,} rows and evaluated on "
            f"{context.test_shape[0]:,} rows."
            if context.train_shape and context.test_shape else fallback
        ),
        "Evaluation Agent": (
            f"Ranked {len(context.metadata.get('leaderboard', []))} models; selected "
            f"{title_case((context.best_model or 'pending').replace('_', ' '))}."
        ),
        "Critic Agent": " ".join(context.critic_comments) if context.critic_comments else fallback,
        "Visualization Agent": f"Generated {len(context.charts)} experiment charts.",
        "Report Agent": f"Generated {sum(bool(path) for path in [context.html_report_path, context.pdf_report_path])} downloadable report formats.",
        "Memory Agent": f"Saved experiment record {context.memory_record_id or 'to the knowledge base'}.",
    }
    return str(mapping.get(agent_name) or fallback or "Completed successfully.")


def render_results(context: ExperimentContext) -> None:
    overview_tab, metrics_tab, charts_tab, report_tab = st.tabs(
        ["Overview", "Metrics", "Charts", "Report"]
    )

    with overview_tab:
        render_dataset_quality_report(context)
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
    st.markdown('<p class="section-copy">Transformations applied before model training.</p>', unsafe_allow_html=True)

    changes = context.cleaning_changes
    if not changes or _only_noop_cleaning(changes):
        with st.container(border=True):
            st.markdown("**✓ Dataset structure preserved**")
            st.caption("No duplicate rows, empty columns, or missing-target rows required destructive cleaning.")
            render_preprocessing_actions(context)
        return

    counts = pd.Series([change.get("change") for change in changes]).value_counts()
    action_labels = {
        "Removed Duplicate Row": "Duplicate rows removed",
        "Removed Empty Column": "Empty columns removed",
        "Removed Missing Target Row": "Rows with missing targets removed",
        "Filled Missing Value": "Missing values filled",
        "Corrected Invalid Value": "Invalid values corrected",
    }
    columns = st.columns(3)
    for index, (key, label) in enumerate(action_labels.items()):
        if counts.get(key, 0):
            columns[index % 3].success(f"✓ {label}: {int(counts[key]):,}")
    render_preprocessing_actions(context)

    render_change_table(changes, "Removed Duplicate Rows", "Removed Duplicate Row")
    render_change_table(changes, "Filled Missing Values", "Filled Missing Value")
    render_change_table(changes, "Corrected Invalid Values", "Corrected Invalid Value")
    render_change_table(changes, "Removed Empty Rows", "Removed Missing Target Row")
    render_change_table(changes, "Removed Empty Columns", "Removed Empty Column")

    with st.expander("Before Vs After Cleaning", expanded=False):
        before, after = st.columns(2)
        before.caption("Before Cleaning")
        before.dataframe(context.dataset, width="stretch", height=260)
        after.caption("After Cleaning")
        after.dataframe(context.active_dataset, width="stretch", height=260)


def render_preprocessing_actions(context: ExperimentContext) -> None:
    features = context.active_dataset.drop(columns=[context.target_column], errors="ignore")
    numeric_count = len(features.select_dtypes(include="number").columns)
    categorical_count = features.shape[1] - numeric_count
    notes = []
    if features.isna().any().any():
        notes.append("Missing feature values handled during model preprocessing")
    if categorical_count:
        notes.append(f"{categorical_count} categorical feature(s) encoded")
    if numeric_count:
        notes.append(f"{numeric_count} numerical feature(s) standardized")
    if notes:
        st.caption(" · ".join(f"✓ {note}" for note in notes))


def render_dataset_quality_report(context: ExperimentContext) -> None:
    report = context.quality_report or {}
    if not report:
        return

    st.subheader("Dataset Quality Report")
    st.markdown('<p class="section-copy">Structural health captured by the data inspector before cleaning.</p>', unsafe_allow_html=True)
    dataframe = context.dataset
    missing = int(report.get("missing_values_total", dataframe.isna().sum().sum()))
    duplicates = int(report.get("duplicate_rows", dataframe.duplicated().sum()))
    total_cells = max(dataframe.shape[0] * dataframe.shape[1], 1)
    health = max(0.0, 100.0 - ((missing + duplicates * dataframe.shape[1]) / total_cells * 100))
    numeric = len(dataframe.select_dtypes(include="number").columns)
    categorical = dataframe.shape[1] - numeric
    metrics = [
        ("◉", "Dataset Health", f"{health:.1f}%", "Structural completeness score"),
        ("▤", "Rows", f"{dataframe.shape[0]:,}", "Before cleaning"),
        ("▦", "Columns", f"{dataframe.shape[1]:,}", "Before cleaning"),
        ("○", "Missing Values", f"{missing:,}", "Across all cells"),
        ("≋", "Duplicate Rows", f"{duplicates:,}", "Repeated records"),
        ("#", "Numerical Columns", f"{numeric:,}", "Numeric dtypes"),
        ("Aa", "Categorical Columns", f"{categorical:,}", "Non-numeric dtypes"),
        ("◫", "Memory Usage", format_bytes(dataframe.memory_usage(deep=True).sum()), "Deep memory footprint"),
    ]
    for offset in range(0, len(metrics), 4):
        for column, metric in zip(st.columns(4), metrics[offset:offset + 4]):
            with column:
                render_html_metric(*metric)

    dtype_summary = pd.Series(report.get("data_types", {})).value_counts().rename_axis("Data Type").reset_index(name="Columns")
    with st.expander("Data types summary", expanded=False):
        st.dataframe(dtype_summary, width="stretch", hide_index=True)


def render_change_table(
    changes: list[dict[str, Any]],
    heading: str,
    change_type: str,
) -> None:
    rows = [change for change in changes if change.get("change") == change_type]
    if not rows:
        return

    st.write(heading)
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


def render_leaderboard(context: ExperimentContext) -> None:
    st.subheader("Model Leaderboard")
    st.markdown('<p class="section-copy">Candidate models ranked against the selected optimization objective.</p>', unsafe_allow_html=True)

    leaderboard = context.metadata.get("leaderboard", [])
    if not leaderboard:
        st.info("No leaderboard available yet.")
        return

    frame = pd.DataFrame(leaderboard).reset_index(drop=True)
    frame.insert(0, "Rank", range(1, len(frame) + 1))
    frame["Model"] = frame["model"].apply(lambda value: title_case(str(value).replace("_", " ")))
    frame = frame.drop(columns=["model"])
    frame = frame.rename(columns={column: title_case(column.replace("_", " ")) for column in frame.columns})
    st.dataframe(frame, width="stretch", hide_index=True)


def render_best_model(context: ExperimentContext) -> None:
    if not context.best_model:
        st.info("No best model selected yet.")
        return

    metrics = context.metrics.get(context.best_model, {})
    primary_metric = context.metadata.get("primary_metric", "score")
    leaderboard = context.metadata.get("leaderboard", [])
    rank = next((index for index, item in enumerate(leaderboard, 1) if item.get("model") == context.best_model), 1)
    reason = best_model_reason(context, metrics, primary_metric)
    st.markdown(
        f"""<div class="feature-card"><div class="eyebrow">Best model</div>
        <h3>🏆 {html.escape(title_case(context.best_model.replace('_', ' ')))}</h3>
        <div class="feature-grid">
          <div><div class="feature-label">Optimization goal</div><div class="feature-value">{html.escape(GOAL_LABELS.get(context.optimization_goal, title_case(context.optimization_goal)))}</div></div>
          <div><div class="feature-label">Performance score</div><div class="feature-value">{html.escape(format_metric_value(metrics.get(primary_metric)))}</div></div>
          <div><div class="feature-label">Training time</div><div class="feature-value">{metrics.get('training_seconds', 0):.3f} sec</div></div>
          <div><div class="feature-label">Model rank</div><div class="feature-value">#{rank} of {len(leaderboard)}</div></div>
        </div><div class="feature-reason">{html.escape(reason)}</div></div>""",
        unsafe_allow_html=True,
    )


def render_ai_review(context: ExperimentContext) -> None:
    st.subheader("AI Review")
    st.markdown('<p class="section-copy">Evidence-based observations from the critic agent.</p>', unsafe_allow_html=True)

    if not context.critic_analysis:
        st.info("No AI review available yet.")
        return

    recommendations = context.critic_analysis.get("recommendations", [])
    risks = [item for item in recommendations if not item.lower().startswith("no major")]
    strengths = [
        f"Completed evaluation across {len(context.metrics)} candidate model(s).",
        f"Selected {title_case((context.best_model or 'a baseline').replace('_', ' '))} against the {GOAL_LABELS.get(context.optimization_goal, title_case(context.optimization_goal))} objective.",
    ]
    if not risks:
        strengths.append("No major experiment risks were detected in the automated first-pass review.")
    observations = [
        f"The task was detected as {context.problem_type} using {context.target_column} as the target.",
        f"The holdout split reserved {context.test_size:.0%} of the prepared dataset for evaluation.",
    ]
    panels = [
        ("Strengths", strengths),
        ("Potential Risks", risks or ["No material risks were flagged by the current checks."]),
        ("Observations", observations),
        ("Recommendations", recommendations),
    ]
    for offset in range(0, 4, 2):
        columns = st.columns(2)
        for column, (heading, items) in zip(columns, panels[offset:offset + 2]):
            body = "".join(f"<p>• {html.escape(str(item))}</p>" for item in items)
            column.markdown(f'<div class="insight-card"><h4>{heading}</h4>{body}</div>', unsafe_allow_html=True)

    business_recommendation = context.critic_analysis.get("business_recommendation")
    if business_recommendation:
        st.subheader("Business Recommendation")
        goal_alignment = best_model_reason(
            context,
            context.metrics.get(context.best_model or "", {}),
            context.metadata.get("primary_metric", "score"),
        )
        with st.container(border=True):
            st.markdown(f"**{GOAL_LABELS.get(context.optimization_goal, title_case(context.optimization_goal))} alignment**")
            st.write(goal_alignment)
            st.write(business_recommendation)


def render_metric_cards(context: ExperimentContext) -> None:
    st.subheader("Evaluation Metrics")
    st.markdown('<p class="section-copy">Holdout performance for the selected model.</p>', unsafe_allow_html=True)

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
    st.dataframe(pd.DataFrame(report), width="stretch", hide_index=True)


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
    st.dataframe(frame[["Feature", "Contribution", "Importance"]], width="stretch", hide_index=True)


def render_charts(context: ExperimentContext) -> None:
    if not context.charts:
        st.info("No charts available yet.")
        return

    chart_items = list(context.charts.items())
    for index in range(0, len(chart_items), 2):
        columns = st.columns(2)
        for column, (title, chart_path) in zip(columns, chart_items[index : index + 2]):
            with column:
                clean_title = title_case(title.replace("_", " "))
                description = next(
                    (copy for key, copy in CHART_DESCRIPTIONS.items() if key in title.lower()),
                    "A visual diagnostic generated from the completed experiment.",
                )
                st.markdown(f"**{clean_title}**")
                st.markdown(f'<p class="chart-copy">{html.escape(description)}</p>', unsafe_allow_html=True)
                path = Path(chart_path)
                if path.exists():
                    st.image(str(path), width="stretch")
                else:
                    st.warning(f"Chart file not found: {chart_path}")
        st.write("")


def render_report_download(context: ExperimentContext) -> None:
    st.subheader("Experiment Report")
    st.markdown('<p class="section-copy">Portable artifacts generated from this experiment run.</p>', unsafe_allow_html=True)

    if not context.html_report_path and not context.pdf_report_path:
        st.info("No report available yet.")
        return

    available_paths = [Path(path) for path in [context.html_report_path, context.pdf_report_path] if path and Path(path).exists()]
    generated_at = max((path.stat().st_mtime for path in available_paths), default=None)
    status_columns = st.columns(2)
    status_columns[0].metric("Report Status", "Ready")
    status_columns[1].metric(
        "Last Generated",
        datetime.fromtimestamp(generated_at).astimezone().strftime("%d %b %Y · %H:%M") if generated_at else "Unavailable",
    )
    st.write("")
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
            label=f"↓  {label}",
            data=report_path.read_bytes(),
            file_name=report_path.name,
            mime=mime,
            width="stretch",
        )


def render_history(context: ExperimentContext) -> None:
    history = load_persisted_history() or st.session_state.experiment_history
    if not history:
        return

    with st.expander(f"Experiment History · {len(history)} runs", expanded=False):
        st.caption("The 25 most recent experiment records from the existing knowledge base.")
        st.dataframe(history_frame(history), width="stretch", hide_index=True)


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
