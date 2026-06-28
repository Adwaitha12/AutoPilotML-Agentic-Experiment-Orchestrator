# AgentLab AI

### Multi-Agent ML Experiment Orchestrator

> A multi-agent AI platform that automates tabular machine learning experimentation—from dataset inspection to reproducible, shareable reporting.

AgentLab AI turns a CSV dataset and a prediction target into a structured machine learning experiment. A planner coordinates specialized agents through a shared experiment context, while a Streamlit dashboard makes every stage of the workflow visible and reviewable.

## Project Overview

Building a reliable machine learning baseline involves much more than fitting a model. Data must be inspected, cleaned, preprocessed, split consistently, evaluated with task-appropriate metrics, compared against alternative algorithms, documented, and recorded for future reference.

AgentLab AI coordinates these responsibilities as a sequential multi-agent workflow. Each agent owns one well-defined stage and reads from or writes to the same `ExperimentContext`. This keeps the workflow observable while avoiding direct coupling between agents.

The result is a repeatable experiment containing:

- A dataset quality profile and cleaning log
- Automatic classification or regression detection
- A task-appropriate portfolio of candidate models
- Evaluation metrics and a ranked model leaderboard
- Automated risk observations and a business-facing recommendation
- Diagnostic charts and feature-importance data when available
- Downloadable HTML and PDF reports
- A locally persisted SQLite experiment history

## Business Problem

Data scientists routinely spend substantial time on work surrounding model training:

- Inspecting dataset shape, types, missing values, duplicates, and target quality
- Cleaning structural issues and preparing mixed numerical and categorical features
- Determining whether a problem is classification or regression
- Selecting and training appropriate baseline algorithms
- Comparing model performance with consistent evaluation criteria
- Reviewing experimental risks and communicating recommendations
- Producing reports and maintaining an experiment history

AgentLab AI automates this end-to-end baseline workflow. It reduces repetitive setup and documentation effort while preserving the metrics, artifacts, and execution trail needed to review each result.

## Key Features

- **Dataset quality analysis** — Profiles rows, columns, data types, missing values, duplicates, target completeness, and target distribution.
- **Automated data cleaning** — Removes duplicate rows, fully empty columns, and rows without a target while recording applied changes.
- **Feature preprocessing** — Applies median imputation and standardization to numerical features, plus frequent-value imputation and one-hot encoding to categorical features.
- **Task detection** — Identifies classification and regression problems from target data type and cardinality.
- **Automatic model selection** — Chooses a compatible classification or regression model portfolio and supports an explainability-focused subset.
- **Multi-model training** — Trains linear or logistic models, decision trees, random forests, and gradient boosting models as appropriate for the detected task.
- **Optimization goals** — Supports best accuracy, fastest model, and most explainable experiment objectives.
- **Performance leaderboard** — Ranks trained candidates using the selected primary metric and identifies the best-performing baseline.
- **Automated experiment review** — Flags signals such as small datasets, class imbalance, unusually high classification scores, and weak regression performance.
- **Business recommendation** — Connects the selected baseline model to the experiment's stated business goal and recommends domain-specific validation.
- **Interactive dashboard** — Provides dataset previews, execution status, metrics, classification reports, feature importance, charts, and report downloads.
- **Experiment reports** — Generates HTML and PDF experiment summaries.
- **Experiment history** — Stores experiment summaries in a local SQLite database for later review.

## Multi-Agent Workflow

The planner executes agents in order and passes the same shared context through every step. Agents do not call one another directly.

```text
CSV upload and experiment configuration
                │
                ▼
        Planner Agent
     initializes orchestration
                │
                ▼
 Data Inspector Agent (QA Agent)
     profiles dataset quality
                │
                ▼
 Data Engineer Agent (Cleaning Agent)
    cleans structural data issues
                │
                ▼
 ML Strategy Agent (Task Detection Agent)
 detects classification or regression
                │
                ▼
 Model Architect Agent (Model Selection Agent)
      selects candidate models
                │
                ▼
 Experiment Agent (Training Agent)
     preprocesses and trains models
                │
                ▼
 Performance Analyst Agent (Evaluation Agent)
       ranks model performance
                │
                ▼
 AI Reviewer Agent (Critic Agent)
     reviews risks and recommendations
                │
                ▼
 Insights Agent (Visualization Agent)
       generates diagnostic charts
                │
                ▼
 Documentation Agent (Report Agent)
       creates HTML and PDF reports
                │
                ▼
 Knowledge Base Agent (Memory Agent)
       stores experiment history
```

Every pipeline agent follows the same public interface:

```python
execute(context)
```

## Technology Stack

| Area | Technology | Purpose |
|---|---|---|
| Frontend | Streamlit | Interactive experiment configuration and results dashboard |
| Backend | Python | Agent orchestration, shared experiment state, and artifact management |
| Data processing | pandas, NumPy | Tabular data loading, profiling, and transformation |
| Machine learning | scikit-learn | Preprocessing pipelines, model training, and evaluation |
| Visualization | Generated SVG charts | Leaderboards, data diagnostics, feature importance, and classification visuals |
| Database | SQLite | Local experiment-summary persistence |
| Reports | HTML, ReportLab PDF | Downloadable experiment documentation |
| Version control | Git and GitHub | Source control and collaboration |

## Project Structure

```text
AgentLab-AI/
├── app.py                     # Streamlit application and presentation layer
├── agents/
│   ├── planner.py             # Sequential agent orchestration
│   ├── qa_agent.py            # Dataset quality inspection
│   ├── cleaning_agent.py      # Structural data cleaning
│   ├── task_agent.py          # Classification/regression detection
│   ├── model_agent.py         # Candidate model selection
│   ├── training_agent.py      # Preprocessing, training, and scoring
│   ├── evaluation_agent.py    # Leaderboard and best-model selection
│   ├── critic_agent.py        # Automated experiment review
│   ├── visualization_agent.py # Diagnostic chart generation
│   ├── report_agent.py        # HTML and PDF report generation
│   └── memory_agent.py        # SQLite experiment persistence
├── core/
│   ├── context.py             # Shared ExperimentContext and execution records
│   └── orchestrator.py        # Supporting orchestration implementation
├── utils/
│   ├── helpers.py             # Artifact and filename helpers
│   ├── metrics.py             # Classification/regression metrics
│   └── preprocessing.py       # Cleaning and preprocessing pipelines
├── ui/                        # Supporting UI modules
├── datasets/
│   └── sample_loan_data.csv   # Sample tabular dataset
├── database/                  # Local SQLite experiment databases
├── outputs/                   # Generated models, charts, and reports
├── tests/                     # Agent and orchestration tests
├── requirements.txt           # Python dependencies
└── README.md
```

Generated files under `database/` and `outputs/` are created or updated as experiments run.

## Installation

### Prerequisites

- Python 3.9 or newer
- Git

### 1. Clone the repository

```bash
git clone https://github.com/Adwaitha12/AutoPilotML-Agentic-Experiment-Orchestrator.git
cd AutoPilotML-Agentic-Experiment-Orchestrator
```

### 2. Create and activate a virtual environment

macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Start the application

```bash
streamlit run app.py
```

Streamlit will display the local application URL in the terminal.

## How to Use

1. **Upload a CSV file** using the sidebar uploader.
2. **Select the target column** that the experiment should predict.
3. **Choose an optimization goal**: Best Accuracy, Fastest Model, or Most Explainable.
4. Optionally adjust the **test size** and enter a **business goal**.
5. Click **Run Experiment** to start the multi-agent workflow.
6. Follow the **agent execution timeline** as each stage completes.
7. Review the dataset quality report, cleaning summary, leaderboard, selected model, metrics, charts, and automated review.
8. Download the generated **HTML or PDF report** from the Report tab.
9. Open **Experiment History** to review locally stored experiment summaries.

> AgentLab AI currently accepts CSV datasets through the dashboard. The target column must be present and the dataset must contain data.

## Screenshots

Screenshots can be added to `docs/images/` using the suggested filenames below.

| View | Placeholder |
|---|---|
| Dashboard | `docs/images/dashboard.png` |
| Dataset Summary | `docs/images/datasetsummary.png` |
| Evaluation Metrics | `docs/images/metrics.png` |
| Experiment Charts | `docs/images/charts.png` |
| Reports and History | `docs/images/report.png` |

<!-- Replace the placeholders above with repository screenshots when available. -->

## Business Value

| Benefit | Impact |
|---|---|
| Reduced experimentation time | Automates repetitive inspection, preprocessing, baseline training, and comparison tasks |
| Improved reproducibility | Uses a consistent agent order, shared context, fixed model random states where supported, and recorded configuration |
| Better model comparison | Evaluates multiple compatible algorithms and presents a ranked leaderboard |
| Reduced manual effort | Produces quality summaries, cleaning records, metrics, charts, and recommendations in one workflow |
| Automatic reporting | Generates HTML and PDF artifacts directly from experiment results |
| Professional documentation | Preserves experiment metadata and history for review and communication |

## Current Scope

AgentLab AI is designed for automated baseline experimentation on tabular CSV data. Model recommendations remain subject to domain review, validation on representative data, and production-specific acceptance criteria. Generated reports and experiment records remain local; the project does not currently deploy models to a serving environment.

## Future Enhancements

The following items are roadmap ideas and are **not part of the current implementation**:

- LLM-powered, context-aware experiment recommendations
- Cloud-hosted deployment and managed artifact storage
- Scheduled and recurring experiment execution
- Expanded explainable AI diagnostics
- Distributed and large-scale model training

## Team

| Contributor | Role | Profile |
|---|---|---|
| Karthik | Project Lead / AI Engineer | [GitHub](https://github.com/Karthik-Kandi) |
| Adwaitha | Machine Learning Engineer | [GitHub](https://github.com/Adwaitha12) |
| Praneeth | UI/UX and Product Engineer | [GitHub](https://github.com/praneethram57) |

## License

This project is licensed under the [MIT License](LICENSE).

---

Built to make machine learning experimentation more structured, visible, and repeatable.
