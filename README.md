# AgentLab AI

Your AI Data Science Team.

AgentLab AI is a hackathon-ready multi-agent ML experimentation platform. A user uploads a CSV, selects a target column and optimization goal, and the system runs a complete data science workflow through specialized agents connected by one shared `ExperimentContext`.

## Architecture

Agents never call each other directly. The `PlannerAgent` passes the same `ExperimentContext` through every agent in sequence:

```text
QA -> Cleaning -> Task Detection -> Model Selection -> Training
-> Evaluation -> Critic -> Visualization -> Report -> Memory
```

Each agent has one public method:

```python
execute(context)
```

## What It Produces

- Dataset quality report
- Cleaning summary
- Detected ML task
- Selected algorithms
- Trained model leaderboard
- Best model recommendation
- AI critique and business recommendation
- Charts
- HTML and PDF reports
- SQLite experiment history

## Run Locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Project Layout

```text
app.py
agents/
core/context.py
utils/
outputs/charts/
outputs/reports/
database/
```

## Hackathon Story

This is not a single AutoML script. It is an agentic platform where each agent owns a specific data science responsibility and records its contribution into a shared experiment state. The visible agent console, report artifacts, and critic analysis make the multi-agent workflow understandable to judges and non-technical users.
