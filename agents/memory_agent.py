from __future__ import annotations

import json
<<<<<<< HEAD
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from agents.base_agent import BaseAgent
from core.context import ExperimentContext


class MemoryAgent(BaseAgent):
    """Persist experiment runs in SQLite and provide history retrieval."""

    def __init__(self, db_path: Optional[str] = None, logger: Optional[logging.Logger] = None) -> None:
        self.db_path = Path(db_path or "outputs/experiments.db")
        self.logger = logger or logging.getLogger(__name__)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_database()

    def _initialize_database(self) -> None:
        """Create the SQLite schema if it does not already exist."""
        with sqlite3.connect(self.db_path) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS experiments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    dataset_name TEXT,
                    problem_type TEXT,
                    best_model TEXT,
                    metrics TEXT,
                    report_path TEXT
                )
                """
            )
            connection.commit()

    def execute(self, context: ExperimentContext) -> ExperimentContext:
        """Persist the experiment state and return the updated context."""
        started_at = self._log_execution(context, "memory_agent")
        record = self.save_experiment(context)
        context.experiment_metadata["memory_record"] = record
        self._record_agent_thought(
            context,
            "memory_agent",
            f"I saved the experiment history with record id {record.get('id')}.",
        )
        self._log_completion("memory_agent", started_at)
        return context

    def run(self, context: ExperimentContext) -> ExperimentContext:
        """Backward-compatible alias for execute."""
        return self.execute(context)

    def save_experiment(self, context: ExperimentContext, dataset_name: Optional[str] = None) -> Dict[str, Any]:
        """Persist a single experiment record from the provided context."""
        record = self._build_record(context, dataset_name)
        with sqlite3.connect(self.db_path) as connection:
            cursor = connection.execute(
                """
                INSERT INTO experiments (timestamp, dataset_name, problem_type, best_model, metrics, report_path)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    record["timestamp"],
                    record["dataset_name"],
                    record["problem_type"],
                    record["best_model"],
                    json.dumps(record["metrics"]),
                    str(record["report_path"]) if record["report_path"] else None,
                ),
            )
            record["id"] = cursor.lastrowid
            connection.commit()

        self.logger.info("Saved experiment record with id %s", record["id"])
        return record

    def get_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Retrieve the most recent experiment history from SQLite."""
        query = "SELECT id, timestamp, dataset_name, problem_type, best_model, metrics, report_path FROM experiments ORDER BY timestamp DESC"
        if limit is not None:
            query += f" LIMIT {int(limit)}"

        with sqlite3.connect(self.db_path) as connection:
            rows = connection.execute(query).fetchall()

        history: List[Dict[str, Any]] = []
        for row in rows:
            history.append(
                {
                    "id": row[0],
                    "timestamp": row[1],
                    "dataset_name": row[2],
                    "problem_type": row[3],
                    "best_model": row[4],
                    "metrics": json.loads(row[5]) if row[5] else {},
                    "report_path": row[6],
                }
            )
        return history

    def _build_record(self, context: ExperimentContext, dataset_name: Optional[str]) -> Dict[str, Any]:
        """Create a serializable record from the experiment context."""
        best_model = None
        if context.best_model is not None:
            best_model = context.best_model.__class__.__name__ if hasattr(context.best_model, "__class__") else str(context.best_model)

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "dataset_name": dataset_name or self._infer_dataset_name(context),
            "problem_type": context.problem_type,
            "best_model": best_model,
            "metrics": context.metrics or {},
            "report_path": context.report_path,
        }

    def _infer_dataset_name(self, context: ExperimentContext) -> Optional[str]:
        """Try to infer a dataset name from the context or its dataset object."""
        if context.dataset is None:
            return None
        if hasattr(context.dataset, "name") and context.dataset.name:
            return str(context.dataset.name)
        return getattr(context.dataset, "__class__", type(context.dataset)).__name__


__all__ = ["MemoryAgent"]
=======
import sqlite3

from core.context import ExperimentContext
from utils.helpers import ensure_directory


class MemoryAgent:
    """Persist experiment summaries to a local SQLite history database."""

    DATABASE_PATH = "database/experiments.sqlite"

    def execute(self, context: ExperimentContext) -> ExperimentContext:
        ensure_directory("database")
        payload = context.summary()

        with sqlite3.connect(self.DATABASE_PATH) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS experiments (
                    experiment_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    dataset_name TEXT NOT NULL,
                    target_column TEXT NOT NULL,
                    problem_type TEXT,
                    best_model TEXT,
                    best_model_score REAL,
                    payload_json TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                INSERT OR REPLACE INTO experiments (
                    experiment_id,
                    created_at,
                    dataset_name,
                    target_column,
                    problem_type,
                    best_model,
                    best_model_score,
                    payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    context.experiment_id,
                    payload["created_at"],
                    context.dataset_name,
                    context.target_column,
                    context.problem_type,
                    context.best_model,
                    context.best_model_score,
                    json.dumps(payload),
                ),
            )

        context.memory_record_id = context.experiment_id
        return context
>>>>>>> origin/main
