from __future__ import annotations

import json
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
