"""
Knowledge Base Agent

Stores experiment runs and metadata in a local SQLite database.

Schema (experiments table):
- id INTEGER PRIMARY KEY
- timestamp TEXT
- name TEXT
- dataset_summary TEXT (JSON)
- target TEXT
- task_type TEXT
- models TEXT (JSON)
- best_model TEXT (JSON)
- metrics TEXT (JSON)
- report_paths TEXT (JSON)
"""
from typing import Dict, Any, List, Optional
import os
import sqlite3
import json
from datetime import datetime

DB_PATH = os.path.join("database", "experiments.db")


class KnowledgeBase:
    def __init__(self, db_path: str = DB_PATH) -> None:
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._ensure_tables()

    def _connect(self):
        return sqlite3.connect(self.db_path)

    def _ensure_tables(self) -> None:
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS experiments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    name TEXT,
                    dataset_summary TEXT,
                    target TEXT,
                    task_type TEXT,
                    models TEXT,
                    best_model TEXT,
                    metrics TEXT,
                    report_paths TEXT
                )
                """
            )
            conn.commit()

    def save_experiment(self, name: str, dataset_summary: Dict[str, Any], target: str, task_type: str, models: List[Dict[str, Any]], best_model: Dict[str, Any], metrics: Dict[str, Any], report_paths: Dict[str, Any]) -> int:
        timestamp = datetime.utcnow().isoformat()
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO experiments (timestamp, name, dataset_summary, target, task_type, models, best_model, metrics, report_paths) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    timestamp,
                    name,
                    json.dumps(dataset_summary),
                    target,
                    task_type,
                    json.dumps(models),
                    json.dumps(best_model),
                    json.dumps(metrics),
                    json.dumps(report_paths),
                ),
            )
            conn.commit()
            return cur.lastrowid

    def list_experiments(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute("SELECT id, timestamp, name, dataset_summary, target, task_type, models, best_model, metrics, report_paths FROM experiments ORDER BY id DESC LIMIT ?", (limit,))
            rows = cur.fetchall()
            results = []
            for r in rows:
                results.append({
                    "id": r[0],
                    "timestamp": r[1],
                    "name": r[2],
                    "dataset_summary": json.loads(r[3]) if r[3] else None,
                    "target": r[4],
                    "task_type": r[5],
                    "models": json.loads(r[6]) if r[6] else None,
                    "best_model": json.loads(r[7]) if r[7] else None,
                    "metrics": json.loads(r[8]) if r[8] else None,
                    "report_paths": json.loads(r[9]) if r[9] else None,
                })
            return results

    def get_experiment(self, experiment_id: int) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute("SELECT id, timestamp, name, dataset_summary, target, task_type, models, best_model, metrics, report_paths FROM experiments WHERE id = ?", (experiment_id,))
            r = cur.fetchone()
            if not r:
                return None
            return {
                "id": r[0],
                "timestamp": r[1],
                "name": r[2],
                "dataset_summary": json.loads(r[3]) if r[3] else None,
                "target": r[4],
                "task_type": r[5],
                "models": json.loads(r[6]) if r[6] else None,
                "best_model": json.loads(r[7]) if r[7] else None,
                "metrics": json.loads(r[8]) if r[8] else None,
                "report_paths": json.loads(r[9]) if r[9] else None,
            }
