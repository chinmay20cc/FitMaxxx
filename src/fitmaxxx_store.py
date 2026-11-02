"""Local SQLite storage for a FitMaxxx demo profile and session summaries."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

DEFAULT_PROFILE = {
    "display_name": "Alex",
    "age": 25,
    "weight_kg": 70.0,
    "goal": "Build consistency",
    "fitness_level": "Intermediate",
    "steps_goal": 8000,
    "active_goal_min": 45,
}


def _connect(db_path: str | Path) -> sqlite3.Connection:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("""CREATE TABLE IF NOT EXISTS profile (
        id INTEGER PRIMARY KEY CHECK (id = 1), payload TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )""")
    conn.execute("""CREATE TABLE IF NOT EXISTS sessions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        saved_at TEXT NOT NULL,
        source TEXT NOT NULL,
        device TEXT,
        goal TEXT,
        steps INTEGER,
        active_minutes REAL,
        sleep_hours REAL,
        heart_rate INTEGER,
        calories_kcal REAL,
        activity_score INTEGER,
        duration_min REAL,
        top_activity TEXT
    )""")
    return conn


def load_profile(db_path: str | Path) -> dict[str, Any]:
    with _connect(db_path) as conn:
        row = conn.execute("SELECT payload FROM profile WHERE id=1").fetchone()
    if not row:
        return DEFAULT_PROFILE.copy()
    try:
        data = json.loads(row["payload"])
        return {**DEFAULT_PROFILE, **data}
    except (TypeError, json.JSONDecodeError):
        return DEFAULT_PROFILE.copy()


def save_profile(db_path: str | Path, profile: dict[str, Any]) -> None:
    payload = {**DEFAULT_PROFILE, **profile}
    with _connect(db_path) as conn:
        conn.execute(
            "INSERT INTO profile (id, payload, updated_at) VALUES (1, ?, ?) "
            "ON CONFLICT(id) DO UPDATE SET payload=excluded.payload, updated_at=excluded.updated_at",
            (json.dumps(payload), datetime.now().isoformat(timespec="seconds")),
        )


def save_session(db_path: str | Path, snapshot: dict[str, Any], profile: dict[str, Any], *, source: str = "Wearable simulation", duration_min: float | None = None, top_activity: str | None = None) -> None:
    with _connect(db_path) as conn:
        conn.execute(
            """INSERT INTO sessions (
                saved_at, source, device, goal, steps, active_minutes, sleep_hours,
                heart_rate, calories_kcal, activity_score, duration_min, top_activity
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                datetime.now().isoformat(timespec="seconds"), source,
                snapshot.get("device", "Demo device"), profile.get("goal"),
                int(snapshot.get("steps", 0)), float(snapshot.get("active_minutes", 0)),
                float(snapshot.get("sleep_hours", 0)), int(snapshot.get("current_hr", 0)),
                float(snapshot.get("calories_kcal", 0)), int(snapshot.get("activity_score", 0)),
                duration_min, top_activity,
            ),
        )


def load_sessions(db_path: str | Path):
    import pandas as pd

    with _connect(db_path) as conn:
        df = pd.read_sql_query("SELECT * FROM sessions ORDER BY saved_at", conn)
    if not df.empty:
        df["saved_at"] = pd.to_datetime(df["saved_at"])
    return df


def clear_sessions(db_path: str | Path) -> None:
    with _connect(db_path) as conn:
        conn.execute("DELETE FROM sessions")
