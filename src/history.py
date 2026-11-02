"""Tiny SQLite store for session summaries (numbers only, never raw sensor data)."""
import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd

COLUMNS = ["saved_at", "label", "duration_min", "active_min", "upright_min",
           "sedentary_min", "kcal", "score", "confidence", "top_activity", "placement"]

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    saved_at TEXT NOT NULL,
    label TEXT,
    duration_min REAL, active_min REAL, upright_min REAL, sedentary_min REAL,
    kcal REAL, score INTEGER, confidence REAL, top_activity TEXT, placement TEXT
)
"""


def _connect(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute(SCHEMA)
    return conn


def save_session(path, row, when=None):
    values = [when or datetime.now().isoformat(timespec="seconds")] + [row[c] for c in COLUMNS[1:]]
    with _connect(path) as conn:
        conn.execute(f"INSERT INTO sessions ({', '.join(COLUMNS)}) VALUES ({', '.join('?' * len(COLUMNS))})", values)


def load_sessions(path):
    with _connect(path) as conn:
        df = pd.read_sql_query("SELECT * FROM sessions ORDER BY saved_at", conn)
    df["saved_at"] = pd.to_datetime(df["saved_at"])
    return df


def clear_sessions(path):
    with _connect(path) as conn:
        conn.execute("DELETE FROM sessions")


def streak_days(df, target=70):
    """Consecutive calendar days, ending on the latest saved day, with a session scoring >= target."""
    if df.empty:
        return 0
    best = df.assign(day=df["saved_at"].dt.normalize()).groupby("day")["score"].max()
    days = sorted(best[best >= target].index)
    if not days or days[-1] != df["saved_at"].dt.normalize().max():
        return 0
    streak = 1
    for later, earlier in zip(days[::-1], days[::-1][1:]):
        if (later - earlier).days == 1:
            streak += 1
        else:
            break
    return streak
