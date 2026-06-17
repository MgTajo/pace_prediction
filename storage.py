"""
Tiny SQLite persistence layer.

Two tables: `users` (one profile per person, with their personal baseline and
heat-response settings) and `sessions` (one logged workout each).  No auth --
this is a local, single-machine app shared between friends by name.
"""

from __future__ import annotations

import sqlite3
from datetime import date, datetime
from pathlib import Path

import physiology as phys

DB_PATH = Path(__file__).with_name("pace.db")


def _conn():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    return con


def init_db():
    with _conn() as con:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                name            TEXT UNIQUE NOT NULL,
                created         TEXT NOT NULL,
                baseline_pace   REAL NOT NULL,   -- sec/km in cool conditions
                baseline_type   TEXT NOT NULL,   -- 'vo2max' | 'threshold'
                lt_fraction     REAL NOT NULL    -- personal vLT/vVO2max guess
            )
            """
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id       INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                date          TEXT NOT NULL,
                session_type  TEXT NOT NULL,     -- 'vo2max' | 'threshold'
                pace_sec      REAL NOT NULL,
                temp_c        REAL NOT NULL,
                sky           TEXT NOT NULL,
                rain          TEXT NOT NULL,
                humidity      REAL NOT NULL,
                time_of_day   TEXT,              -- 'HH:MM' (nullable) for the solar term
                notes         TEXT,
                created       TEXT NOT NULL
            )
            """
        )
        # Migration: add time_of_day to databases created before this column.
        cols = {r["name"] for r in con.execute("PRAGMA table_info(sessions)")}
        if "time_of_day" not in cols:
            con.execute("ALTER TABLE sessions ADD COLUMN time_of_day TEXT")


# ---- users ---------------------------------------------------------------

def list_users() -> list[sqlite3.Row]:
    with _conn() as con:
        return con.execute("SELECT * FROM users ORDER BY name").fetchall()


def get_user(user_id: int) -> sqlite3.Row | None:
    with _conn() as con:
        return con.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()


def create_user(name: str, baseline_pace: float, baseline_type: str,
                lt_fraction: float = phys.DEFAULT_LT_FRACTION) -> int:
    with _conn() as con:
        cur = con.execute(
            "INSERT INTO users (name, created, baseline_pace, baseline_type, lt_fraction)"
            " VALUES (?, ?, ?, ?, ?)",
            (name.strip(), datetime.now().isoformat(timespec="seconds"),
             baseline_pace, baseline_type, lt_fraction),
        )
        return cur.lastrowid


def update_user(user_id: int, baseline_pace: float, baseline_type: str,
                lt_fraction: float):
    with _conn() as con:
        con.execute(
            "UPDATE users SET baseline_pace=?, baseline_type=?, lt_fraction=? WHERE id=?",
            (baseline_pace, baseline_type, lt_fraction, user_id),
        )


def delete_user(user_id: int):
    with _conn() as con:
        con.execute("DELETE FROM users WHERE id = ?", (user_id,))


# ---- sessions ------------------------------------------------------------

def add_session(user_id: int, d: date, session_type: str, pace_sec: float,
                temp_c: float, sky: str, rain: str, humidity: float,
                time_of_day: str | None = None, notes: str = "") -> int:
    with _conn() as con:
        cur = con.execute(
            "INSERT INTO sessions (user_id, date, session_type, pace_sec, temp_c,"
            " sky, rain, humidity, time_of_day, notes, created)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (user_id, d.isoformat(), session_type, pace_sec, temp_c, sky, rain,
             humidity, time_of_day, notes,
             datetime.now().isoformat(timespec="seconds")),
        )
        return cur.lastrowid


def update_session(session_id: int, d: date, session_type: str, pace_sec: float,
                   temp_c: float, sky: str, rain: str, humidity: float,
                   time_of_day: str | None, notes: str = ""):
    """Update every editable field of an existing session."""
    with _conn() as con:
        con.execute(
            "UPDATE sessions SET date=?, session_type=?, pace_sec=?, temp_c=?,"
            " sky=?, rain=?, humidity=?, time_of_day=?, notes=? WHERE id=?",
            (d.isoformat(), session_type, pace_sec, temp_c, sky, rain, humidity,
             time_of_day, notes, session_id),
        )


def list_sessions(user_id: int) -> list[dict]:
    """Return sessions as model-ready dicts (date parsed to datetime.date)."""
    with _conn() as con:
        rows = con.execute(
            "SELECT * FROM sessions WHERE user_id = ? ORDER BY date, id", (user_id,)
        ).fetchall()
    out = []
    for r in rows:
        out.append({
            "id": r["id"],
            "date": date.fromisoformat(r["date"]),
            "session_type": r["session_type"],
            "pace_sec": r["pace_sec"],
            "temp_c": r["temp_c"],
            "sky": r["sky"],
            "rain": r["rain"],
            "humidity": r["humidity"],
            "time": r["time_of_day"],   # 'HH:MM' or None
            "notes": r["notes"] or "",
        })
    return out


def delete_session(session_id: int):
    with _conn() as con:
        con.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
