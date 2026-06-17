"""
Persistence layer (SQLAlchemy Core) -- works on both SQLite and Postgres.

Backend selection:
  * If the env var DATABASE_URL is set (e.g. the Neon Postgres connection
    string, injected from Streamlit secrets in the cloud) -> Postgres.
  * Otherwise -> a local SQLite file (pace.db), so `./run.sh` works offline
    with zero configuration and your existing local data is untouched.

Two tables: `users` (one profile per person) and `sessions` (one workout).
No auth here -- access is gated at the platform level (Streamlit Cloud viewer
allowlist); profiles only separate each person's data.
"""

from __future__ import annotations

import os
from datetime import date, datetime
from pathlib import Path

from sqlalchemy import (Column, Float, ForeignKey, Integer, MetaData, Table,
                        Text, create_engine, delete, inspect, insert, select,
                        text, update)

import physiology as phys

DB_PATH = Path(__file__).with_name("pace.db")

metadata = MetaData()

users = Table(
    "users", metadata,
    Column("id", Integer, primary_key=True),
    Column("name", Text, unique=True, nullable=False),
    Column("created", Text, nullable=False),
    Column("baseline_pace", Float, nullable=False),   # sec/km in cool conditions
    Column("baseline_type", Text, nullable=False),    # 'vo2max' | 'threshold'
    Column("lt_fraction", Float, nullable=False),
)

sessions = Table(
    "sessions", metadata,
    Column("id", Integer, primary_key=True),
    Column("user_id", Integer, ForeignKey("users.id"), nullable=False),
    Column("date", Text, nullable=False),
    Column("session_type", Text, nullable=False),     # 'vo2max' | 'threshold'
    Column("pace_sec", Float, nullable=False),
    Column("temp_c", Float, nullable=False),
    Column("sky", Text, nullable=False),
    Column("rain", Text, nullable=False),
    Column("humidity", Float, nullable=False),
    Column("time_of_day", Text),                       # 'HH:MM' (nullable)
    Column("notes", Text),
    Column("created", Text, nullable=False),
)

_engine = None


def _get_engine():
    global _engine
    if _engine is None:
        url = os.environ.get("DATABASE_URL", "").strip()
        if url:
            # SQLAlchemy 2.0 wants the 'postgresql://' scheme.
            if url.startswith("postgres://"):
                url = url.replace("postgres://", "postgresql://", 1)
            _engine = create_engine(url, pool_pre_ping=True)
        else:
            _engine = create_engine(
                f"sqlite:///{DB_PATH}",
                connect_args={"check_same_thread": False},
            )
    return _engine


def init_db():
    eng = _get_engine()
    metadata.create_all(eng)  # checkfirst=True -> safe on existing databases
    # Migration: add time_of_day to databases created before that column.
    cols = {c["name"] for c in inspect(eng).get_columns("sessions")}
    if "time_of_day" not in cols:
        with eng.begin() as con:
            con.execute(text("ALTER TABLE sessions ADD COLUMN time_of_day TEXT"))


def _rows(result) -> list[dict]:
    return [dict(r._mapping) for r in result]


# ---- users ---------------------------------------------------------------

def list_users() -> list[dict]:
    with _get_engine().connect() as con:
        return _rows(con.execute(select(users).order_by(users.c.name)))


def get_user(user_id: int) -> dict | None:
    with _get_engine().connect() as con:
        r = con.execute(select(users).where(users.c.id == user_id)).first()
        return dict(r._mapping) if r else None


def create_user(name: str, baseline_pace: float, baseline_type: str,
                lt_fraction: float = phys.DEFAULT_LT_FRACTION) -> int:
    with _get_engine().begin() as con:
        res = con.execute(insert(users).values(
            name=name.strip(),
            created=datetime.now().isoformat(timespec="seconds"),
            baseline_pace=baseline_pace, baseline_type=baseline_type,
            lt_fraction=lt_fraction))
        return int(res.inserted_primary_key[0])


def update_user(user_id: int, baseline_pace: float, baseline_type: str,
                lt_fraction: float):
    with _get_engine().begin() as con:
        con.execute(update(users).where(users.c.id == user_id).values(
            baseline_pace=baseline_pace, baseline_type=baseline_type,
            lt_fraction=lt_fraction))


def delete_user(user_id: int):
    # Explicit cascade (portable; doesn't rely on FK enforcement settings).
    with _get_engine().begin() as con:
        con.execute(delete(sessions).where(sessions.c.user_id == user_id))
        con.execute(delete(users).where(users.c.id == user_id))


# ---- sessions ------------------------------------------------------------

def add_session(user_id: int, d: date, session_type: str, pace_sec: float,
                temp_c: float, sky: str, rain: str, humidity: float,
                time_of_day: str | None = None, notes: str = "") -> int:
    with _get_engine().begin() as con:
        res = con.execute(insert(sessions).values(
            user_id=user_id, date=d.isoformat(), session_type=session_type,
            pace_sec=pace_sec, temp_c=temp_c, sky=sky, rain=rain,
            humidity=humidity, time_of_day=time_of_day, notes=notes,
            created=datetime.now().isoformat(timespec="seconds")))
        return int(res.inserted_primary_key[0])


def update_session(session_id: int, d: date, session_type: str, pace_sec: float,
                   temp_c: float, sky: str, rain: str, humidity: float,
                   time_of_day: str | None, notes: str = ""):
    """Update every editable field of an existing session."""
    with _get_engine().begin() as con:
        con.execute(update(sessions).where(sessions.c.id == session_id).values(
            date=d.isoformat(), session_type=session_type, pace_sec=pace_sec,
            temp_c=temp_c, sky=sky, rain=rain, humidity=humidity,
            time_of_day=time_of_day, notes=notes))


def list_sessions(user_id: int) -> list[dict]:
    """Return sessions as model-ready dicts (date parsed to datetime.date)."""
    with _get_engine().connect() as con:
        rows = _rows(con.execute(
            select(sessions).where(sessions.c.user_id == user_id)
            .order_by(sessions.c.date, sessions.c.id)))
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
    with _get_engine().begin() as con:
        con.execute(delete(sessions).where(sessions.c.id == session_id))
