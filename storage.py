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

import hashlib
import hmac
import os
from datetime import date, datetime
from pathlib import Path

from sqlalchemy import (Column, Float, ForeignKey, Integer, MetaData, Table,
                        Text, create_engine, delete, func, inspect, insert,
                        select, text, update)
from sqlalchemy.exc import SQLAlchemyError

import physiology as phys

DB_PATH = Path(__file__).with_name("pace.db")

metadata = MetaData()

users = Table(
    "users", metadata,
    Column("id", Integer, primary_key=True),
    Column("name", Text, unique=True, nullable=False),
    Column("created", Text, nullable=False),
    # Legacy/vestigial: the model no longer uses a user baseline (it anchors
    # itself from the logged sessions), so these are nullable and unused.
    Column("baseline_pace", Float),                   # sec/km in cool conditions
    Column("baseline_type", Text),                    # 'vo2max' | 'threshold'
    Column("lt_fraction", Float),
    Column("pw_salt", Text),                          # per-user password salt
    Column("pw_hash", Text),                          # PBKDF2-HMAC-SHA256 hash
    # Running location (for the sun/solar-radiation geometry).
    Column("city", Text),
    Column("lat", Float),
    Column("lon", Float),
    Column("tz", Text),                               # IANA timezone name
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
    try:
        metadata.create_all(eng)  # checkfirst=True -> safe on existing databases
    except SQLAlchemyError as e:
        # Streamlit redacts the in-app message; print the real DB error to the
        # (un-redacted) server logs so the root cause is visible.
        orig = str(getattr(e, "orig", e)).lower()
        print(f"[init_db] create_all failed: {getattr(e, 'orig', e)!r}", flush=True)
        # Tables already existing is harmless; anything else is a real problem.
        if "already exists" not in orig and "duplicate" not in orig:
            raise
    # Migrations: add columns to databases created before they existed.
    _add_column_if_missing(eng, "sessions", "time_of_day", "TEXT")
    _add_column_if_missing(eng, "users", "pw_salt", "TEXT")
    _add_column_if_missing(eng, "users", "pw_hash", "TEXT")
    _add_column_if_missing(eng, "users", "city", "TEXT")
    _add_column_if_missing(eng, "users", "lat", "DOUBLE PRECISION")
    _add_column_if_missing(eng, "users", "lon", "DOUBLE PRECISION")
    _add_column_if_missing(eng, "users", "tz", "TEXT")


def _add_column_if_missing(eng, table: str, column: str, sql_type: str):
    try:
        cols = {c["name"] for c in inspect(eng).get_columns(table)}
        if column not in cols:
            with eng.begin() as con:
                con.execute(text(f"ALTER TABLE {table} ADD COLUMN {column} {sql_type}"))
    except SQLAlchemyError as e:
        print(f"[init_db] migration {table}.{column} skipped: "
              f"{getattr(e, 'orig', e)!r}", flush=True)


# ---- password hashing ----------------------------------------------------

def _hash_password(password: str, salt: str | None = None) -> tuple[str, str]:
    """Return (salt, hash) hex strings using PBKDF2-HMAC-SHA256."""
    if salt is None:
        salt = os.urandom(16).hex()
    h = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"),
                            bytes.fromhex(salt), 200_000).hex()
    return salt, h


def verify_password(user: dict, password: str) -> bool:
    """Constant-time check of a password against a stored user row."""
    if not user or not user.get("pw_salt") or not user.get("pw_hash"):
        return False
    _, h = _hash_password(password, user["pw_salt"])
    return hmac.compare_digest(h, user["pw_hash"])


def _rows(result) -> list[dict]:
    return [dict(r._mapping) for r in result]


# ---- users ---------------------------------------------------------------

def get_user(user_id: int) -> dict | None:
    with _get_engine().connect() as con:
        r = con.execute(select(users).where(users.c.id == user_id)).first()
        return dict(r._mapping) if r else None


def get_user_by_name(name: str) -> dict | None:
    """Look up a single profile by name (case-insensitive). Used for login;
    we never expose a list of all profiles."""
    with _get_engine().connect() as con:
        r = con.execute(select(users).where(
            func.lower(users.c.name) == name.strip().lower())).first()
        return dict(r._mapping) if r else None


def create_user(name: str, password: str, city: str, lat: float, lon: float,
                tz: str) -> int:
    """Create a profile from name + password + chosen running location. No
    baseline is asked for; the model anchors itself from the logged sessions.

    The baseline_* columns are vestigial (the model ignores them), but older
    databases were created with a NOT NULL constraint on them, so we write
    harmless placeholder values to stay compatible with both old and new
    schemas without a table rebuild.
    """
    salt, h = _hash_password(password)
    with _get_engine().begin() as con:
        res = con.execute(insert(users).values(
            name=name.strip(),
            created=datetime.now().isoformat(timespec="seconds"),
            baseline_pace=240.0, baseline_type="vo2max",
            lt_fraction=phys.DEFAULT_LT_FRACTION,
            pw_salt=salt, pw_hash=h,
            city=city, lat=lat, lon=lon, tz=tz))
        return int(res.inserted_primary_key[0])


def set_user_password(user_id: int, password: str):
    salt, h = _hash_password(password)
    with _get_engine().begin() as con:
        con.execute(update(users).where(users.c.id == user_id).values(
            pw_salt=salt, pw_hash=h))


def update_user_location(user_id: int, city: str, lat: float, lon: float,
                         tz: str):
    with _get_engine().begin() as con:
        con.execute(update(users).where(users.c.id == user_id).values(
            city=city, lat=lat, lon=lon, tz=tz))


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
