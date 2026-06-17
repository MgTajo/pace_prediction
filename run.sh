#!/usr/bin/env bash
# One-command launcher: creates the venv on first run, then starts the app.
set -e
cd "$(dirname "$0")"

if [ ! -d ".venv" ]; then
  echo "Setting up virtual environment (first run only)…"
  python3 -m venv .venv
  ./.venv/bin/python -m pip install --upgrade pip >/dev/null
  ./.venv/bin/python -m pip install -r requirements.txt
fi

exec ./.venv/bin/streamlit run app.py
