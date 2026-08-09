"""
logging_utils.py

Google Sheets logging via gspread + a service account.

- Credentials: st.secrets in deployed mode, local JSON file in dev mode,
  branched on config.APP_ENV.
- Client is cached with @st.cache_resource so we don't re-authenticate on
  every Streamlit rerun (Streamlit reruns the whole script on each
  interaction).
- Two tabs: "trials" (one row per trial submission, written immediately,
  never batched) and "sessions" (one row per participant at session start,
  so partial/abandoned sessions are still tracked).
- Duplicate-participation guard: check participant_id against the
  "sessions" tab before allowing a session to start.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

import streamlit as st
import gspread
from google.oauth2.service_account import Credentials

from config import APP_ENV, GOOGLE_SHEET_NAME, TRIALS_TAB, SESSIONS_TAB, LOCAL_CREDS_PATH

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

TRIALS_HEADER = [
    "timestamp_utc", "participant_id", "group", "scenario_id", "trial_index",
    "condition", "signal_shown", "confidence_shown", "decision", "confidence_rating",
    "response_time_seconds",
]

SESSIONS_HEADER = [
    "timestamp_utc", "participant_id", "group", "status",
]


@st.cache_resource(show_spinner=False)
def _get_client() -> gspread.Client:
    if APP_ENV == "deployed":
        creds_info = dict(st.secrets["gcp_service_account"])
        creds = Credentials.from_service_account_info(creds_info, scopes=SCOPES)
    else:
        if not LOCAL_CREDS_PATH.exists():
            raise FileNotFoundError(
                f"Local Google service-account credentials not found at "
                f"{LOCAL_CREDS_PATH}. In dev mode, place your service account "
                f"JSON key there (it's gitignored). In deployed mode, set "
                f"SPY_XAI_APP_ENV=deployed and configure st.secrets['gcp_service_account']."
            )
        creds = Credentials.from_service_account_file(str(LOCAL_CREDS_PATH), scopes=SCOPES)
    return gspread.authorize(creds)


@st.cache_resource(show_spinner=False)
def _get_spreadsheet():
    client = _get_client()
    try:
        sh = client.open(GOOGLE_SHEET_NAME)
    except gspread.SpreadsheetNotFound:
        sh = client.create(GOOGLE_SHEET_NAME)
    _ensure_tab(sh, TRIALS_TAB, TRIALS_HEADER)
    _ensure_tab(sh, SESSIONS_TAB, SESSIONS_HEADER)
    return sh


def _ensure_tab(sh, tab_name: str, header: list[str]):
    try:
        ws = sh.worksheet(tab_name)
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title=tab_name, rows=1000, cols=len(header) + 5)
        ws.append_row(header)
    return ws


def has_participated(participant_id: str) -> bool:
    """Duplicate-participation guard: check participant_id against the
    sessions log before allowing a session to start."""
    sh = _get_spreadsheet()
    ws = sh.worksheet(SESSIONS_TAB)
    existing_ids = ws.col_values(2)  # column 2 = participant_id (1-indexed, after timestamp)
    return participant_id in existing_ids


def log_session_start(participant_id: str, group: str) -> None:
    sh = _get_spreadsheet()
    ws = sh.worksheet(SESSIONS_TAB)
    ws.append_row([
        dt.datetime.utcnow().isoformat(),
        participant_id,
        group,
        "started",
    ])


def log_trial(
    participant_id: str,
    group: str,
    scenario_id: str,
    trial_index: int,
    condition: str,
    signal_shown: str,
    confidence_shown: float,
    decision: str,
    confidence_rating: int,
    response_time_seconds: float,
) -> None:
    """Writes immediately -- never batched, per spec, so partial sessions
    still leave a complete trial-level record."""
    sh = _get_spreadsheet()
    ws = sh.worksheet(TRIALS_TAB)
    ws.append_row([
        dt.datetime.utcnow().isoformat(),
        participant_id,
        group,
        scenario_id,
        trial_index,
        condition,
        signal_shown,
        confidence_shown,
        decision,
        confidence_rating,
        response_time_seconds,
    ])


def log_session_complete(participant_id: str) -> None:
    """Appends a 'completed' row for this participant (kept alongside the
    'started' row so partial vs. completed sessions are distinguishable by
    scanning for the latest status per participant_id)."""
    sh = _get_spreadsheet()
    ws = sh.worksheet(SESSIONS_TAB)
    # Look up the group for this participant from their 'started' row.
    records = ws.get_all_records()
    group = next((r["group"] for r in records if str(r["participant_id"]) == participant_id), "unknown")
    ws.append_row([
        dt.datetime.utcnow().isoformat(),
        participant_id,
        group,
        "completed",
    ])
