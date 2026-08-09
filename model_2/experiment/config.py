"""
config.py

Central config for the experiment app: study parameters, paths, colors,
and the plain-language feature name mapping (kept in sync with
generate_scenarios.py's FEATURE_DISPLAY_NAMES -- duplicated here rather
than imported, so the app has zero dependency on the offline scenario
generation script / its heavier deps like xgboost+shap at runtime).
"""

from pathlib import Path

EXPERIMENT_DIR = Path(__file__).parent
SCENARIOS_PATH = EXPERIMENT_DIR / "scenarios.json"

# --- Study design ---------------------------------------------------------
CONDITIONS = ("explained", "unexplained")
GROUPS = ("group_1", "group_2")  # group_1: A explained -> B unexplained
                                   # group_2: B explained -> A unexplained
DECISION_OPTIONS = ["buy", "hold", "sell"]  # consistent with model's up/flat/down labeling
CONFIDENCE_SCALE = [1, 2, 3, 4, 5]

# Maps model signal label -> the decision option a fully-trusting participant
# would pick, used only for logging/analysis alignment, never shown to
# participants (would reveal the "correct" answer and bias behavior).
SIGNAL_TO_DECISION = {"up": "buy", "flat": "hold", "down": "sell"}

# --- Google Sheets ---------------------------------------------------------
GOOGLE_SHEET_NAME = "dsa4900_expdata"  # the spreadsheet's title in Drive
TRIALS_TAB = "trials"
SESSIONS_TAB = "sessions"
LOCAL_CREDS_PATH = EXPERIMENT_DIR / "credentials" / "gsheets_creds.json"

# --- Plain-language feature names (spec: no raw column names like
# "MACDh_12_26_9" shown to participants) -----------------------------------
FEATURE_DISPLAY_NAMES = {
    "trade_count": "Number of trades",
    "session_progress": "Time of day (session progress)",
    "session_sin": "Time of day (cyclical)",
    "session_cos": "Time of day (cyclical)",
    "vwap_deviation": "Price vs. volume-weighted average",
    "overnight_gap": "Overnight gap",
    "log_return": "Recent price momentum",
    "avg_trade_size": "Average trade size",
    "rsi_14": "RSI (momentum strength)",
    "atr_14": "Average True Range (volatility)",
    "MACD_12_26_9": "MACD line",
    "MACDh_12_26_9": "MACD momentum",
    "MACDs_12_26_9": "MACD signal line",
    "BBL_20_2.0": "Bollinger lower band",
    "BBM_20_2.0": "Bollinger midline",
    "BBU_20_2.0": "Bollinger upper band",
    "BBB_20_2.0": "Bollinger band width",
    "BBP_20_2.0": "Position within Bollinger bands",
    "obv": "On-balance volume (buying/selling pressure)",
    "garch_vol": "Modeled volatility (GARCH)",
    "vix_close_lagged": "Market-wide volatility (VIX, prior day)",
    "ADX_14": "Trend strength (ADX)",
    "DMP_14": "Upward trend pressure",
    "DMN_14": "Downward trend pressure",
    "dist_from_20d_high": "Distance from 20-day high",
    "dist_from_20d_low": "Distance from 20-day low",
    "realized_vol_intraday": "Realized intraday volatility",
}

# --- Interface: muted, restrained palette (spec: avoid default bright
# red/green so color itself doesn't bias urgency perception) ---------------
COLORS = {
    "background": "#FAFAF8",
    "panel_bg": "#FFFFFF",
    "border": "#E3E1DC",
    "text_primary": "#2B2B28",
    "text_muted": "#6F6D66",
    "accent": "#5B6B63",       # muted sage, used for primary actions/highlights
    "up_muted": "#7A9B87",     # muted green-gray, not bright green
    "down_muted": "#B0776B",   # muted terracotta, not bright red
    "flat_muted": "#9A9890",   # muted gray
    "chart_line": "#4A5A52",
    "shap_positive": "#7A9B87",
    "shap_negative": "#B0776B",
}

# Links to external consent/exit forms -- placeholders, must be replaced
# with the real Google Form URLs before deployment.
CONSENT_FORM_URL = "https://forms.gle/ybGURtajcS8EU9RE6"

# Environment: "dev" uses LOCAL_CREDS_PATH, "deployed" uses st.secrets.
# Set via an environment variable so the same code runs in both.
import streamlit as st  # noqa: E402

try:
    _has_deployed_secrets = "gcp_service_account" in st.secrets
except Exception:
    _has_deployed_secrets = False

APP_ENV = "deployed" if _has_deployed_secrets else "dev"