"""
features.py

Defines the feature sets used by each model, per the project spec:

- XGBoost (primary): all engineered/derived columns EXCEPT raw metadata
  (symbol, date, is_session_open) and raw OHLCV/vwap price levels (open,
  high, low, close, volume, vwap), since raw price level is non-stationary
  per the ADF/KPSS diagnostics already run. Derived/normalized versions
  (vwap_deviation, log_return, session_progress, etc.) are used instead.

- LSTM (comparison baseline): a reduced feature set suitable for sequence
  input, explicitly listed in the spec:
    log_return, vwap_deviation, rsi_14, atr_14, MACDh_12_26_9,
    BBP_20_2.0, ADX_14, garch_vol, session_sin, session_cos

Both lists are ORDERED — order matters because it's persisted to
feature_list.json and the model expects columns in that exact order at
inference time.
"""

from __future__ import annotations

import json
from pathlib import Path

# Columns that are raw metadata (identifiers/labels), never model inputs.
METADATA_COLS = ["symbol", "date", "is_session_open"]

# Raw price/volume columns excluded from XGBoost input because raw price
# level is non-stationary (per ADF/KPSS). Derived features (log_return,
# vwap_deviation, session_progress, etc.) are used instead.
RAW_PRICE_COLS = ["open", "high", "low", "close", "volume", "vwap"]

# Columns that are label-construction artifacts, not real-time-available
# features -- must NEVER be fed to the model (would leak the target).
LABEL_ARTIFACT_COLS = [
    "fwd_return_raw",
    "horizon_valid",
    "target",
    "target_name",
]

# minutes_since_open is redundant with session_progress/session_sin/
# session_cos (same information, worse encoding for a tree/NN) but is not
# raw price, so it's not excluded by the spec's explicit rule. We keep it
# out of the *default* XGBoost set because session_sin/session_cos/
# session_progress already encode it without the day-boundary discontinuity;
# this is called out explicitly rather than silently dropped.
REDUNDANT_TIME_COL = ["minutes_since_open"]


def get_xgboost_feature_list(df_columns: list[str]) -> list[str]:
    """
    Returns the ordered XGBoost feature list: every column in df_columns
    except metadata, raw price/volume columns, label artifacts, and the
    redundant raw minutes_since_open counter. Order follows the column
    order of df_columns (i.e. the schema's natural order) for reproducibility.
    """
    excluded = set(METADATA_COLS) | set(RAW_PRICE_COLS) | set(LABEL_ARTIFACT_COLS) | set(REDUNDANT_TIME_COL)
    return [c for c in df_columns if c not in excluded]


LSTM_FEATURE_LIST = [
    "log_return",
    "vwap_deviation",
    "rsi_14",
    "atr_14",
    "MACDh_12_26_9",
    "BBP_20_2.0",
    "ADX_14",
    "garch_vol",
    "session_sin",
    "session_cos",
]


def save_feature_list(
    xgb_features: list[str],
    lstm_features: list[str],
    n_bars: int,
    flat_std_mult: float,
    label_thresholds: dict,
    out_path: str | Path,
) -> None:
    """Persist the exact ordered feature lists + labeling config the models
    expect, so inference-time code can reconstruct identical inputs."""
    payload = {
        "xgboost_features": xgb_features,
        "lstm_features": lstm_features,
        "label_config": {
            "n_bars": n_bars,
            "flat_std_mult": flat_std_mult,
            "thresholds": label_thresholds,
            "label_map": {"0": "down", "1": "flat", "2": "up"},
        },
    }
    Path(out_path).write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    import pandas as pd

    df = pd.read_parquet("/home/claude/spy-thesis-project/data/spy_5min_sample.parquet")
    xgb_feats = get_xgboost_feature_list(list(df.columns))
    print(f"XGBoost feature count: {len(xgb_feats)}")
    for c in xgb_feats:
        print(" ", c)
    print(f"\nLSTM feature count: {len(LSTM_FEATURE_LIST)}")
    missing_lstm = [c for c in LSTM_FEATURE_LIST if c not in df.columns]
    if missing_lstm:
        print(f"WARNING: LSTM features missing from dataframe: {missing_lstm}")
    else:
        print("All LSTM features present in dataframe. OK.")
