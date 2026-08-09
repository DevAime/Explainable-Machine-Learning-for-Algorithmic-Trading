"""
labeling.py

Target definition for the SPY 5-minute direction classifier.

Target: 3-class direction of forward return over N bars (default N=12, i.e.
1 hour ahead on 5-min bars).

Classes: up / down / flat
Thresholds: derived from the *forward* N-bar return distribution's standard
deviation (not fixed arbitrary values). A bar is "flat" if its forward return
falls within +/- k * std of the forward-return distribution; k defaults to
0.5 and is a tunable parameter.

Boundary handling: a label is invalid and must be DROPPED (not imputed) if
the N-bar horizon would cross into the next trading day. We detect this
using `date` (via `is_session_open` to find day boundaries) rather than
naively using calendar time, since the dataset only contains regular
trading-hours bars (gaps between 16:00 one day and 09:30 the next are
expected and are not missing data).

Usage:
    from labeling import add_labels
    df_labeled = add_labels(df, n_bars=12, flat_std_mult=0.5)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


LABEL_MAP = {0: "down", 1: "flat", 2: "up"}
LABEL_MAP_INV = {v: k for k, v in LABEL_MAP.items()}


def _validate_input(df: pd.DataFrame) -> None:
    required = {"close", "date", "is_session_open"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input dataframe missing required columns: {missing}")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Input dataframe must have a DatetimeIndex.")
    if not df.index.is_monotonic_increasing:
        raise ValueError(
            "Input dataframe index must be sorted ascending (chronological) "
            "before labeling — forward-looking horizon logic assumes row i+1 "
            "is the next bar in time."
        )


def compute_forward_return(df: pd.DataFrame, n_bars: int) -> pd.Series:
    """
    Forward log return over the next n_bars, computed on close price.
    fwd_return[t] = log(close[t + n_bars] / close[t])

    This is NOT yet masked for day-boundary crossings — that happens in
    add_labels via the validity mask. Kept separate so it can be unit
    tested / inspected independently.
    """
    close = df["close"]
    fwd_close = close.shift(-n_bars)
    fwd_return = np.log(fwd_close / close)
    return fwd_return


def compute_horizon_validity_mask(df: pd.DataFrame, n_bars: int) -> pd.Series:
    """
    True where a forward n_bars horizon starting at this row stays within
    the SAME trading day (i.e. does not cross into the next session).

    Method: for each row t, the label horizon touches rows t, t+1, ..., t+n_bars.
    We find the day boundary positions (index positions where is_session_open
    is True) and require that the entire window [t, t+n_bars] falls before
    the next day-boundary position after t. Equivalently: 'date' at position
    t+n_bars must equal 'date' at position t. Additionally, t+n_bars must
    be a valid position within the dataframe (not run off the end).
    """
    n = len(df)
    dates = df["date"].to_numpy()

    valid = np.zeros(n, dtype=bool)
    end_positions = np.arange(n) + n_bars
    in_bounds = end_positions < n  # strictly less than n (need close at t+n_bars)

    # Only compare where in_bounds True; otherwise leave False (dropped)
    idx_in_bounds = np.where(in_bounds)[0]
    end_idx = end_positions[idx_in_bounds]
    same_day = dates[idx_in_bounds] == dates[end_idx]
    valid[idx_in_bounds] = same_day

    return pd.Series(valid, index=df.index, name="horizon_valid")


def compute_flat_thresholds(fwd_return: pd.Series, flat_std_mult: float) -> tuple[float, float]:
    """
    Derive up/down/flat thresholds from the std of the forward return
    distribution (computed over valid, non-NaN forward returns only).

    Returns (lower_threshold, upper_threshold). A forward return r is:
        flat  if lower <= r <= upper
        down  if r < lower
        up    if r > upper
    """
    valid_returns = fwd_return.dropna()
    if len(valid_returns) < 30:
        raise ValueError(
            "Too few valid forward returns to estimate a stable std "
            f"({len(valid_returns)} available); need at least 30."
        )
    std = valid_returns.std()
    lower = -flat_std_mult * std
    upper = flat_std_mult * std
    return lower, upper


def add_labels(
    df: pd.DataFrame,
    n_bars: int = 12,
    flat_std_mult: float = 0.5,
    label_col: str = "target",
) -> pd.DataFrame:
    """
    Adds forward-return-based 3-class labels to df.

    Adds columns:
        fwd_return_raw   : raw forward log return (may be NaN / span days)
        horizon_valid     : bool, True if the n_bars horizon stays within
                             the same trading day and doesn't run off the
                             end of the dataframe
        {label_col}       : int in {0,1,2} -> down/flat/up, NaN-equivalent
                             (pd.NA) where horizon_valid is False
        {label_col}_name  : string label ("down"/"flat"/"up"), None where
                             invalid

    Rows where horizon_valid is False are NOT dropped from the returned
    dataframe (so the caller can inspect them / align with feature rows);
    instead the label columns are set to null. Callers should filter on
    horizon_valid == True before training.

    Thresholds (lower, upper) are also returned via the dataframe's attrs
    dict (df.attrs['label_thresholds']) for logging/reproducibility.
    """
    _validate_input(df)

    fwd_return = compute_forward_return(df, n_bars)
    valid_mask = compute_horizon_validity_mask(df, n_bars)

    # Thresholds must be computed only from returns that are both valid
    # (same-day horizon) AND non-NaN (fwd close/return actually exists).
    usable_returns = fwd_return.where(valid_mask)
    lower, upper = compute_flat_thresholds(usable_returns, flat_std_mult)

    labels = pd.Series(pd.array([pd.NA] * len(df), dtype="Int64"), index=df.index)
    up_mask = valid_mask & (fwd_return > upper)
    down_mask = valid_mask & (fwd_return < lower)
    flat_mask = valid_mask & (fwd_return >= lower) & (fwd_return <= upper)

    labels[down_mask] = LABEL_MAP_INV["down"]
    labels[flat_mask] = LABEL_MAP_INV["flat"]
    labels[up_mask] = LABEL_MAP_INV["up"]

    label_names = labels.map(LABEL_MAP).astype("object")
    label_names[~valid_mask] = None

    out = df.copy()
    out["fwd_return_raw"] = fwd_return
    out["horizon_valid"] = valid_mask
    out[label_col] = labels
    out[f"{label_col}_name"] = label_names

    out.attrs["label_thresholds"] = {"lower": lower, "upper": upper}
    out.attrs["label_n_bars"] = n_bars
    out.attrs["label_flat_std_mult"] = flat_std_mult

    return out


def label_summary(df_labeled: pd.DataFrame, label_col: str = "target") -> pd.DataFrame:
    """Quick class-balance + drop-rate report for sanity checking."""
    total = len(df_labeled)
    valid = df_labeled["horizon_valid"].sum()
    dropped = total - valid
    counts = df_labeled.loc[df_labeled["horizon_valid"], label_col].map(LABEL_MAP).value_counts()
    summary = counts.to_frame("count")
    summary["pct_of_valid"] = (summary["count"] / valid * 100).round(2)
    print(f"Total rows: {total}")
    print(f"Valid (same-day horizon) rows: {valid}")
    print(f"Dropped (crosses day boundary or runs off end): {dropped} "
          f"({dropped / total * 100:.2f}%)")
    print(f"Thresholds: {df_labeled.attrs.get('label_thresholds')}")
    print(summary)
    return summary


if __name__ == "__main__":
    # Quick smoke test against the sample data
    df = pd.read_parquet("/home/claude/spy-thesis-project/data/spy_5min_sample.parquet")
    df = df.sort_index()
    labeled = add_labels(df, n_bars=12, flat_std_mult=0.5)
    label_summary(labeled)

    # Sanity check: every row flagged invalid should indeed cross a day
    # boundary or run off the end of the frame.
    n_bars = 12
    invalid_rows = labeled[~labeled["horizon_valid"]]
    print(f"\nInvalid row count: {len(invalid_rows)} (expected ~= 42 days * {n_bars} "
          f"end-of-day bars, minus edge effects at the very end of the sample)")
