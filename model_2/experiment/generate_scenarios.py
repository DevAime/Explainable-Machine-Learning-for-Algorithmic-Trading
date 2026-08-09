"""
generate_scenarios.py

Run ONCE, OFFLINE, to build experiment/scenarios.json. The Streamlit app
loads this static file and never calls the model live.

Critical constraint from the spec: scenarios must come from the
TEST/HOLDOUT portion of the walk-forward split, and must NEVER be scored
using a model that was trained on that scenario's own data. This script
therefore:

  1. Re-runs the walk-forward fold generation (same code as training).
  2. For EACH fold, trains a fresh XGBoost model on ONLY that fold's
     training days (not the final full-history model from train_xgboost.py).
  3. Scores that fold's test days with that fold-specific model, and
     computes SHAP TreeExplainer values with that SAME fold-specific model
     (so the explanation is honest for what actually produced the
     prediction -- using the final model's SHAP values on an early test
     fold would explain a different model than the one that generated the
     scenario's signal).
  4. Pools all (row, prediction, confidence, shap_values) tuples across all
     folds' test sets as scenario CANDIDATES.
  5. Samples ~20 candidates mixing signal class (up/down/flat) and
     confidence level (high/low), splits into two matched ~10-scenario
     sets (Set A, Set B).
  6. For each selected scenario, also pulls the preceding 2-3 hours of
     price bars (for the chart) from the raw dataframe.

Output: experiment/scenarios.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
import shap

MODELS_DIR = Path(__file__).parent.parent / "models"
EXPERIMENT_DIR = Path(__file__).parent
DATA_PATH = Path(__file__).parent.parent / "data" / "spy_5min_cleaned_features.parquet"

sys.path.insert(0, str(MODELS_DIR))
from labeling import add_labels, LABEL_MAP  # noqa: E402
from features import get_xgboost_feature_list  # noqa: E402
from walk_forward import generate_walk_forward_folds  # noqa: E402

# Human-readable feature name mapping (spec: "not raw column names like
# 'MACDh_12_26_9', relabel to 'MACD momentum'") -- used later by the
# Streamlit app's ui_components.py too, but defined once here to keep the
# scenarios.json contributions already in plain language.
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

N_SCENARIOS_TARGET = 20
CHART_LOOKBACK_BARS = 30  # ~2.5 hours of 5-min bars for chart context
CADENCE = "monthly"  # larger test windows -> more usable candidates per fold
MIN_TRAIN_DAYS = 20
CONFIDENCE_HIGH_THRESHOLD = 0.55  # max predicted-class probability >= this => "high" confidence


def train_fold_model(train_df: pd.DataFrame, feature_cols: list[str]) -> XGBClassifier | None:
    X_train = train_df[feature_cols]
    y_train = train_df["target"].astype(int)
    if y_train.nunique() < 2:
        return None
    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        n_jobs=-1,
        random_state=42,
        missing=np.nan,
    )
    model.fit(X_train, y_train)
    return model


def collect_candidates(df: pd.DataFrame, feature_cols: list[str], min_train_days: int = MIN_TRAIN_DAYS) -> list[dict]:
    trading_days = sorted(df["date"].unique())
    folds = generate_walk_forward_folds(
        trading_days, cadence=CADENCE, window_type="expanding", min_train_days=min_train_days
    )
    if not folds:
        raise RuntimeError(
            f"No walk-forward folds generated for cadence={CADENCE}, "
            f"min_train_days={min_train_days}. Not enough trading days in this dataset."
        )

    valid = df["horizon_valid"]
    candidates = []

    for fold in folds:
        train_mask = valid & df["date"].isin(fold.train_days)
        test_mask = valid & df["date"].isin(fold.test_days)
        train_df = df.loc[train_mask]
        test_df = df.loc[test_mask]
        if train_df.empty or test_df.empty:
            continue

        model = train_fold_model(train_df, feature_cols)
        if model is None:
            print(f"  fold {fold.fold_id}: skipped (single-class training data)")
            continue

        X_test = test_df[feature_cols]
        probs = model.predict_proba(X_test)
        preds = probs.argmax(axis=1)

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        # shap_values shape for multi-class TreeExplainer: (n_samples, n_features, n_classes)
        # or a list of n_classes arrays of shape (n_samples, n_features), depending on SHAP
        # version. Normalize to (n_samples, n_features, n_classes).
        if isinstance(shap_values, list):
            shap_arr = np.stack(shap_values, axis=-1)
        else:
            shap_arr = shap_values

        for i, (idx, row) in enumerate(test_df.iterrows()):
            pred_class = int(preds[i])
            confidence = float(probs[i, pred_class])
            row_shap = shap_arr[i, :, pred_class]  # contributions to the PREDICTED class
            candidates.append({
                "timestamp": idx,
                "fold_id": fold.fold_id,
                "predicted_class": pred_class,
                "predicted_label": LABEL_MAP[pred_class],
                "confidence": confidence,
                "confidence_level": "high" if confidence >= CONFIDENCE_HIGH_THRESHOLD else "low",
                "shap_values": dict(zip(feature_cols, row_shap.tolist())),
                "feature_values": dict(zip(feature_cols, row[feature_cols].tolist())),
            })

        print(f"  fold {fold.fold_id}: {len(test_df)} candidates "
              f"(test {fold.test_days[0]} -> {fold.test_days[-1]})")

    return candidates


def select_diverse_scenarios(candidates: list[dict], n_target: int) -> list[dict]:
    """Pick a mix of signal classes (up/down/flat) and confidence levels
    (high/low), roughly evenly, up to n_target scenarios."""
    if len(candidates) < n_target:
        print(f"WARNING: only {len(candidates)} candidates available, "
              f"fewer than target {n_target}. Using all of them.")
        selected = candidates
    else:
        buckets: dict[tuple, list[dict]] = {}
        for c in candidates:
            key = (c["predicted_label"], c["confidence_level"])
            buckets.setdefault(key, []).append(c)

        per_bucket = max(1, n_target // max(1, len(buckets)))
        selected = []
        rng = np.random.default_rng(42)
        for key, items in buckets.items():
            items_sorted = sorted(items, key=lambda c: c["timestamp"])
            idx = rng.choice(len(items_sorted), size=min(per_bucket, len(items_sorted)), replace=False)
            selected.extend(items_sorted[i] for i in sorted(idx))

        # Top up / trim to hit n_target as closely as possible
        if len(selected) < n_target:
            remaining = [c for c in candidates if c not in selected]
            rng.shuffle(remaining)
            selected.extend(remaining[: n_target - len(selected)])
        selected = selected[:n_target]

    selected.sort(key=lambda c: c["timestamp"])
    return selected


def attach_price_windows(df_raw: pd.DataFrame, scenarios: list[dict], lookback_bars: int) -> None:
    """Mutates each scenario dict in place, adding 'price_window': a list of
    OHLC bars for the CHART_LOOKBACK_BARS bars ending at (and including) the
    scenario's timestamp. Uses raw OHLC (not derived features) since this is
    purely for chart rendering, not model input."""
    for sc in scenarios:
        ts = sc["timestamp"]
        pos = df_raw.index.get_loc(ts)
        start_pos = max(0, pos - lookback_bars + 1)
        window = df_raw.iloc[start_pos: pos + 1]
        sc["price_window"] = [
            {
                "timestamp": str(t),
                "open": float(r["open"]),
                "high": float(r["high"]),
                "low": float(r["low"]),
                "close": float(r["close"]),
                "volume": float(r["volume"]),
            }
            for t, r in window.iterrows()
        ]


def build_scenario_payload(scenarios: list[dict]) -> list[dict]:
    payload = []
    for sc in scenarios:
        # Top contributing features (by absolute SHAP value), plain-language names
        shap_items = sorted(sc["shap_values"].items(), key=lambda kv: abs(kv[1]), reverse=True)
        top_features = [
            {
                "feature": FEATURE_DISPLAY_NAMES.get(name, name),
                "raw_column": name,
                "shap_value": float(val),
                "direction": "supports" if val > 0 else "against",
            }
            for name, val in shap_items[:8]
        ]
        payload.append({
            "scenario_id": str(sc["timestamp"]),
            "timestamp": str(sc["timestamp"]),
            "fold_id": sc["fold_id"],
            "signal": sc["predicted_label"],
            "confidence": round(sc["confidence"], 4),
            "confidence_level": sc["confidence_level"],
            "top_features": top_features,
            "price_window": sc["price_window"],
        })
    return payload


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=str(DATA_PATH))
    parser.add_argument("--n-target", type=int, default=N_SCENARIOS_TARGET)
    parser.add_argument("--min-train-days", type=int, default=MIN_TRAIN_DAYS)
    args = parser.parse_args()

    data_path = Path(args.data)
    print(f"Loading + labeling data from {data_path} ...")
    df = pd.read_parquet(data_path).sort_index()
    df_labeled = add_labels(df, n_bars=12, flat_std_mult=0.5)
    feature_cols = get_xgboost_feature_list(list(df_labeled.columns))

    print(f"\nCollecting out-of-fold candidates (cadence={CADENCE})...")
    candidates = collect_candidates(df_labeled, feature_cols, min_train_days=args.min_train_days)
    print(f"\nTotal candidates pooled across folds: {len(candidates)}")

    if not candidates:
        raise RuntimeError(
            "No scenario candidates were generated. This usually means the dataset "
            "doesn't have enough trading days for the configured cadence/min_train_days "
            "to produce any walk-forward test folds."
        )

    print(f"\nSelecting ~{args.n_target} diverse scenarios "
          f"(mix of signal class + confidence level)...")
    selected = select_diverse_scenarios(candidates, args.n_target)
    print(f"Selected {len(selected)} scenarios.")
    for sc in selected:
        print(f"  {sc['timestamp']}  {sc['predicted_label']:>4}  "
              f"conf={sc['confidence']:.3f} ({sc['confidence_level']})  fold={sc['fold_id']}")

    print("\nAttaching preceding price windows for charting...")
    attach_price_windows(df, selected, CHART_LOOKBACK_BARS)

    payload = build_scenario_payload(selected)

    # Split into two matched sets, Set A / Set B, alternating to keep both
    # sets balanced across the chronological / class mix rather than a
    # naive first-half/second-half split.
    set_a = payload[0::2]
    set_b = payload[1::2]
    for sc in set_a:
        sc["set"] = "A"
    for sc in set_b:
        sc["set"] = "B"

    out = {
        "generated_from_cadence": CADENCE,
        "n_scenarios": len(payload),
        "set_a": set_a,
        "set_b": set_b,
    }

    out_path = EXPERIMENT_DIR / "scenarios.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved {len(set_a)} Set A + {len(set_b)} Set B scenarios -> {out_path}")


if __name__ == "__main__":
    main()
