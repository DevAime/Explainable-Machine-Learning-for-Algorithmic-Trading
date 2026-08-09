"""
train_xgboost.py

Trains the primary XGBoost direction classifier under walk-forward
validation, testing both weekly and monthly retraining cadence and
reporting which generalizes better empirically (per spec -- we don't
assume one is better).

Outputs (under /models):
    xgboost_model.json        - final model, retrained on ALL valid rows
                                 using the empirically better cadence's
                                 approach (i.e. just the full-data final
                                 fit; cadence only affects walk-forward
                                 fold construction during validation)
    feature_list.json          - exact ordered feature list + label config
    walk_forward_results.csv   - per-fold metrics for BOTH cadences,
                                  labeled by cadence, model="xgboost"

Run: python train_xgboost.py [--data PATH] [--n-bars 12] [--flat-std-mult 0.5]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, log_loss
from xgboost import XGBClassifier

from labeling import add_labels, LABEL_MAP
from features import get_xgboost_feature_list, save_feature_list
from walk_forward import generate_walk_forward_folds, Fold

MODELS_DIR = Path(__file__).parent
DEFAULT_DATA_PATH = MODELS_DIR.parent / "data" / "spy_5min_cleaned_features.parquet"


def load_and_label(data_path: Path, n_bars: int, flat_std_mult: float) -> pd.DataFrame:
    df = pd.read_parquet(data_path)
    df = df.sort_index()
    labeled = add_labels(df, n_bars=n_bars, flat_std_mult=flat_std_mult)
    return labeled


def evaluate_fold(
    df: pd.DataFrame,
    fold: Fold,
    feature_cols: list[str],
    label_col: str = "target",
) -> dict:
    """Train on fold.train_days, evaluate on fold.test_days. Only rows with
    horizon_valid == True are used for both train and test (day-boundary
    label leakage rows excluded)."""
    valid = df["horizon_valid"]

    train_mask = valid & df["date"].isin(fold.train_days)
    test_mask = valid & df["date"].isin(fold.test_days)

    train_df = df.loc[train_mask]
    test_df = df.loc[test_mask]

    if train_df.empty or test_df.empty:
        return None

    X_train = train_df[feature_cols]
    y_train = train_df[label_col].astype(int)
    X_test = test_df[feature_cols]
    y_test = test_df[label_col].astype(int)

    # Guard: need at least 2 classes present in training data for XGBoost
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
        missing=np.nan,  # XGBoost handles NaN natively (indicator warm-up, overnight_gap, etc.)
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)

    acc = accuracy_score(y_test, preds)
    f1_macro = f1_score(y_test, preds, average="macro", zero_division=0)
    try:
        # log_loss requires all classes present in y_test to match probs' columns cleanly;
        # guard against a fold whose test set happens to be single-class.
        ll = log_loss(y_test, probs, labels=[0, 1, 2])
    except ValueError:
        ll = np.nan

    return {
        "fold_id": fold.fold_id,
        "cadence": fold.cadence,
        "window_type": fold.window_type,
        "model": "xgboost",
        "train_start": str(fold.train_days[0]),
        "train_end": str(fold.train_days[-1]),
        "n_train_days": len(fold.train_days),
        "n_train_rows": len(train_df),
        "test_start": str(fold.test_days[0]),
        "test_end": str(fold.test_days[-1]),
        "n_test_days": len(fold.test_days),
        "n_test_rows": len(test_df),
        "accuracy": acc,
        "f1_macro": f1_macro,
        "log_loss": ll,
    }


def run_walk_forward_comparison(
    df: pd.DataFrame,
    feature_cols: list[str],
    min_train_days: int = 20,
) -> pd.DataFrame:
    trading_days = sorted(df["date"].unique())
    results = []

    for cadence in ("weekly", "monthly"):
        folds = generate_walk_forward_folds(
            trading_days, cadence=cadence, window_type="expanding", min_train_days=min_train_days
        )
        if not folds:
            print(f"  [{cadence}] no folds generated (not enough trading days "
                  f"given min_train_days={min_train_days}) -- skipping")
            continue
        for fold in folds:
            r = evaluate_fold(df, fold, feature_cols)
            if r is not None:
                results.append(r)
                print(f"  [{cadence}] fold {fold.fold_id}: acc={r['accuracy']:.4f} "
                      f"f1_macro={r['f1_macro']:.4f} log_loss={r['log_loss']:.4f}")
            else:
                print(f"  [{cadence}] fold {fold.fold_id}: skipped (empty split or single-class train)")

    return pd.DataFrame(results)


def compare_cadences(results_df: pd.DataFrame) -> str:
    """Empirically pick the better cadence by mean test accuracy and f1_macro
    across folds (per spec: test both, report which generalizes better,
    don't assume)."""
    if results_df.empty:
        print("No walk-forward results available to compare cadences.")
        return "weekly"  # fallback default

    summary = results_df.groupby("cadence")[["accuracy", "f1_macro", "log_loss"]].mean()
    print("\nCadence comparison (mean across folds):")
    print(summary)

    best_cadence = summary["f1_macro"].idxmax()
    print(f"\n-> Empirically better cadence (by mean f1_macro): {best_cadence}")
    return best_cadence


def train_final_model(df: pd.DataFrame, feature_cols: list[str], label_col: str = "target"):
    """Final deployed model: trained on ALL valid historical rows. Cadence
    choice from walk-forward affects how retraining WOULD be scheduled in
    production (weekly vs monthly refresh), not this one-off artifact fit."""
    valid_df = df.loc[df["horizon_valid"]]
    X = valid_df[feature_cols]
    y = valid_df[label_col].astype(int)

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
    model.fit(X, y)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--n-bars", type=int, default=12)
    parser.add_argument("--flat-std-mult", type=float, default=0.5)
    parser.add_argument("--min-train-days", type=int, default=20)
    args = parser.parse_args()

    data_path = Path(args.data)
    print(f"Loading + labeling data from {data_path} ...")
    df = load_and_label(data_path, args.n_bars, args.flat_std_mult)

    feature_cols = get_xgboost_feature_list(list(df.columns))
    print(f"\nXGBoost feature set ({len(feature_cols)} features): {feature_cols}\n")

    print("Running walk-forward validation (weekly vs monthly cadence)...")
    results_df = run_walk_forward_comparison(df, feature_cols, min_train_days=args.min_train_days)

    results_path = MODELS_DIR / "walk_forward_results.csv"
    if results_path.exists():
        # LSTM script appends its own rows; don't clobber if it ran first.
        existing = pd.read_csv(results_path)
        existing = existing[existing["model"] != "xgboost"]
        results_df = pd.concat([existing, results_df], ignore_index=True)
    results_df.to_csv(results_path, index=False)
    print(f"\nSaved walk-forward results -> {results_path}")

    xgb_results = results_df[results_df["model"] == "xgboost"]
    best_cadence = compare_cadences(xgb_results)

    print("\nTraining final XGBoost model on full valid history...")
    final_model = train_final_model(df, feature_cols)
    model_path = MODELS_DIR / "xgboost_model.json"
    final_model.save_model(str(model_path))
    print(f"Saved final model -> {model_path}")

    thresholds = df.attrs.get("label_thresholds", {})
    save_feature_list(
        xgb_features=feature_cols,
        lstm_features=[],  # populated / merged by train_lstm.py if run after
        n_bars=args.n_bars,
        flat_std_mult=args.flat_std_mult,
        label_thresholds=thresholds,
        out_path=MODELS_DIR / "feature_list.json",
    )
    print(f"Saved feature_list.json (recommended retraining cadence: {best_cadence})")


if __name__ == "__main__":
    main()
