"""
train_lstm.py

Trains the LSTM comparison baseline under the SAME walk-forward folds as
XGBoost, so the two models' results are directly comparable. Per spec,
this is an HONEST comparison baseline -- it is not assumed to beat XGBoost;
the diagnostics (weak/near-negligible raw return autocorrelation) suggest
it likely won't, and that's a valid, reportable finding.

Sequence construction:
- Lookback window = 24 bars (2 hours) by default.
- Reduced feature set (per spec): log_return, vwap_deviation, rsi_14,
  atr_14, MACDh_12_26_9, BBP_20_2.0, ADX_14, garch_vol, session_sin,
  session_cos.
- A sequence for row t uses bars [t-lookback+1, ..., t] as input to predict
  the label AT row t (same target as XGBoost: forward n_bars direction from
  row t). This means the sequence's own last bar is the "current" bar the
  signal would be generated at -- consistent with how the Streamlit app
  will present "signal as of this bar."
- Sequences ARE ALLOWED to look back across a prior day's bars (e.g. a
  sequence starting at 09:35 pulls some bars from the prior session's
  close). This is real past information available at prediction time, not
  leakage -- only the FORWARD label horizon is day-boundary-restricted
  (per labeling.py). Sequences that would run off the very start of the
  dataset (insufficient history) are dropped.
- NaN handling: unlike XGBoost, Keras/LSTM cannot handle NaN natively. Rows
  with NaN in any LSTM feature (indicator warm-up at the very start of the
  dataset, or is_session_open bars where log_return is NaN) are forward-
  filled within the feature matrix before windowing; any sequence that
  still contains NaN after fill (i.e. NaN at the very start of history,
  before any real value exists) is dropped.
- Scaling: StandardScaler fit ONLY on each fold's training sequences, then
  applied to that fold's test sequences -- refit per fold to avoid any
  cross-fold leakage, exactly mirroring how XGBoost is retrained per fold.

Outputs:
    lstm_model.h5                    - final model trained on full valid history
    lstm_scaler.pkl                  - StandardScaler fit on full training data
                                        (needed to preprocess new data at inference)
    walk_forward_results.csv         - appended with model="lstm" rows
                                        (merged with XGBoost's rows if present)
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, log_loss

from labeling import add_labels
from features import LSTM_FEATURE_LIST, save_feature_list, get_xgboost_feature_list
from walk_forward import generate_walk_forward_folds, Fold

MODELS_DIR = Path(__file__).parent
DEFAULT_DATA_PATH = MODELS_DIR.parent / "data" / "spy_5min_cleaned_features.parquet"

LOOKBACK = 24  # bars = 2 hours on 5-min data


def build_sequences(
    df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str,
    lookback: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.Index]:
    """
    Builds (X, y, dates_at_t, index_at_t) across the WHOLE dataframe.
    X[i] = feature_cols values for rows [t-lookback+1 .. t] (shape: lookback, n_features)
    y[i] = label at row t
    dates_at_t[i] = the 'date' at row t (used later to assign sequences to
                    train/test by walk-forward fold, matching XGBoost's
                    day-based splitting)

    Only rows where horizon_valid is True AND a full lookback window of
    history (no leading NaN) is available are included.
    """
    feat_matrix = df[feature_cols].to_numpy(dtype=np.float64)

    # Forward-fill NaNs within each feature column (handles is_session_open
    # log_return NaN, garch_vol NaN, and any early warm-up). Any row that is
    # STILL NaN afterward (i.e. no prior real value exists yet) makes any
    # sequence containing it unusable and is excluded via the finite-check below.
    feat_df = pd.DataFrame(feat_matrix, columns=feature_cols)
    feat_df = feat_df.ffill()
    feat_matrix = feat_df.to_numpy(dtype=np.float64)

    valid = df["horizon_valid"].to_numpy()
    labels = df[label_col].to_numpy()
    dates = df["date"].to_numpy()
    idx = df.index

    n = len(df)
    X_list, y_list, date_list, idx_list = [], [], [], []

    for t in range(lookback - 1, n):
        if not valid[t]:
            continue
        window = feat_matrix[t - lookback + 1: t + 1]
        if not np.isfinite(window).all():
            continue  # leading NaN with no prior real value -- drop
        X_list.append(window)
        y_list.append(labels[t])
        date_list.append(dates[t])
        idx_list.append(idx[t])

    if not X_list:
        return (np.empty((0, lookback, len(feature_cols))), np.empty((0,)),
                np.empty((0,), dtype=object), pd.Index([]))

    X = np.stack(X_list)
    y = np.array(y_list, dtype=int)
    dates_arr = np.array(date_list, dtype=object)
    return X, y, dates_arr, pd.Index(idx_list)


def build_lstm_model(lookback: int, n_features: int) -> "tf.keras.Model":
    import tensorflow as tf
    from tensorflow.keras import layers, models

    model = models.Sequential([
        layers.Input(shape=(lookback, n_features)),
        layers.LSTM(32, return_sequences=False),
        layers.Dropout(0.2),
        layers.Dense(16, activation="relu"),
        layers.Dense(3, activation="softmax"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def evaluate_fold_lstm(
    X: np.ndarray,
    y: np.ndarray,
    dates_arr: np.ndarray,
    fold: Fold,
    lookback: int,
    n_features: int,
    epochs: int = 15,
    batch_size: int = 32,
) -> dict | None:
    train_mask = np.isin(dates_arr, fold.train_days)
    test_mask = np.isin(dates_arr, fold.test_days)

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    if len(X_train) == 0 or len(X_test) == 0 or len(np.unique(y_train)) < 2:
        return None

    # Fit scaler on training data only (flatten to 2D for StandardScaler, reshape back)
    scaler = StandardScaler()
    n_train = X_train.shape[0]
    X_train_scaled = scaler.fit_transform(
        X_train.reshape(-1, n_features)
    ).reshape(n_train, lookback, n_features)
    n_test = X_test.shape[0]
    X_test_scaled = scaler.transform(
        X_test.reshape(-1, n_features)
    ).reshape(n_test, lookback, n_features)

    model = build_lstm_model(lookback, n_features)
    model.fit(
        X_train_scaled, y_train,
        epochs=epochs, batch_size=batch_size,
        verbose=0,
    )

    probs = model.predict(X_test_scaled, verbose=0)
    preds = probs.argmax(axis=1)

    acc = accuracy_score(y_test, preds)
    f1_macro = f1_score(y_test, preds, average="macro", zero_division=0)
    try:
        ll = log_loss(y_test, probs, labels=[0, 1, 2])
    except ValueError:
        ll = np.nan

    return {
        "fold_id": fold.fold_id,
        "cadence": fold.cadence,
        "window_type": fold.window_type,
        "model": "lstm",
        "train_start": str(fold.train_days[0]),
        "train_end": str(fold.train_days[-1]),
        "n_train_days": len(fold.train_days),
        "n_train_rows": len(X_train),
        "test_start": str(fold.test_days[0]),
        "test_end": str(fold.test_days[-1]),
        "n_test_days": len(fold.test_days),
        "n_test_rows": len(X_test),
        "accuracy": acc,
        "f1_macro": f1_macro,
        "log_loss": ll,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--n-bars", type=int, default=12)
    parser.add_argument("--flat-std-mult", type=float, default=0.5)
    parser.add_argument("--min-train-days", type=int, default=20)
    parser.add_argument("--lookback", type=int, default=LOOKBACK)
    parser.add_argument("--epochs", type=int, default=15)
    args = parser.parse_args()

    data_path = Path(args.data)
    print(f"Loading + labeling data from {data_path} ...")
    df = pd.read_parquet(data_path).sort_index()
    df = add_labels(df, n_bars=args.n_bars, flat_std_mult=args.flat_std_mult)

    feature_cols = LSTM_FEATURE_LIST
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"LSTM feature columns missing from data: {missing}")

    print(f"\nBuilding sequences (lookback={args.lookback} bars, "
          f"{len(feature_cols)} features)...")
    X, y, dates_arr, seq_idx = build_sequences(df, feature_cols, "target", args.lookback)
    print(f"Built {len(X)} sequences (dropped rows with insufficient history or invalid horizon)")

    if len(X) == 0:
        print("No sequences available -- dataset too small for the given lookback "
              "and min_train_days. Skipping LSTM training.")
        return

    trading_days = sorted(df["date"].unique())
    results = []
    for cadence in ("weekly", "monthly"):
        folds = generate_walk_forward_folds(
            trading_days, cadence=cadence, window_type="expanding", min_train_days=args.min_train_days
        )
        if not folds:
            print(f"  [{cadence}] no folds generated -- skipping")
            continue
        for fold in folds:
            r = evaluate_fold_lstm(X, y, dates_arr, fold, args.lookback, len(feature_cols),
                                    epochs=args.epochs)
            if r is not None:
                results.append(r)
                print(f"  [{cadence}] fold {fold.fold_id}: acc={r['accuracy']:.4f} "
                      f"f1_macro={r['f1_macro']:.4f} log_loss={r['log_loss']:.4f}")
            else:
                print(f"  [{cadence}] fold {fold.fold_id}: skipped (empty split or single-class train)")

    results_df = pd.DataFrame(results)
    results_path = MODELS_DIR / "walk_forward_results.csv"
    if results_path.exists():
        existing = pd.read_csv(results_path)
        existing = existing[existing["model"] != "lstm"]
        results_df = pd.concat([existing, results_df], ignore_index=True)
    results_df.to_csv(results_path, index=False)
    print(f"\nSaved walk-forward results -> {results_path}")

    if not results_df[results_df["model"] == "lstm"].empty:
        summary = results_df[results_df["model"] == "lstm"].groupby("cadence")[
            ["accuracy", "f1_macro", "log_loss"]
        ].mean()
        print("\nLSTM cadence comparison (mean across folds):")
        print(summary)

    # Final model: trained on full valid history
    print("\nTraining final LSTM model on full valid history...")
    scaler = StandardScaler()
    n_seq = X.shape[0]
    n_features = len(feature_cols)
    X_scaled = scaler.fit_transform(X.reshape(-1, n_features)).reshape(n_seq, args.lookback, n_features)

    final_model = build_lstm_model(args.lookback, n_features)
    final_model.fit(X_scaled, y, epochs=args.epochs, batch_size=32, verbose=0)

    model_path = MODELS_DIR / "lstm_model.h5"
    final_model.save(str(model_path))
    print(f"Saved final model -> {model_path}")

    scaler_path = MODELS_DIR / "lstm_scaler.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"Saved scaler -> {scaler_path}")

    # Merge into feature_list.json (XGBoost script writes xgboost_features;
    # this fills in lstm_features + lookback so it's a complete artifact
    # regardless of which script ran first).
    feat_list_path = MODELS_DIR / "feature_list.json"
    import json
    if feat_list_path.exists():
        payload = json.loads(feat_list_path.read_text())
    else:
        payload = {
            "xgboost_features": get_xgboost_feature_list(list(df.columns)),
            "label_config": {
                "n_bars": args.n_bars,
                "flat_std_mult": args.flat_std_mult,
                "thresholds": df.attrs.get("label_thresholds", {}),
                "label_map": {"0": "down", "1": "flat", "2": "up"},
            },
        }
    payload["lstm_features"] = feature_cols
    payload["lstm_lookback_bars"] = args.lookback
    feat_list_path.write_text(json.dumps(payload, indent=2))
    print(f"Updated {feat_list_path} with LSTM feature config")


if __name__ == "__main__":
    main()
