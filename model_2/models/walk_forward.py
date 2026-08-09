"""
walk_forward.py

Walk-forward (expanding window) validation splitter, shared by both the
XGBoost and LSTM training scripts so their splits are identical and the
comparison between them is fair.

Design:
- Splits are defined on TRADING DAYS, not raw row counts, so "weekly" and
  "monthly" retraining cadence map to calendar-meaningful chunks (5 trading
  days / ~21 trading days) rather than arbitrary row counts.
- Expanding window: each fold's training set is every day strictly before
  the fold's test window (from the start of the dataset). This is the
  default. A rolling (fixed-length) window is also supported via
  `window_type="rolling"` for the empirical cadence comparison the spec
  asks for.
- A fold's TEST set is exactly one cadence period (one week or one month
  of trading days).
- Folds only include rows where `horizon_valid` is True (so no label
  leakage from day-boundary-crossing rows) -- this filtering happens in
  the training scripts, not here; this module only returns date-based
  fold boundaries so both scripts apply it identically.
- A minimum training-period length (in trading days) is enforced so early
  folds aren't trained on too little data to be meaningful.

Usage:
    folds = generate_walk_forward_folds(
        trading_days=sorted(df['date'].unique()),
        cadence="weekly",           # or "monthly"
        window_type="expanding",    # or "rolling"
        min_train_days=20,
        rolling_train_days=60,      # only used if window_type == "rolling"
    )
    # folds: list of dicts {train_days: [...], test_days: [...], fold_id: int}
"""

from __future__ import annotations

from dataclasses import dataclass, field


CADENCE_DAYS = {
    "weekly": 5,
    "monthly": 21,
}


@dataclass
class Fold:
    fold_id: int
    cadence: str
    window_type: str
    train_days: list = field(default_factory=list)
    test_days: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "fold_id": self.fold_id,
            "cadence": self.cadence,
            "window_type": self.window_type,
            "train_start": str(self.train_days[0]) if self.train_days else None,
            "train_end": str(self.train_days[-1]) if self.train_days else None,
            "n_train_days": len(self.train_days),
            "test_start": str(self.test_days[0]) if self.test_days else None,
            "test_end": str(self.test_days[-1]) if self.test_days else None,
            "n_test_days": len(self.test_days),
        }


def generate_walk_forward_folds(
    trading_days: list,
    cadence: str = "weekly",
    window_type: str = "expanding",
    min_train_days: int = 20,
    rolling_train_days: int = 60,
) -> list[Fold]:
    """
    trading_days: sorted list/array of unique calendar dates (ascending),
                   e.g. df['date'].sort_values().unique()
    cadence: "weekly" (5 trading days/fold) or "monthly" (21 trading days/fold)
    window_type: "expanding" (train on all history to date) or "rolling"
                  (train on the last `rolling_train_days` days only)
    min_train_days: minimum number of trading days required before the
                     first test fold is emitted (skip folds that would
                     have too little training history to be meaningful)
    rolling_train_days: fixed lookback length used only when
                         window_type == "rolling"

    Returns a list of Fold objects in chronological order. Each fold's
    test period is non-overlapping with every other fold's test period
    (walk-forward, not cross-validation).
    """
    if cadence not in CADENCE_DAYS:
        raise ValueError(f"cadence must be one of {list(CADENCE_DAYS)}, got {cadence!r}")
    if window_type not in ("expanding", "rolling"):
        raise ValueError(f"window_type must be 'expanding' or 'rolling', got {window_type!r}")

    trading_days = sorted(trading_days)
    step = CADENCE_DAYS[cadence]

    folds: list[Fold] = []
    fold_id = 0
    test_start_idx = min_train_days

    while test_start_idx < len(trading_days):
        test_end_idx = min(test_start_idx + step, len(trading_days))
        test_days = trading_days[test_start_idx:test_end_idx]

        if window_type == "expanding":
            train_days = trading_days[:test_start_idx]
        else:  # rolling
            train_start_idx = max(0, test_start_idx - rolling_train_days)
            train_days = trading_days[train_start_idx:test_start_idx]

        if len(train_days) >= min_train_days and len(test_days) > 0:
            folds.append(
                Fold(
                    fold_id=fold_id,
                    cadence=cadence,
                    window_type=window_type,
                    train_days=train_days,
                    test_days=test_days,
                )
            )
            fold_id += 1

        test_start_idx = test_end_idx

    return folds


def summarize_folds(folds: list[Fold]) -> None:
    print(f"{len(folds)} fold(s) generated "
          f"(cadence={folds[0].cadence if folds else 'n/a'}, "
          f"window_type={folds[0].window_type if folds else 'n/a'})")
    for f in folds:
        d = f.to_dict()
        print(f"  fold {d['fold_id']:>2}: train [{d['train_start']} -> {d['train_end']}] "
              f"({d['n_train_days']}d)  test [{d['test_start']} -> {d['test_end']}] "
              f"({d['n_test_days']}d)")


if __name__ == "__main__":
    import pandas as pd

    df = pd.read_parquet("/home/claude/spy-thesis-project/data/spy_5min_sample.parquet")
    days = sorted(df["date"].unique())
    print(f"Sample has {len(days)} trading days: {days[0]} -> {days[-1]}\n")

    print("=== WEEKLY, expanding, min_train_days=20 ===")
    weekly_folds = generate_walk_forward_folds(days, cadence="weekly", window_type="expanding", min_train_days=20)
    summarize_folds(weekly_folds)

    print("\n=== MONTHLY, expanding, min_train_days=20 ===")
    monthly_folds = generate_walk_forward_folds(days, cadence="monthly", window_type="expanding", min_train_days=20)
    summarize_folds(monthly_folds)
