# SPY 5-Minute Intraday Signal Generator with XAI Toggle

DSA3900 thesis project. Full pipeline: labeling → XGBoost + LSTM training
under walk-forward validation → SHAP explainability → precomputed
experiment scenarios → Streamlit within-subject decision-task app with
Google Sheets logging.

## Status

Everything below has been **built and verified end-to-end against a
42-day sample** (`data/spy_5min_sample.parquet`, Oct 28 – Dec 29, 2022).
It has **not yet been run against the full dataset**
(`data/spy_5min_cleaned_features.parquet`, Aug 2022 – Jul 2026,
~78,900 rows) — drop that file into `data/` and re-run the build order
below for the real thesis results. Everything is parameterized via CLI
flags (`--data`), so no code changes should be needed.

| Component | Status |
|---|---|
| `models/labeling.py` | Verified: day-boundary drop logic exact-matches expected count (504 = 42 days × 12 bars) |
| `models/features.py` | Verified: 27 XGBoost features, 10 LSTM features, matches spec exactly |
| `models/walk_forward.py` | Verified: weekly (5d) / monthly (21d) expanding-window folds correct |
| `models/train_xgboost.py` | Ran clean on sample; picks better cadence empirically (monthly won on this sample — small-sample result, not a claim about the full dataset) |
| `models/train_lstm.py` | Ran clean on sample; results comparable to XGBoost, consistent with the "honest baseline" framing in the diagnostics |
| `experiment/generate_scenarios.py` | Ran clean; produces 20 scenarios (10/10 split) using **fold-specific out-of-fold models** for both predictions and SHAP (never the final full-history model — avoids leaking future training data into a "historical" scenario) |
| `experiment/app.py` + supporting modules | Boots without exceptions (tested with Streamlit's `AppTest` framework + a mocked logging layer); consent → duplicate-ID guard → counterbalanced group assignment → trial loop → thank-you all verified |
| Google Sheets logging (`logging_utils.py`) | Code complete, follows spec (dev/deployed credential branching, immediate per-trial writes, two tabs). **Not tested against a real spreadsheet** — you'll need to supply a service-account JSON key |

## Setup

```bash
pip install -r requirements.txt
```

## Build order

1. Drop the full dataset at `data/spy_5min_cleaned_features.parquet`
2. `cd models && python train_xgboost.py` — trains XGBoost, runs walk-forward
   comparison (weekly vs monthly), saves `xgboost_model.json`,
   `walk_forward_results.csv`, `feature_list.json`
3. `python train_lstm.py` — same, for the LSTM baseline; appends to
   `walk_forward_results.csv` and merges into `feature_list.json`
4. Review `walk_forward_results.csv` — compare XGBoost vs LSTM honestly,
   note which cadence generalizes better
5. `cd ../experiment && python generate_scenarios.py` — produces
   `scenarios.json` (run once, offline)
6. Set up Google Sheets logging:
   - Create a Google Cloud service account, enable Sheets + Drive APIs
   - Download the JSON key to `experiment/credentials/gsheets_creds.json`
     (already gitignored)
   - Share a target spreadsheet (or let the app auto-create one named
     `spy_xai_experiment_log`) with the service account's email
   - For local dev: leave `SPY_XAI_APP_ENV` unset (defaults to `dev`,
     reads the local JSON file)
   - For deployment (e.g. Streamlit Community Cloud): set
     `SPY_XAI_APP_ENV=deployed` and configure `st.secrets["gcp_service_account"]`
     with the service account's fields
7. Replace the placeholder URLs in `experiment/config.py`
   (`CONSENT_FORM_URL`, `EXIT_SURVEY_URL`) with your real Google Forms
8. `streamlit run app.py` — test with a few dummy participant IDs before
   sending the real link to participants
9. Polish interface details only after confirming the full flow works
   end to end (per spec's build order)

## Known things to double check on the full dataset

- **Class balance**: the sample produced ~48% flat / 26% up / 25% down
  with `flat_std_mult=0.5`. Re-check this on the full dataset — if the
  full dataset's return distribution differs meaningfully across the
  4-year span (different vol regimes per the HMM diagnostics already
  run), the flat threshold may need tuning.
- **Walk-forward fold count**: the sample only produced 5 weekly / 2
  monthly folds. The full ~4-year dataset will produce far more folds per
  cadence — the cadence comparison in `walk_forward_results.csv` will be
  much more statistically meaningful there than on this sample.
- **Scenario diversity**: with only 20 candidate-generating folds in the
  full run (monthly cadence, ~4 years), there should be plenty of
  candidates to pick a genuinely diverse 20-scenario set; the sample only
  had 2 folds to draw from.
