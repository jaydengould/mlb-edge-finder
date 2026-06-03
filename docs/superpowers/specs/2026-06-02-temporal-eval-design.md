# Temporal Out-of-Time Evaluation — Design Spec

**Date:** 2026-06-02
**Status:** Approved

## Problem

The current backtest and model metrics on the dashboard use a random stratified 80/20 split across all seasons mixed together. A model trained on games from 2024 can "see" patterns from 2019 games in the test set — and vice versa. This is a methodology concern that technical interviewers will raise. It also means the dashboard P&L and win-rate numbers are not a credible forward-looking signal.

## Goal

Replace all dashboard evaluation numbers (accuracy, ROC-AUC, win rate, ROI, Sharpe, P&L chart) with results from a true temporal holdout: train on 2019–2024, test blind on 2025. One source of truth, one methodology, clearly labeled on the dashboard.

---

## Architecture

### New module: `src/mlb_edge_finder/temporal_eval.py`

Single public function:

```python
def run(holdout_season: int = 2025, force: bool = False) -> dict
```

**Steps:**

1. Load the full training set via `training_data.load_training_set()`. Raise `RuntimeError` if not found (user must run `build_training_set` first).
2. Split by `season` column:
   - `train_df`: rows where `season < holdout_season`
   - `test_df`: rows where `season == holdout_season`
   - Raise `RuntimeError` if either slice is empty.
3. Extract features: drop `NON_FEATURE_COLS` from both slices. Align `test_df` columns to `train_df` columns (reindex, fill missing with NaN).
4. Train a fresh `XGBClassifier` on `train_df` using `config.XGB_N_ESTIMATORS` and `config.XGB_MAX_DEPTH`. Do a further 75/25 split of `train_df` into fit/val for calibration (mirrors the 60/20 ratio from `model._three_way_split`).
5. Calibrate via `model.calibrate(clf, X_val, y_val)`.
6. Evaluate on `test_df` via `model.evaluate(calibrated_clf, X_test, y_test)` → accuracy, ROC-AUC, log_loss, brier_score.
7. Run the bet simulation directly on `test_df` — **do not call `run_backtest()`**, which does its own internal split. Instead, `backtest.py` gains a new public function `simulate_bets(clf, X_test, y_test, meta_df, ev_threshold)` that contains the current per-bet loop logic extracted from `run_backtest`. Both `run_backtest` and `temporal_eval.run` call this shared helper. `meta_df` is `test_df[["game_date","home_name","away_name"]]`.
8. Compute summary via `backtest.compute_summary(backtest_df)` → win rate, ROI, Sharpe, n_bets.
9. Build P&L series from `backtest_df["cumulative_pnl"]` (same format as current `backtest_pnl.json`).
10. Write output JSON to `config.MODELS_DIR / f"temporal_eval_{holdout_season}.json"`. Skip if file exists and `force=False`.
11. Return the written dict.

**The diagnostic model is never saved as a `.pkl`.** The production model (trained on all seasons) is untouched.

**Output JSON schema:**

```json
{
  "holdout_season": 2025,
  "train_seasons": [2019, 2021, 2022, 2023, 2024],
  "n_train": 12669,
  "n_test": 2430,
  "accuracy": 0.572,
  "roc_auc": 0.601,
  "log_loss": 0.681,
  "brier_score": 0.243,
  "n_bets": 1800,
  "win_rate": 0.603,
  "roi_pct": 15.1,
  "sharpe_ratio": 0.42,
  "total_pnl": 1800.0,
  "avg_ev": 0.28,
  "max_drawdown": -420.0,
  "pnl_series": [
    {"date": "2025-03-20", "cumulative_pnl": 95.45},
    ...
  ]
}
```

### CLI entry point

```bash
python -m mlb_edge_finder.temporal_eval [--holdout-season 2025] [--force]
```

Implemented via `if __name__ == "__main__"` block in the module. Prints a summary table on completion.

---

## Dashboard Changes (`generate_site.py`)

### Reading data

Replace separate `_load_metrics(metrics_path)` and `_load_pnl(pnl_path)` calls with a single:

```python
def _load_temporal_eval(models_dir: Path) -> dict | None
```

Globs `models_dir` for `temporal_eval_*.json`, loads the most recent (by filename). Returns `None` if not found — dashboard degrades gracefully (stats cards hidden, charts empty).

### Stats cards

Current model stats card (accuracy, ROC-AUC from `metrics_*.json`) is replaced by temporal eval stats. Add a subtitle line under the card header:

> "Trained on 2019–2024 · Tested on 2025 holdout"

Show: ROC-AUC, Accuracy, Win Rate, ROI, Sharpe.

### P&L chart

Replace `data/backtest_pnl.json` as the chart data source with `pnl_series` from the temporal eval JSON. Format is identical — list of `{date, cumulative_pnl}` objects — so the Chart.js rendering code is unchanged except for the data source.

### Graceful degradation

If `temporal_eval_*.json` does not exist, the stats cards and P&L chart are hidden (same behavior as today when `metrics_*.json` or `pnl.json` is missing). No crashes.

---

## Running Temporal Eval

**One-off (local or workflow_dispatch):**

```bash
python -m mlb_edge_finder.temporal_eval
```

Writes `models/temporal_eval_2025.json`. Commit the file — it is a static artifact like the model `.pkl` and `.json` metrics files.

**Daily workflow:** Does NOT rerun temporal eval. The committed JSON is read as a static file. Temporal eval only reruns when explicitly triggered (e.g., after a model retrain or at end of season).

**Regenerate dashboard after running:**

```bash
python -m mlb_edge_finder.generate_site
```

---

## Files Changed

| File | Change |
|---|---|
| `src/mlb_edge_finder/temporal_eval.py` | New module |
| `src/mlb_edge_finder/generate_site.py` | Replace `_load_metrics` + `_load_pnl` with `_load_temporal_eval`; update stats card and P&L chart data source |
| `models/temporal_eval_2025.json` | New committed artifact (generated by running the module) |
| `tests/test_temporal_eval.py` | New test file |
| `src/mlb_edge_finder/backtest.py` | Extract `simulate_bets()` helper from `run_backtest()` |
| `tests/test_generate_site.py` | Update existing tests for new data loading path |

`data/backtest_pnl.json` is no longer read by `generate_site.py`. It remains on disk (not deleted) but becomes unused by the dashboard.

---

## Tests (`tests/test_temporal_eval.py`)

1. `test_temporal_split_no_leakage` — train slice contains no rows from holdout season
2. `test_temporal_split_test_only_holdout` — test slice contains only rows from holdout season
3. `test_run_writes_json` — `run()` writes JSON to `MODELS_DIR`
4. `test_run_json_has_required_keys` — output has all expected top-level keys
5. `test_run_pnl_series_is_list` — `pnl_series` is a non-empty list of `{date, cumulative_pnl}` dicts
6. `test_run_skips_if_exists` — `run(force=False)` skips when file already exists
7. `test_run_force_overwrites` — `run(force=True)` overwrites existing file
8. `test_load_temporal_eval_returns_none_when_missing` — `_load_temporal_eval` returns `None` gracefully

Existing `tests/test_generate_site.py` tests updated to mock `_load_temporal_eval` instead of `_load_metrics` / `_load_pnl`.
