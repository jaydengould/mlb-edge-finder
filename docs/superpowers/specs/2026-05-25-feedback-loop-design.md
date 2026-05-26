# Current Season Feedback Loop — Design Spec

## Goal

Make the model adaptive during the season by committing `historical_2026.csv` daily and retraining the XGBoost model whenever ≥15 new completed games have accumulated since the last train date.

## Architecture

Three changes working together:

1. **New `feedback.py` module** — owns refresh-and-retrain logic, called from the workflow after the pipeline runs each morning.
2. **`.gitignore`** — unignore `data/raw/historical_*.csv` so historical CSVs are committed to the repo. Gives CI a committed fallback and ensures tomorrow's pipeline has current-season data without a live API call.
3. **GitHub Actions `daily.yml`** — one new step (run feedback loop) and an expanded commit step that stages historical and model files alongside the edges CSV.

`pipeline.py` is untouched. The feedback loop runs after today's edges are already produced — it has no effect on the current day's output.

---

## `config.py` changes

Add one constant:

```python
RETRAIN_THRESHOLD: int = 15  # retrain after this many new games since last model
```

---

## `feedback.py` module

### `refresh_historical(season: int) -> pd.DataFrame`

Calls `fetch_historical(season, force=True)`. Always fetches the latest completed games for the season from the MLB Stats API and overwrites the local cache. Returns the full season DataFrame.

### `games_since_last_train(historical_df: pd.DataFrame, last_train_date: date) -> int`

Counts rows in `historical_df` where `game_date > last_train_date`. Used to decide whether retraining is warranted.

### `run_feedback_loop(season: int) -> dict`

Orchestrates the full feedback cycle:

1. Call `refresh_historical(season)` — get latest completed games, update cache
2. Glob `MODELS_DIR` for `xgb_*.pkl` sorted by filename to find the latest model date
3. If no model exists, set `last_train_date` to `date.min` (triggers immediate retrain)
4. Call `games_since_last_train(historical_df, last_train_date)`
5. If count >= `config.RETRAIN_THRESHOLD`:
   - Call `build_training_set([2019, 2021, 2022, 2023, 2024, 2025, 2026], force=False)` — cache-first for past seasons, live fetch for 2026 stats
   - Call `model.train(training_df)` → returns `(clf, X_val, X_test, y_val, y_test)`
   - Call `model.calibrate(clf, X_val, y_val)` → calibrated classifier
   - Call `model.evaluate(clf, X_test, y_test)` → metrics dict
   - Call `model.save_model(clf, metrics, date.today())` → writes `xgb_YYYY-MM-DD.pkl` + `metrics_YYYY-MM-DD.json`
6. Return `{"season": season, "games_in_season": int, "new_games": int, "retrained": bool, "model_date": date | None}`

**Note on 2026 stats:** `build_training_set` calls `fetch_stats(date(season, 9, 28))` per season. For 2026, September 28 hasn't occurred, so it fetches and caches current cumulative stats under `stats_2026-09-28.csv`. This is a valid proxy — pybaseball returns full-season cumulative stats regardless of the exact date queried.

---

## `.gitignore` changes

Add one exception line:

```
!data/raw/historical_*.csv
```

This unignores all historical season CSVs. Past seasons (2019, 2021–2025) are static and small; the current season (2026) grows by ~1 row per game.

---

## GitHub Actions `daily.yml` changes

### New step: Run feedback loop

Inserted after "Run pipeline", before "Promote edges file":

```yaml
- name: Run feedback loop
  continue-on-error: true
  env:
    ODDS_API_KEY: ${{ secrets.ODDS_API_KEY }}
  run: |
    python -c "
    from mlb_edge_finder.feedback import run_feedback_loop
    import json
    result = run_feedback_loop(2026)
    print(json.dumps(result, default=str))
    "
```

`continue-on-error: true` ensures a retrain failure (e.g. stats API down) never blocks the edges commit. Today's edges always get through.

### Updated commit step

Replaces the existing "Commit and push edges file" step:

```yaml
- name: Commit and push artifacts
  run: |
    DATE=$(date -u +%Y-%m-%d)
    git config user.name "github-actions[bot]"
    git config user.email "github-actions[bot]@users.noreply.github.com"
    git add "outputs/edges_${DATE}.csv"
    git add "data/raw/historical_2026.csv"
    git add "models/"
    if git diff --staged --quiet; then
      echo "Nothing to commit."
    else
      git commit -m "chore: daily update ${DATE}"
      git push origin HEAD:${{ github.ref_name }}
    fi
```

---

## Testing

- `test_refresh_historical` — patches `fetch_historical`, verifies it is called with `force=True`
- `test_games_since_last_train` — verifies correct count for a DataFrame with games before and after `last_train_date`
- `test_games_since_last_train_zero` — all games before cutoff returns 0
- `test_run_feedback_loop_no_retrain` — fewer than 15 new games, verifies `retrained=False` and model not saved
- `test_run_feedback_loop_retrain` — 15+ new games, verifies `retrained=True` and `save_model` called
- `test_run_feedback_loop_no_model` — no existing model, verifies retrain always triggers

---

## Data Flow

```
GitHub Actions (9:30 AM ET)
  │
  ├─ pipeline.run() ──► uses committed historical_2026.csv (no live API call)
  │                      produces edges_YYYY-MM-DD.csv
  │
  ├─ run_feedback_loop(2026)
  │    ├─ fetch_historical(2026, force=True) ──► refreshed historical_2026.csv
  │    ├─ count new games since last model date
  │    └─ if ≥15: retrain ──► xgb_YYYY-MM-DD.pkl + metrics_YYYY-MM-DD.json
  │
  └─ git commit: edges + historical_2026.csv + models/ (if retrained)
```

---

## Files Changed

| File | Change |
|---|---|
| `src/mlb_edge_finder/feedback.py` | Create |
| `src/mlb_edge_finder/config.py` | Add `RETRAIN_THRESHOLD = 15` |
| `tests/test_feedback.py` | Create |
| `.gitignore` | Add `!data/raw/historical_*.csv` |
| `.github/workflows/daily.yml` | Add feedback loop step, update commit step |
