# Phase 8: Expand Training Seasons Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expand `_HISTORICAL_SEASONS` from `[2023, 2024, 2025]` to `[2019, 2021, 2022, 2023, 2024, 2025]`, skipping 2020 (60-game anomaly), so `fetch_all_historical()` and `build_training_set()` cover ~5× more training data by default.

**Architecture:** One constant change in `historical_ingestion.py` drives both `fetch_all_historical()` and the notebook training invocation. The pipeline is already season-agnostic — all name/abbreviation mappings for pre-2023 seasons are already in place.

**Tech Stack:** Python 3.10+, pytest, statsapi, XGBoost, Jupyter

---

### Task 1: Update test to expect 6 seasons, verify it fails, then fix the constant

**Files:**
- Modify: `tests/test_historical_ingestion.py:124-133`
- Modify: `src/mlb_edge_finder/historical_ingestion.py:18`

- [ ] **Step 1: Update the test assertion**

In `tests/test_historical_ingestion.py`, change `test_fetch_all_historical_concatenates`:

```python
def test_fetch_all_historical_concatenates(tmp_path):
    from mlb_edge_finder import historical_ingestion
    one_game = [_make_game("Yankees", "Red Sox", 5, 3)]
    with patch("mlb_edge_finder.historical_ingestion.statsapi.schedule", return_value=one_game), \
         patch("mlb_edge_finder.historical_ingestion.config.DATA_RAW_DIR", tmp_path):
        df = historical_ingestion.fetch_all_historical(force=True)
    # 6 seasons × 1 game each
    assert len(df) == 6
    assert "home_starter_name" in df.columns
    assert "away_starter_name" in df.columns
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_historical_ingestion.py::test_fetch_all_historical_concatenates -v
```

Expected: **FAIL** — `assert 3 == 6`

- [ ] **Step 3: Update `_HISTORICAL_SEASONS` in `historical_ingestion.py`**

Change line 18:

```python
_HISTORICAL_SEASONS = [2019, 2021, 2022, 2023, 2024, 2025]
```

- [ ] **Step 4: Run the full test suite**

```bash
pytest tests/ -v
```

Expected: **116 passed** (no failures). The `test_fetch_all_historical_concatenates` test now passes; all other tests are unaffected since the mock-based suite is season-agnostic.

- [ ] **Step 5: Commit**

```bash
git add src/mlb_edge_finder/historical_ingestion.py tests/test_historical_ingestion.py
git commit -m "feat: expand training seasons to 2019, 2021-2025 (skip 2020)"
```

---

### Task 2: Update notebook to use expanded seasons list

**Files:**
- Modify: `notebooks/01_exploration.ipynb` (cell 12)

- [ ] **Step 1: Update the seasons variable in cell 12**

Open `notebooks/01_exploration.ipynb`. Cell 12 currently reads:

```python
from mlb_edge_finder import training_data

seasons = [2023, 2024, 2025]
# force=True required after Phase 7 — rebuilds cache to include home_sp_*/away_sp_* pitcher columns
training_df = training_data.build_training_set(seasons, force=True)
print(f"{len(training_df)} rows, {len(training_df.columns)} columns")
training_df.head()
```

Change it to:

```python
from mlb_edge_finder import training_data

seasons = [2019, 2021, 2022, 2023, 2024, 2025]
# force=True rebuilds cache to include all 6 seasons (2019, 2021-2025; 2020 skipped — 60-game anomaly)
training_df = training_data.build_training_set(seasons, force=True)
print(f"{len(training_df)} rows, {len(training_df.columns)} columns")
training_df.head()
```

- [ ] **Step 2: Clear all notebook outputs**

In Jupyter: Kernel → Restart & Clear Output (or via CLI):

```bash
jupyter nbconvert --clear-output --inplace notebooks/01_exploration.ipynb
```

- [ ] **Step 3: Commit**

```bash
git add notebooks/01_exploration.ipynb
git commit -m "chore: update notebook seasons list to 2019, 2021-2025"
```

---

### Task 3: Fetch historical data for new seasons and retrain

> This task runs live API calls and real model training. It is not part of the automated test suite — run it interactively in the notebook or a Python shell.

- [ ] **Step 1: Fetch historical data for the new seasons**

Run in the notebook (Section 4a) or a Python shell:

```python
from mlb_edge_finder import historical_ingestion, config
config.setup_logging()

# Fetch the three new seasons — already-cached 2023/2024/2025 are skipped automatically
for season in [2019, 2021, 2022]:
    df = historical_ingestion.fetch_historical(season, force=False)
    print(f"{season}: {len(df)} games")
```

Expected output (approximately):
```
2019: 2430 games
2021: 2430 games
2022: 2430 games
```

- [ ] **Step 2: Rebuild the training set**

```python
from mlb_edge_finder import training_data

seasons = [2019, 2021, 2022, 2023, 2024, 2025]
training_df = training_data.build_training_set(seasons, force=True)
print(f"{len(training_df)} rows, {len(training_df.columns)} columns")
```

Expected: ~4860+ rows (6 full seasons of completed games after inner-join on stats).

- [ ] **Step 3: Retrain the model**

```python
from datetime import date
from mlb_edge_finder import model

clf, X_test, y_test = model.train(training_df)
metrics = model.evaluate(clf, X_test, y_test)
print(metrics)
model.save_model(clf, metrics, date.today())
```

Expected: new `models/xgb_2026-05-12.pkl` and `models/metrics_2026-05-12.json` written.

- [ ] **Step 4: Commit model artifacts**

```bash
git add models/xgb_2026-05-12.pkl models/metrics_2026-05-12.json
git commit -m "feat: retrain model on expanded 2019-2025 training set"
```
