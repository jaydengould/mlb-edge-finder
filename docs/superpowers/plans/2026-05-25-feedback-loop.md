# Current Season Feedback Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Commit `historical_2026.csv` daily and retrain the XGBoost model whenever ≥15 new completed games have accumulated since the last train date.

**Architecture:** New `feedback.py` module with three functions (`refresh_historical`, `games_since_last_train`, `run_feedback_loop`) called from a new GitHub Actions workflow step. `.gitignore` updated so historical CSVs are committable. The pipeline itself is untouched — the feedback loop runs after today's edges are already produced.

**Tech Stack:** Python stdlib `datetime`, pandas, statsapi, existing `model`, `training_data`, `historical_ingestion` modules, GitHub Actions.

---

## File Map

| File | Change |
|---|---|
| `src/mlb_edge_finder/config.py` | Add `RETRAIN_THRESHOLD = 15` |
| `src/mlb_edge_finder/feedback.py` | Create — owns refresh + retrain logic |
| `tests/test_feedback.py` | Create — TDD tests for all three functions |
| `.gitignore` | Change `data/raw/` → `data/raw/*` + add `!data/raw/historical_*.csv` |
| `.github/workflows/daily.yml` | Add feedback loop step, update commit step |
| `CLAUDE.md` | Document feedback loop |
| `README.md` | Update test count, add roadmap entry |

---

### Task 1: Add RETRAIN_THRESHOLD to config.py

**Files:**
- Modify: `src/mlb_edge_finder/config.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_config.py` if it exists, otherwise add a standalone assertion in a new file `tests/test_feedback.py` (we'll expand this file in Task 2):

```python
# tests/test_feedback.py
"""Tests for feedback module."""
from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


def test_retrain_threshold_constant():
    from mlb_edge_finder import config
    assert config.RETRAIN_THRESHOLD == 15
```

- [ ] **Step 2: Run test to confirm it fails**

```bash
pytest tests/test_feedback.py::test_retrain_threshold_constant -v
```

Expected: FAIL — `AttributeError: module 'mlb_edge_finder.config' has no attribute 'RETRAIN_THRESHOLD'`

- [ ] **Step 3: Add the constant to config.py**

In `src/mlb_edge_finder/config.py`, add after `MIN_PROB_EDGE`:

```python
RETRAIN_THRESHOLD: int = 15  # retrain after this many new games since last model date
```

- [ ] **Step 4: Run test to confirm it passes**

```bash
pytest tests/test_feedback.py::test_retrain_threshold_constant -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/mlb_edge_finder/config.py tests/test_feedback.py
git commit -m "feat: add RETRAIN_THRESHOLD constant to config"
```

---

### Task 2: Create feedback.py — refresh_historical and games_since_last_train

**Files:**
- Create: `src/mlb_edge_finder/feedback.py`
- Modify: `tests/test_feedback.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_feedback.py`:

```python
def _make_historical_df(game_dates: list[str]) -> pd.DataFrame:
    return pd.DataFrame({
        "game_date": game_dates,
        "home_name": "Yankees",
        "away_name": "Red Sox",
        "home_score": 5,
        "away_score": 3,
        "home_win": 1,
        "home_starter_name": None,
        "away_starter_name": None,
    })


def test_refresh_historical_calls_fetch_with_force():
    from mlb_edge_finder import feedback
    mock_df = _make_historical_df(["2026-04-01"])
    with patch("mlb_edge_finder.feedback.fetch_historical", return_value=mock_df) as mock_fetch:
        result = feedback.refresh_historical(2026)
    mock_fetch.assert_called_once_with(2026, force=True)
    assert len(result) == 1


def test_games_since_last_train_counts_correctly():
    from mlb_edge_finder import feedback
    cutoff = date(2026, 4, 10)
    # 3 games after cutoff, 2 before
    df = _make_historical_df([
        "2026-04-08", "2026-04-09",      # before
        "2026-04-11", "2026-04-12", "2026-04-13",  # after
    ])
    assert feedback.games_since_last_train(df, cutoff) == 3


def test_games_since_last_train_zero_when_all_before():
    from mlb_edge_finder import feedback
    cutoff = date(2026, 5, 1)
    df = _make_historical_df(["2026-04-01", "2026-04-02", "2026-04-03"])
    assert feedback.games_since_last_train(df, cutoff) == 0
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/test_feedback.py::test_refresh_historical_calls_fetch_with_force tests/test_feedback.py::test_games_since_last_train_counts_correctly tests/test_feedback.py::test_games_since_last_train_zero_when_all_before -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'mlb_edge_finder.feedback'`

- [ ] **Step 3: Create src/mlb_edge_finder/feedback.py**

```python
"""Refresh current-season historical data and conditionally retrain the model."""
import logging
from datetime import date

import pandas as pd

from mlb_edge_finder import config, model
from mlb_edge_finder.historical_ingestion import fetch_historical
from mlb_edge_finder.training_data import build_training_set

logger = logging.getLogger(__name__)

_TRAINING_SEASONS = [2019, 2021, 2022, 2023, 2024, 2025, 2026]


def refresh_historical(season: int) -> pd.DataFrame:
    """Force-fetch the latest completed games for the given season.

    Always bypasses the local cache to ensure today's completed games
    are included. Overwrites data/raw/historical_YYYY.csv.

    Args:
        season: The season year to refresh (e.g. 2026).

    Returns:
        DataFrame of all completed regular-season games for the season.
    """
    return fetch_historical(season, force=True)


def games_since_last_train(historical_df: pd.DataFrame, last_train_date: date) -> int:
    """Count completed games that occurred after last_train_date.

    Args:
        historical_df: Output of refresh_historical() or fetch_historical().
        last_train_date: The date of the most recently saved model.

    Returns:
        Number of rows in historical_df with game_date > last_train_date.
    """
    game_dates = pd.to_datetime(historical_df["game_date"]).dt.date
    return int((game_dates > last_train_date).sum())
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
pytest tests/test_feedback.py::test_refresh_historical_calls_fetch_with_force tests/test_feedback.py::test_games_since_last_train_counts_correctly tests/test_feedback.py::test_games_since_last_train_zero_when_all_before -v
```

Expected: all 3 PASS

- [ ] **Step 5: Run full suite to confirm no regressions**

```bash
pytest tests/ -q
```

Expected: 179 passed (175 prior + 1 Task 1 + 3 Task 2)

- [ ] **Step 6: Commit**

```bash
git add src/mlb_edge_finder/feedback.py tests/test_feedback.py
git commit -m "feat: add refresh_historical and games_since_last_train to feedback.py"
```

---

### Task 3: Add run_feedback_loop to feedback.py

**Files:**
- Modify: `src/mlb_edge_finder/feedback.py`
- Modify: `tests/test_feedback.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_feedback.py`:

```python
def test_run_feedback_loop_no_retrain(tmp_path):
    """Fewer than 15 new games — should skip retrain."""
    from mlb_edge_finder import feedback

    model_date = date(2026, 4, 1)
    (tmp_path / f"xgb_{model_date}.pkl").touch()

    # 14 games after model_date — one short of threshold
    game_dates = [str(model_date + timedelta(days=i + 1)) for i in range(14)]
    hist_df = _make_historical_df(game_dates)

    with patch("mlb_edge_finder.feedback.refresh_historical", return_value=hist_df), \
         patch("mlb_edge_finder.feedback.config.MODELS_DIR", tmp_path), \
         patch("mlb_edge_finder.feedback.config.RETRAIN_THRESHOLD", 15), \
         patch("mlb_edge_finder.feedback.model.save_model") as mock_save:
        result = feedback.run_feedback_loop(2026)

    assert result["retrained"] is False
    assert result["new_games"] == 14
    assert result["season"] == 2026
    mock_save.assert_not_called()


def test_run_feedback_loop_retrain(tmp_path):
    """Exactly 15 new games — should retrain and save model."""
    from mlb_edge_finder import feedback

    model_date = date(2026, 4, 1)
    (tmp_path / f"xgb_{model_date}.pkl").touch()

    # 15 games after model_date — hits threshold
    game_dates = [str(model_date + timedelta(days=i + 1)) for i in range(15)]
    hist_df = _make_historical_df(game_dates)

    mock_clf = MagicMock()
    mock_training_df = pd.DataFrame({"home_win": [1, 0]})

    with patch("mlb_edge_finder.feedback.refresh_historical", return_value=hist_df), \
         patch("mlb_edge_finder.feedback.config.MODELS_DIR", tmp_path), \
         patch("mlb_edge_finder.feedback.config.RETRAIN_THRESHOLD", 15), \
         patch("mlb_edge_finder.feedback.build_training_set", return_value=mock_training_df) as mock_build, \
         patch("mlb_edge_finder.feedback.model.train", return_value=(mock_clf, MagicMock(), MagicMock(), MagicMock(), MagicMock())), \
         patch("mlb_edge_finder.feedback.model.calibrate", return_value=mock_clf), \
         patch("mlb_edge_finder.feedback.model.evaluate", return_value={}), \
         patch("mlb_edge_finder.feedback.model.save_model") as mock_save:
        result = feedback.run_feedback_loop(2026)

    assert result["retrained"] is True
    assert result["new_games"] == 15
    mock_build.assert_called_once_with([2019, 2021, 2022, 2023, 2024, 2025, 2026], force=True)
    mock_save.assert_called_once()


def test_run_feedback_loop_no_existing_model(tmp_path):
    """No model on disk — should always retrain regardless of game count."""
    from mlb_edge_finder import feedback

    # Only 2 games — well below threshold, but no model exists
    hist_df = _make_historical_df(["2026-04-01", "2026-04-02"])

    mock_clf = MagicMock()

    with patch("mlb_edge_finder.feedback.refresh_historical", return_value=hist_df), \
         patch("mlb_edge_finder.feedback.config.MODELS_DIR", tmp_path), \
         patch("mlb_edge_finder.feedback.config.RETRAIN_THRESHOLD", 15), \
         patch("mlb_edge_finder.feedback.build_training_set", return_value=pd.DataFrame({"home_win": [1]})), \
         patch("mlb_edge_finder.feedback.model.train", return_value=(mock_clf, MagicMock(), MagicMock(), MagicMock(), MagicMock())), \
         patch("mlb_edge_finder.feedback.model.calibrate", return_value=mock_clf), \
         patch("mlb_edge_finder.feedback.model.evaluate", return_value={}), \
         patch("mlb_edge_finder.feedback.model.save_model") as mock_save:
        result = feedback.run_feedback_loop(2026)

    assert result["retrained"] is True
    mock_save.assert_called_once()
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/test_feedback.py::test_run_feedback_loop_no_retrain tests/test_feedback.py::test_run_feedback_loop_retrain tests/test_feedback.py::test_run_feedback_loop_no_existing_model -v
```

Expected: FAIL — `AttributeError: module 'mlb_edge_finder.feedback' has no attribute 'run_feedback_loop'`

- [ ] **Step 3: Add run_feedback_loop to feedback.py**

Append to `src/mlb_edge_finder/feedback.py`:

```python
def run_feedback_loop(season: int) -> dict:
    """Refresh historical data for the season and retrain if enough new games exist.

    Always refreshes historical_YYYY.csv from the MLB Stats API. Checks how many
    games have been played since the most recent saved model. If the count reaches
    config.RETRAIN_THRESHOLD (or no model exists), rebuilds the training set for
    all seasons and retrains + calibrates + saves a new model.

    Args:
        season: The current season year (e.g. 2026).

    Returns:
        Dict with keys: season (int), games_in_season (int), new_games (int),
        retrained (bool).
    """
    historical_df = refresh_historical(season)

    pkls = sorted(config.MODELS_DIR.glob("xgb_*.pkl"))
    if pkls:
        last_train_date = date.fromisoformat(pkls[-1].stem[4:])  # strip "xgb_"
        new_games = games_since_last_train(historical_df, last_train_date)
        do_retrain = new_games >= config.RETRAIN_THRESHOLD
    else:
        last_train_date = None
        new_games = len(historical_df)
        do_retrain = True

    retrained = False
    if do_retrain:
        logger.info(
            "Retraining model: %d new games since %s (threshold: %d)",
            new_games, last_train_date, config.RETRAIN_THRESHOLD,
        )
        training_df = build_training_set(_TRAINING_SEASONS, force=True)
        clf, X_val, X_test, y_val, y_test = model.train(training_df)
        clf = model.calibrate(clf, X_val, y_val)
        metrics = model.evaluate(clf, X_test, y_test)
        model.save_model(clf, metrics, date.today())
        retrained = True
    else:
        logger.info(
            "Skipping retrain: %d new games since %s (threshold: %d)",
            new_games, last_train_date, config.RETRAIN_THRESHOLD,
        )

    return {
        "season": season,
        "games_in_season": len(historical_df),
        "new_games": new_games,
        "retrained": retrained,
    }
```

- [ ] **Step 4: Run new tests to confirm they pass**

```bash
pytest tests/test_feedback.py::test_run_feedback_loop_no_retrain tests/test_feedback.py::test_run_feedback_loop_retrain tests/test_feedback.py::test_run_feedback_loop_no_existing_model -v
```

Expected: all 3 PASS

- [ ] **Step 5: Run full suite to confirm no regressions**

```bash
pytest tests/ -q
```

Expected: 182 passed (175 prior + 1 Task 1 + 3 Task 2 + 3 Task 3)

- [ ] **Step 6: Commit**

```bash
git add src/mlb_edge_finder/feedback.py tests/test_feedback.py
git commit -m "feat: add run_feedback_loop to feedback.py"
```

---

### Task 4: Update .gitignore and commit existing historical CSVs

**Files:**
- Modify: `.gitignore`

**Background:** `data/raw/` in `.gitignore` uses a trailing slash which tells git to ignore the directory and ALL its contents — negation patterns can't override this. The fix is to change `data/raw/` to `data/raw/*` (matches contents, not the directory itself), which allows a negation `!data/raw/historical_*.csv` to work.

- [ ] **Step 1: Update .gitignore**

In `.gitignore`, find the line:

```
data/raw/
```

Replace it with:

```
data/raw/*
!data/raw/historical_*.csv
```

- [ ] **Step 2: Stage any existing historical CSVs**

```bash
git add -f data/raw/historical_*.csv 2>/dev/null; echo "done"
```

If no historical CSVs exist locally yet (they live in the gitignored directory), this is a no-op. The workflow will create and commit them on first run.

- [ ] **Step 3: Commit**

```bash
git add .gitignore
git commit -m "chore: unignore historical CSVs so they can be committed by CI"
```

(If historical CSVs were staged in Step 2, they'll be included automatically.)

---

### Task 5: Update GitHub Actions workflow

**Files:**
- Modify: `.github/workflows/daily.yml`

- [ ] **Step 1: Add the feedback loop step**

In `.github/workflows/daily.yml`, insert this new step after "Run pipeline" and before "Promote edges file to outputs/":

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

- [ ] **Step 2: Replace the commit step**

Replace the existing "Commit and push edges file" step with:

```yaml
      - name: Commit and push artifacts
        run: |
          DATE=$(date -u +%Y-%m-%d)
          git config user.name "github-actions[bot]"
          git config user.email "github-actions[bot]@users.noreply.github.com"
          git add "outputs/edges_${DATE}.csv"
          git add "data/raw/historical_2026.csv" 2>/dev/null || true
          git add "models/" 2>/dev/null || true
          if git diff --staged --quiet; then
            echo "Nothing to commit."
          else
            git commit -m "chore: daily update ${DATE}"
            git push origin HEAD:${{ github.ref_name }}
          fi
```

- [ ] **Step 3: Verify the workflow file is valid YAML**

```bash
python -c "import yaml; yaml.safe_load(open('.github/workflows/daily.yml'))" && echo "valid"
```

Expected: `valid`

- [ ] **Step 4: Commit**

```bash
git add .github/workflows/daily.yml
git commit -m "feat: add feedback loop step and update commit step in daily workflow"
```

---

### Task 6: Update CLAUDE.md and README.md

**Files:**
- Modify: `CLAUDE.md`
- Modify: `README.md`

- [ ] **Step 1: Update CLAUDE.md current phase header**

Find:

```
**Historical ingestion resilience complete.**
```

Replace with:

```
**Current season feedback loop complete.**
```

- [ ] **Step 2: Add feedback loop bullet to CLAUDE.md current phase section**

After the historical ingestion resilience bullet, add:

```
- **Current season feedback loop complete:** `feedback.py` — `refresh_historical(season)` force-fetches `historical_YYYY.csv`, `games_since_last_train(historical_df, last_train_date)` counts new games, `run_feedback_loop(season)` orchestrates: refresh → check count → retrain if `new_games >= RETRAIN_THRESHOLD`. Retrains with `_TRAINING_SEASONS = [2019, 2021, 2022, 2023, 2024, 2025, 2026]`. `config.RETRAIN_THRESHOLD = 15`. `.gitignore` updated: `data/raw/*` + `!data/raw/historical_*.csv` so historical CSVs are committed. Daily workflow gains a `Run feedback loop` step (`continue-on-error: true`) and commits `historical_2026.csv` + new model files alongside edges. 7 new tests (182 total passing).
```

- [ ] **Step 3: Add roadmap entry to CLAUDE.md**

After the historical ingestion resilience roadmap entry, add:

```
- [x] Current season feedback loop — `feedback.py` with `refresh_historical()`, `games_since_last_train()`, `run_feedback_loop()`. Retrains every 15 new games. Daily workflow commits `historical_2026.csv` and new model files. `RETRAIN_THRESHOLD=15` in config.
```

- [ ] **Step 4: Update README.md test count**

Find:

```
175 smoke + integration tests. All pass.
```

Replace with:

```
182 smoke + integration tests. All pass.
```

- [ ] **Step 5: Add roadmap entry to README.md**

After the historical ingestion resilience roadmap entry, add:

```
- [x] Current season feedback loop — `feedback.py` refreshes `historical_2026.csv` daily and retrains the model every 15 new games; workflow commits historical data and new model files alongside edges
```

- [ ] **Step 6: Commit**

```bash
git add CLAUDE.md README.md
git commit -m "docs: update CLAUDE.md and README for feedback loop"
```
