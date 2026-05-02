# Phase 5: Edge Finder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `edge_finder.find_edges()` and `pipeline.run()` to complete the end-to-end MLB edge-finding pipeline.

**Architecture:** `find_edges()` uses `clf.feature_names_in_` to select inference columns, runs two sequential passes (home, away) to find positive-EV bets, and writes `edges_YYYY-MM-DD.csv`. `pipeline.run()` orchestrates all five stages and auto-discovers the latest saved model by globbing `MODELS_DIR` for `xgb_*.pkl` sorted by filename.

**Tech Stack:** pandas, XGBoost (`XGBClassifier`), pytest, unittest.mock

---

## File Map

| File | Action | What changes |
|---|---|---|
| `src/mlb_edge_finder/edge_finder.py` | Modify | Implement `find_edges()` |
| `src/mlb_edge_finder/pipeline.py` | Modify | Implement `run()` |
| `tests/test_edge_finder.py` | Modify | Add 5 behavioural tests for `find_edges()` |
| `tests/test_pipeline.py` | Create | 3 tests for `pipeline.run()` |
| `CLAUDE.md` | Modify | Mark Phase 5 complete, update roadmap |

---

## Task 1: Tests for `find_edges()`

**Files:**
- Modify: `tests/test_edge_finder.py`

- [ ] **Step 1: Add the 5 new tests**

Append the following to `tests/test_edge_finder.py` (keep the existing 4 tests):

```python
import numpy as np
from datetime import date
from unittest.mock import MagicMock, patch


# ---------- helpers ----------

FEATURE_COLS = ["home_bat_avg", "away_bat_avg", "home_era", "away_era"]


def _make_clf(home_proba: float) -> MagicMock:
    """Return a mock XGBClassifier that always predicts home_proba."""
    clf = MagicMock()
    clf.feature_names_in_ = np.array(FEATURE_COLS)
    # predict_proba returns shape (n_samples, 2): [away_prob, home_prob]
    clf.predict_proba.side_effect = lambda X: np.column_stack(
        [np.full(len(X), 1.0 - home_proba), np.full(len(X), home_proba)]
    )
    return clf


def _make_features(home_odds: int, away_odds: int) -> "pd.DataFrame":
    import pandas as pd
    return pd.DataFrame([{
        "game_id": "game_1",
        "home_team": "New York Yankees",
        "away_team": "Boston Red Sox",
        "home_odds_american": home_odds,
        "away_odds_american": away_odds,
        "commence_time": "2025-07-01T18:00:00Z",
        "home_bat_avg": 0.260,
        "away_bat_avg": 0.255,
        "home_era": 3.80,
        "away_era": 4.10,
    }])


GAME_DATE = date(2025, 7, 1)


# ---------- tests ----------

def test_find_edges_returns_home_edge(tmp_path):
    """Home side with EV > threshold and odds >= MIN_AMERICAN_ODDS is flagged."""
    from mlb_edge_finder.edge_finder import find_edges
    # home_prob=0.75, home_odds=+110 → EV = 0.75*1.10 - 0.25 = 0.575 > 0.05 ✓
    # away_prob=0.25, away_odds=-140 → EV = 0.25*(100/140) - 0.75 = -0.571 < 0.05 ✗
    features_df = _make_features(home_odds=110, away_odds=-140)
    clf = _make_clf(home_proba=0.75)

    with patch("mlb_edge_finder.edge_finder.config.DATA_PROCESSED_DIR", tmp_path), \
         patch("mlb_edge_finder.edge_finder.config.EV_THRESHOLD", 0.05), \
         patch("mlb_edge_finder.edge_finder.config.MIN_AMERICAN_ODDS", -300):
        result = find_edges(features_df, clf, GAME_DATE)

    assert len(result) == 1
    assert result.iloc[0]["bet_side"] == "home"
    assert result.iloc[0]["american_odds"] == 110
    assert abs(result.iloc[0]["model_prob"] - 0.75) < 1e-6
    assert result.iloc[0]["ev"] > 0.05
    assert set(result.columns) == {"game_id", "home_team", "away_team", "bet_side", "american_odds", "model_prob", "ev"}


def test_find_edges_filters_min_odds(tmp_path):
    """Game where EV > threshold but odds are below MIN_AMERICAN_ODDS is excluded."""
    from mlb_edge_finder.edge_finder import find_edges
    # home_prob=0.90, home_odds=-400 → EV = 0.90*(100/400) - 0.10 = 0.125 > 0.05 ✓
    # BUT -400 < -300 (MIN_AMERICAN_ODDS) → excluded
    features_df = _make_features(home_odds=-400, away_odds=310)
    clf = _make_clf(home_proba=0.90)

    with patch("mlb_edge_finder.edge_finder.config.DATA_PROCESSED_DIR", tmp_path), \
         patch("mlb_edge_finder.edge_finder.config.EV_THRESHOLD", 0.05), \
         patch("mlb_edge_finder.edge_finder.config.MIN_AMERICAN_ODDS", -300):
        result = find_edges(features_df, clf, GAME_DATE)

    assert result.empty
    assert set(result.columns) == {"game_id", "home_team", "away_team", "bet_side", "american_odds", "model_prob", "ev"}


def test_find_edges_empty_when_no_edges(tmp_path):
    """Returns empty DataFrame with correct columns when no games pass filters."""
    from mlb_edge_finder.edge_finder import find_edges
    # home_prob=0.50, home_odds=-110 → EV = 0.50*(100/110) - 0.50 = -0.045 < 0.05 ✗
    features_df = _make_features(home_odds=-110, away_odds=-110)
    clf = _make_clf(home_proba=0.50)

    with patch("mlb_edge_finder.edge_finder.config.DATA_PROCESSED_DIR", tmp_path), \
         patch("mlb_edge_finder.edge_finder.config.EV_THRESHOLD", 0.05), \
         patch("mlb_edge_finder.edge_finder.config.MIN_AMERICAN_ODDS", -300):
        result = find_edges(features_df, clf, GAME_DATE)

    assert result.empty
    assert set(result.columns) == {"game_id", "home_team", "away_team", "bet_side", "american_odds", "model_prob", "ev"}


def test_find_edges_both_sides(tmp_path):
    """Both home and away edges are returned when both sides pass filters."""
    from mlb_edge_finder.edge_finder import find_edges
    # home_prob=0.60, home_odds=+130 → EV = 0.60*1.30 - 0.40 = 0.38 > 0.05 ✓
    # away_prob=0.40, away_odds=+200 → EV = 0.40*2.00 - 0.60 = 0.20 > 0.05 ✓
    features_df = _make_features(home_odds=130, away_odds=200)
    clf = _make_clf(home_proba=0.60)

    with patch("mlb_edge_finder.edge_finder.config.DATA_PROCESSED_DIR", tmp_path), \
         patch("mlb_edge_finder.edge_finder.config.EV_THRESHOLD", 0.05), \
         patch("mlb_edge_finder.edge_finder.config.MIN_AMERICAN_ODDS", -300):
        result = find_edges(features_df, clf, GAME_DATE)

    assert len(result) == 2
    assert set(result["bet_side"]) == {"home", "away"}


def test_find_edges_missing_feature_column(tmp_path):
    """Raises ValueError when features_df is missing a column the model needs."""
    from mlb_edge_finder.edge_finder import find_edges
    import pandas as pd
    features_df = pd.DataFrame([{
        "game_id": "game_1",
        "home_team": "New York Yankees",
        "away_team": "Boston Red Sox",
        "home_odds_american": 110,
        "away_odds_american": -130,
        # missing home_bat_avg, away_bat_avg, home_era, away_era
    }])
    clf = _make_clf(home_proba=0.60)

    with pytest.raises(ValueError, match="missing columns"):
        find_edges(features_df, clf, GAME_DATE)
```

- [ ] **Step 2: Run the new tests to verify they all fail**

```bash
pytest tests/test_edge_finder.py::test_find_edges_returns_home_edge \
       tests/test_edge_finder.py::test_find_edges_filters_min_odds \
       tests/test_edge_finder.py::test_find_edges_empty_when_no_edges \
       tests/test_edge_finder.py::test_find_edges_both_sides \
       tests/test_edge_finder.py::test_find_edges_missing_feature_column \
       -v
```

Expected: all 5 FAIL with `NotImplementedError` (from the current stub).

---

## Task 2: Implement `find_edges()`

**Files:**
- Modify: `src/mlb_edge_finder/edge_finder.py`

- [ ] **Step 1: Replace the `find_edges` stub**

Replace the entire `find_edges` function (lines 34–54) with:

```python
def find_edges(features_df: pd.DataFrame, clf: XGBClassifier, game_date: date) -> pd.DataFrame:
    """Run inference and return games with positive expected value.

    Uses clf.feature_names_in_ to select exactly the columns the model was
    trained on, then runs two passes (home, away) to find bets where:
      - EV > config.EV_THRESHOLD
      - The relevant team's American odds >= config.MIN_AMERICAN_ODDS

    Logs a warning and returns an empty DataFrame (with correct columns) if no
    edges are found. Writes results to DATA_PROCESSED_DIR/edges_YYYY-MM-DD.csv.

    Args:
        features_df: Output of features.load_features() or build_features().
            Must contain all columns in clf.feature_names_in_, plus
            game_id, home_team, away_team, home_odds_american, away_odds_american.
        clf: Fitted XGBClassifier from model.load_model() or train().
        game_date: Used to name the output CSV.

    Returns:
        DataFrame with columns: game_id, home_team, away_team,
        bet_side, american_odds, model_prob, ev — one row per flagged edge.

    Raises:
        ValueError: If features_df is missing any column in clf.feature_names_in_.
    """
    output_cols = ["game_id", "home_team", "away_team", "bet_side", "american_odds", "model_prob", "ev"]

    feature_cols = list(clf.feature_names_in_)
    missing = [c for c in feature_cols if c not in features_df.columns]
    if missing:
        raise ValueError(f"features_df missing columns required by model: {missing}")

    df = features_df.reset_index(drop=True)
    X = df[feature_cols]
    home_prob = clf.predict_proba(X)[:, 1]
    away_prob = 1.0 - home_prob

    # Home pass
    home_ev = pd.Series(
        [compute_ev(float(p), int(o)) for p, o in zip(home_prob, df["home_odds_american"])]
    )
    home_mask = (home_ev > config.EV_THRESHOLD) & (df["home_odds_american"] >= config.MIN_AMERICAN_ODDS)
    home_edges = df.loc[home_mask, ["game_id", "home_team", "away_team"]].copy()
    home_edges["bet_side"] = "home"
    home_edges["american_odds"] = df.loc[home_mask, "home_odds_american"].values
    home_edges["model_prob"] = home_prob[home_mask.values]
    home_edges["ev"] = home_ev[home_mask].values

    # Away pass
    away_ev = pd.Series(
        [compute_ev(float(p), int(o)) for p, o in zip(away_prob, df["away_odds_american"])]
    )
    away_mask = (away_ev > config.EV_THRESHOLD) & (df["away_odds_american"] >= config.MIN_AMERICAN_ODDS)
    away_edges = df.loc[away_mask, ["game_id", "home_team", "away_team"]].copy()
    away_edges["bet_side"] = "away"
    away_edges["american_odds"] = df.loc[away_mask, "away_odds_american"].values
    away_edges["model_prob"] = away_prob[away_mask.values]
    away_edges["ev"] = away_ev[away_mask].values

    edges = pd.concat([home_edges[output_cols], away_edges[output_cols]], ignore_index=True)

    if edges.empty:
        logger.warning("No edges found for %s", game_date)
        return pd.DataFrame(columns=output_cols)

    config.DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.DATA_PROCESSED_DIR / f"edges_{game_date}.csv"
    edges.to_csv(out_path, index=False)
    logger.info("Found %d edge(s) for %s → %s", len(edges), game_date, out_path)

    return edges
```

- [ ] **Step 2: Run the 5 new tests to verify they all pass**

```bash
pytest tests/test_edge_finder.py -v
```

Expected: all 9 tests PASS (4 original + 5 new).

- [ ] **Step 3: Run the full test suite to check for regressions**

```bash
pytest tests/ -v
```

Expected: all tests PASS.

- [ ] **Step 4: Commit**

```bash
git add tests/test_edge_finder.py src/mlb_edge_finder/edge_finder.py
git commit -m "feat: implement edge_finder.find_edges()"
```

---

## Task 3: Tests for `pipeline.run()`

**Files:**
- Create: `tests/test_pipeline.py`

- [ ] **Step 1: Create the test file**

```python
"""Tests for pipeline.run()."""
import pickle
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


def _make_mock_clf():
    clf = MagicMock()
    clf.feature_names_in_ = np.array(["home_bat_avg", "away_bat_avg"])
    clf.predict_proba.return_value = np.array([[0.4, 0.6]])
    return clf


def _write_pkl(path: Path, clf) -> None:
    with open(path, "wb") as f:
        pickle.dump(clf, f)


def test_run_returns_edges(tmp_path):
    """pipeline.run() returns the edges DataFrame produced by find_edges."""
    from mlb_edge_finder import pipeline

    # Write a real pkl so load_model can read it
    model_date = date(2025, 9, 28)
    clf = _make_mock_clf()
    _write_pkl(tmp_path / f"xgb_{model_date}.pkl", clf)

    features_df = pd.DataFrame([{
        "game_id": "game_1",
        "home_team": "New York Yankees",
        "away_team": "Boston Red Sox",
        "home_odds_american": 130,
        "away_odds_american": 200,
        "commence_time": "2025-07-01T18:00:00Z",
        "home_bat_avg": 0.260,
        "away_bat_avg": 0.255,
    }])
    expected_edges = pd.DataFrame([{
        "game_id": "game_1", "home_team": "New York Yankees",
        "away_team": "Boston Red Sox", "bet_side": "home",
        "american_odds": 130, "model_prob": 0.6, "ev": 0.38,
    }])

    with patch("mlb_edge_finder.pipeline.odds_ingestion.fetch_odds", return_value=pd.DataFrame()), \
         patch("mlb_edge_finder.pipeline.stats_ingestion.fetch_stats", return_value=pd.DataFrame()), \
         patch("mlb_edge_finder.pipeline.features.build_features", return_value=features_df), \
         patch("mlb_edge_finder.pipeline.config.MODELS_DIR", tmp_path), \
         patch("mlb_edge_finder.pipeline.edge_finder.find_edges", return_value=expected_edges) as mock_edges:
        result = pipeline.run(date(2025, 7, 1))

    mock_edges.assert_called_once()
    assert len(result) == 1


def test_run_defaults_to_today():
    """pipeline.run() defaults game_date to date.today() when not provided."""
    from mlb_edge_finder import pipeline
    from datetime import date as date_cls

    captured = {}

    def fake_fetch_odds(game_date):
        captured["game_date"] = game_date
        return pd.DataFrame()

    with patch("mlb_edge_finder.pipeline.odds_ingestion.fetch_odds", side_effect=fake_fetch_odds), \
         patch("mlb_edge_finder.pipeline.stats_ingestion.fetch_stats", return_value=pd.DataFrame()), \
         patch("mlb_edge_finder.pipeline.features.build_features", return_value=pd.DataFrame()), \
         patch("mlb_edge_finder.pipeline.config.MODELS_DIR") as mock_dir, \
         patch("mlb_edge_finder.pipeline.edge_finder.find_edges", return_value=pd.DataFrame()):
        mock_dir.glob.return_value = []  # triggers FileNotFoundError before find_edges
        with pytest.raises(FileNotFoundError):
            pipeline.run()

    assert captured["game_date"] == date_cls.today()


def test_run_raises_when_no_models(tmp_path):
    """pipeline.run() raises FileNotFoundError when MODELS_DIR has no pkl files."""
    from mlb_edge_finder import pipeline

    with patch("mlb_edge_finder.pipeline.odds_ingestion.fetch_odds", return_value=pd.DataFrame()), \
         patch("mlb_edge_finder.pipeline.stats_ingestion.fetch_stats", return_value=pd.DataFrame()), \
         patch("mlb_edge_finder.pipeline.features.build_features", return_value=pd.DataFrame()), \
         patch("mlb_edge_finder.pipeline.config.MODELS_DIR", tmp_path):
        with pytest.raises(FileNotFoundError, match="No trained models"):
            pipeline.run(date(2025, 7, 1))
```

- [ ] **Step 2: Run the new tests to verify they all fail**

```bash
pytest tests/test_pipeline.py -v
```

Expected: all 3 FAIL with `NotImplementedError` (from the current stub).

---

## Task 4: Implement `pipeline.run()`

**Files:**
- Modify: `src/mlb_edge_finder/pipeline.py`

- [ ] **Step 1: Replace the `run` stub**

Replace the entire contents of `src/mlb_edge_finder/pipeline.py` with:

```python
"""Orchestrate all pipeline stages from odds ingestion to edge output."""
import logging
from datetime import date

import pandas as pd

from mlb_edge_finder import config, edge_finder, features, model, odds_ingestion, stats_ingestion

logger = logging.getLogger(__name__)


def run(game_date: date | None = None) -> pd.DataFrame:
    """Run the full MLB edge-finding pipeline for a single game date.

    Stages (in order):
      1. Fetch or load moneyline odds for game_date.
      2. Fetch or load team stats up to game_date.
      3. Build feature DataFrame from odds + stats.
      4. Auto-discover and load the most recently saved model from MODELS_DIR.
      5. Run edge_finder.find_edges() and return the result.

    Args:
        game_date: Date to run the pipeline for. Defaults to today.

    Returns:
        DataFrame of flagged edges (may be empty if none found).
        Same schema as edge_finder.find_edges().

    Raises:
        FileNotFoundError: If no trained models exist in MODELS_DIR.
    """
    if game_date is None:
        game_date = date.today()

    logger.info("Running pipeline for %s", game_date)

    odds_ingestion.fetch_odds(game_date)
    stats_ingestion.fetch_stats(game_date)
    features_df = features.build_features(game_date)

    pkls = sorted(config.MODELS_DIR.glob("xgb_*.pkl"))
    if not pkls:
        raise FileNotFoundError(
            "No trained models found in MODELS_DIR — run model.train() and save_model() first"
        )
    # Filenames are xgb_YYYY-MM-DD.pkl; lexicographic sort puts latest last
    latest_date = date.fromisoformat(pkls[-1].stem[4:])  # strip "xgb_"
    clf = model.load_model(latest_date)
    logger.info("Loaded model from %s", pkls[-1].name)

    return edge_finder.find_edges(features_df, clf, game_date)
```

- [ ] **Step 2: Run the pipeline tests**

```bash
pytest tests/test_pipeline.py -v
```

Expected: all 3 PASS.

- [ ] **Step 3: Run the full test suite**

```bash
pytest tests/ -v
```

Expected: all tests PASS.

- [ ] **Step 4: Commit**

```bash
git add tests/test_pipeline.py src/mlb_edge_finder/pipeline.py
git commit -m "feat: implement pipeline.run() with auto model discovery"
```

---

## Task 5: Update CLAUDE.md

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Mark Phase 5 complete in the roadmap**

In the `## Roadmap` section, change:

```markdown
- [ ] Implement `edge_finder.find_edges()`
- [ ] Implement `pipeline.run()`
```

to:

```markdown
- [x] Implement `edge_finder.find_edges()`
- [x] Implement `pipeline.run()`
```

- [ ] **Step 2: Update the "Current Phase" section**

Replace the current phase description at the top with:

```markdown
**Phase 5 complete.** Phases 1–3, 4a–4c, and 5 are done. Next: `compute_kelly()`, `__main__.py` CLI entry point.

- `edge_finder.find_edges(features_df, clf, game_date)` — uses `clf.feature_names_in_` to select inference features, runs sequential home/away EV passes, filters by `EV_THRESHOLD` and `MIN_AMERICAN_ODDS`, writes `data/processed/edges_YYYY-MM-DD.csv`.
- `pipeline.run(game_date)` — orchestrates all five stages end-to-end; auto-discovers latest model by globbing `MODELS_DIR` for `xgb_*.pkl` sorted by filename date.
```

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: mark Phase 5 complete in CLAUDE.md"
```
