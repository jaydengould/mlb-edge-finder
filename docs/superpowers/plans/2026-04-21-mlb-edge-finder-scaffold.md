# MLB Edge Finder — Scaffold Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Scaffold the full mlb-edge-finder project — src/ layout, all placeholder modules with typed signatures and docstrings, test stubs, config, requirements, and README.

**Architecture:** Flat `src/mlb_edge_finder/` package installed in editable mode (`pip install -e .`) so notebooks can import modules without path hacks. Each module owns one pipeline stage and persists its output as a dated CSV or model artifact. Config is the single source of truth for paths, thresholds, and env vars.

**Tech Stack:** Python 3.10+, pandas, pybaseball, requests (Odds API), scikit-learn, XGBoost, python-dotenv, pytest, jupyter.

---

## File Map

| File | Purpose |
|---|---|
| `pyproject.toml` | Declares package, enables `pip install -e .` |
| `requirements.txt` | All runtime + dev dependencies |
| `src/mlb_edge_finder/__init__.py` | Package marker (empty) |
| `src/mlb_edge_finder/config.py` | Env loading, path constants, `setup_logging()` |
| `src/mlb_edge_finder/odds_ingestion.py` | Fetch/cache moneyline odds from The Odds API |
| `src/mlb_edge_finder/stats_ingestion.py` | Fetch/cache team + pitcher stats via pybaseball |
| `src/mlb_edge_finder/features.py` | Merge odds + stats into model-ready DataFrame |
| `src/mlb_edge_finder/model.py` | Train, evaluate, persist XGBoost model |
| `src/mlb_edge_finder/edge_finder.py` | Compute EV, filter odds, flag edges |
| `src/mlb_edge_finder/pipeline.py` | Orchestrate all stages end-to-end |
| `tests/test_config.py` | Smoke test: config imports, constants present |
| `tests/test_odds_ingestion.py` | Smoke test: public API exists |
| `tests/test_stats_ingestion.py` | Smoke test: public API exists |
| `tests/test_features.py` | Smoke test: public API exists |
| `tests/test_model.py` | Smoke test: public API exists |
| `tests/test_edge_finder.py` | Smoke test: EV math + public API |
| `.env.template` | Committed env key skeleton |
| `README.md` | Project overview skeleton |
| `notebooks/01_exploration.ipynb` | Starter notebook |

---

## Task 1: Project Foundation

**Files:**
- Create: `pyproject.toml`
- Create: `requirements.txt`
- Create: `src/mlb_edge_finder/__init__.py`
- Create: `tests/__init__.py`

- [ ] **Step 1: Create directory structure**

```bash
mkdir -p src/mlb_edge_finder tests notebooks data/raw data/processed models logs
```

- [ ] **Step 2: Write `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "mlb-edge-finder"
version = "0.1.0"
requires-python = ">=3.10"

[tool.setuptools.packages.find]
where = ["src"]
```

- [ ] **Step 3: Write `requirements.txt`**

```
pandas>=2.0
pybaseball>=2.2
requests>=2.31
scikit-learn>=1.4
xgboost>=2.0
python-dotenv>=1.0
jupyter>=1.0
notebook>=7.0
pytest>=8.0
```

- [ ] **Step 4: Create package and test markers**

`src/mlb_edge_finder/__init__.py` — empty file.

`tests/__init__.py` — empty file.

- [ ] **Step 5: Install package in editable mode**

```bash
pip install -e .
```

Expected: `Successfully installed mlb-edge-finder-0.1.0`

- [ ] **Step 6: Update `.gitignore` to cover new directories**

Add this line to `.gitignore`:

```
logs/
```

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml requirements.txt src/ tests/ .gitignore
git commit -m "chore: scaffold project foundation and editable install"
```

---

## Task 2: `config.py`

**Files:**
- Create: `src/mlb_edge_finder/config.py`
- Create: `tests/test_config.py`

- [ ] **Step 1: Write the failing test**

`tests/test_config.py`:

```python
"""Smoke tests: config imports cleanly and exposes expected interface."""
import logging


def test_config_imports():
    from mlb_edge_finder import config
    assert hasattr(config, "ODDS_API_KEY")
    assert hasattr(config, "DATA_RAW_DIR")
    assert hasattr(config, "DATA_PROCESSED_DIR")
    assert hasattr(config, "MODELS_DIR")
    assert hasattr(config, "XGB_N_ESTIMATORS")
    assert hasattr(config, "XGB_MAX_DEPTH")
    assert hasattr(config, "EV_THRESHOLD")
    assert hasattr(config, "MIN_AMERICAN_ODDS")


def test_setup_logging_is_callable():
    from mlb_edge_finder import config
    assert callable(config.setup_logging)


def test_setup_logging_runs():
    from mlb_edge_finder import config
    config.setup_logging(level=logging.DEBUG)
    logger = logging.getLogger("test")
    logger.debug("config smoke test")
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_config.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'mlb_edge_finder.config'` (or similar import error)

- [ ] **Step 3: Write `config.py`**

`src/mlb_edge_finder/config.py`:

```python
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# --- Env-sourced ---
ODDS_API_KEY: str = os.getenv("ODDS_API_KEY", "")
SPORT: str = os.getenv("SPORT", "baseball_mlb")
REGION: str = os.getenv("REGION", "us")
MARKET: str = os.getenv("MARKET", "h2h")

# --- Paths ---
_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW_DIR: Path = _ROOT / "data" / "raw"
DATA_PROCESSED_DIR: Path = _ROOT / "data" / "processed"
MODELS_DIR: Path = _ROOT / "models"
LOGS_DIR: Path = _ROOT / "logs"

# --- Model hyperparameters ---
XGB_N_ESTIMATORS: int = 100
XGB_MAX_DEPTH: int = 4

# --- Edge-finding thresholds ---
EV_THRESHOLD: float = 0.05
MIN_AMERICAN_ODDS: int = -300


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logger with a console handler and a file handler at logs/run.log."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    handlers: list[logging.Handler] = [
        logging.StreamHandler(),
        logging.FileHandler(LOGS_DIR / "run.log"),
    ]
    logging.basicConfig(level=level, format=fmt, handlers=handlers, force=True)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_config.py -v
```

Expected: 3 PASSED

- [ ] **Step 5: Commit**

```bash
git add src/mlb_edge_finder/config.py tests/test_config.py
git commit -m "feat: add config module with env loading, path constants, and logging setup"
```

---

## Task 3: `odds_ingestion.py`

**Files:**
- Create: `src/mlb_edge_finder/odds_ingestion.py`
- Create: `tests/test_odds_ingestion.py`

- [ ] **Step 1: Write the failing test**

`tests/test_odds_ingestion.py`:

```python
"""Smoke tests: odds_ingestion exposes expected public API."""
import inspect


def test_fetch_odds_signature():
    """fetch_odds should accept game_date and return a DataFrame."""
    from mlb_edge_finder import odds_ingestion
    assert callable(odds_ingestion.fetch_odds)
    sig = inspect.signature(odds_ingestion.fetch_odds)
    assert "game_date" in sig.parameters


def test_load_cached_odds_signature():
    """load_cached_odds should accept game_date and return a DataFrame."""
    from mlb_edge_finder import odds_ingestion
    assert callable(odds_ingestion.load_cached_odds)
    sig = inspect.signature(odds_ingestion.load_cached_odds)
    assert "game_date" in sig.parameters
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_odds_ingestion.py -v
```

Expected: FAIL — `ImportError` or `AttributeError`

- [ ] **Step 3: Write `odds_ingestion.py`**

`src/mlb_edge_finder/odds_ingestion.py`:

```python
"""Fetch and cache MLB moneyline odds from The Odds API."""
import logging
from datetime import date
from pathlib import Path

import pandas as pd
import requests

from mlb_edge_finder import config

logger = logging.getLogger(__name__)


def fetch_odds(game_date: date) -> pd.DataFrame:
    """Fetch MLB moneyline odds from The Odds API for a given date.

    Calls GET /v4/sports/{sport}/odds with market=h2h for the configured
    region. Writes the raw response to DATA_RAW_DIR/odds_YYYY-MM-DD.csv
    before returning.

    Args:
        game_date: The date for which to fetch odds.

    Returns:
        DataFrame with columns: game_id, home_team, away_team,
        home_odds_american, away_odds_american, bookmaker, commence_time.

    Raises:
        RuntimeError: If the API request returns a non-200 status.
    """
    raise NotImplementedError


def load_cached_odds(game_date: date) -> pd.DataFrame:
    """Load previously fetched odds from DATA_RAW_DIR/odds_YYYY-MM-DD.csv.

    Args:
        game_date: The date whose cached CSV to load.

    Returns:
        DataFrame with the same schema as fetch_odds().

    Raises:
        FileNotFoundError: If no cached file exists for the given date.
    """
    raise NotImplementedError
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_odds_ingestion.py -v
```

Expected: 2 PASSED

- [ ] **Step 5: Commit**

```bash
git add src/mlb_edge_finder/odds_ingestion.py tests/test_odds_ingestion.py
git commit -m "feat: add odds_ingestion module scaffold with public API"
```

---

## Task 4: `stats_ingestion.py`

**Files:**
- Create: `src/mlb_edge_finder/stats_ingestion.py`
- Create: `tests/test_stats_ingestion.py`

- [ ] **Step 1: Write the failing test**

`tests/test_stats_ingestion.py`:

```python
"""Smoke tests: stats_ingestion exposes expected public API."""
import inspect


def test_fetch_stats_signature():
    """fetch_stats should accept start_date and end_date."""
    from mlb_edge_finder import stats_ingestion
    assert callable(stats_ingestion.fetch_stats)
    sig = inspect.signature(stats_ingestion.fetch_stats)
    assert "start_date" in sig.parameters
    assert "end_date" in sig.parameters


def test_load_cached_stats_signature():
    """load_cached_stats should accept game_date."""
    from mlb_edge_finder import stats_ingestion
    assert callable(stats_ingestion.load_cached_stats)
    sig = inspect.signature(stats_ingestion.load_cached_stats)
    assert "game_date" in sig.parameters
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_stats_ingestion.py -v
```

Expected: FAIL — `ImportError` or `AttributeError`

- [ ] **Step 3: Write `stats_ingestion.py`**

`src/mlb_edge_finder/stats_ingestion.py`:

```python
"""Fetch and cache team and pitcher stats via pybaseball."""
import logging
from datetime import date

import pandas as pd

from mlb_edge_finder import config

logger = logging.getLogger(__name__)


def fetch_stats(start_date: date, end_date: date) -> pd.DataFrame:
    """Fetch team batting and starting pitcher stats for a date range.

    Uses pybaseball.team_batting() and pybaseball.pitching_stats() to pull
    season-to-date aggregates. Writes result to
    DATA_RAW_DIR/stats_YYYY-MM-DD.csv (keyed by end_date).

    Args:
        start_date: First date of the window (inclusive).
        end_date: Last date of the window (inclusive).

    Returns:
        DataFrame with columns: team, era, whip, batting_avg, ops,
        runs_per_game, home_away (placeholder columns — finalize during
        feature engineering design).

    Raises:
        RuntimeError: If pybaseball fails to return data.
    """
    raise NotImplementedError


def load_cached_stats(game_date: date) -> pd.DataFrame:
    """Load previously fetched stats from DATA_RAW_DIR/stats_YYYY-MM-DD.csv.

    Args:
        game_date: The date whose cached CSV to load.

    Returns:
        DataFrame with the same schema as fetch_stats().

    Raises:
        FileNotFoundError: If no cached file exists for the given date.
    """
    raise NotImplementedError
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_stats_ingestion.py -v
```

Expected: 2 PASSED

- [ ] **Step 5: Commit**

```bash
git add src/mlb_edge_finder/stats_ingestion.py tests/test_stats_ingestion.py
git commit -m "feat: add stats_ingestion module scaffold with public API"
```

---

## Task 5: `features.py`

**Files:**
- Create: `src/mlb_edge_finder/features.py`
- Create: `tests/test_features.py`

- [ ] **Step 1: Write the failing test**

`tests/test_features.py`:

```python
"""Smoke tests: features exposes expected public API."""
import inspect


def test_build_features_signature():
    """build_features should accept odds_df and stats_df DataFrames."""
    from mlb_edge_finder import features
    assert callable(features.build_features)
    sig = inspect.signature(features.build_features)
    assert "odds_df" in sig.parameters
    assert "stats_df" in sig.parameters


def test_load_features_signature():
    """load_features should accept game_date."""
    from mlb_edge_finder import features
    assert callable(features.load_features)
    sig = inspect.signature(features.load_features)
    assert "game_date" in sig.parameters
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_features.py -v
```

Expected: FAIL — `ImportError` or `AttributeError`

- [ ] **Step 3: Write `features.py`**

`src/mlb_edge_finder/features.py`:

```python
"""Merge odds and stats into a model-ready feature DataFrame."""
import logging
from datetime import date

import pandas as pd

from mlb_edge_finder import config

logger = logging.getLogger(__name__)


def build_features(odds_df: pd.DataFrame, stats_df: pd.DataFrame) -> pd.DataFrame:
    """Join odds and stats on team name and engineer model features.

    Computes implied probability from American odds, merges team-level
    stats for both home and away sides, and writes the result to
    DATA_PROCESSED_DIR/features_YYYY-MM-DD.csv.

    Args:
        odds_df: Output of odds_ingestion.fetch_odds() or load_cached_odds().
        stats_df: Output of stats_ingestion.fetch_stats() or load_cached_stats().

    Returns:
        DataFrame with one row per game and feature columns ready for
        XGBoost training or inference. Includes implied_prob_home,
        implied_prob_away, and all engineered stat differentials.

    Raises:
        ValueError: If odds_df or stats_df are empty.
    """
    raise NotImplementedError


def load_features(game_date: date) -> pd.DataFrame:
    """Load a previously built feature DataFrame from DATA_PROCESSED_DIR.

    Args:
        game_date: The date whose features CSV to load.

    Returns:
        DataFrame with the same schema as build_features().

    Raises:
        FileNotFoundError: If no features file exists for the given date.
    """
    raise NotImplementedError
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_features.py -v
```

Expected: 2 PASSED

- [ ] **Step 5: Commit**

```bash
git add src/mlb_edge_finder/features.py tests/test_features.py
git commit -m "feat: add features module scaffold with public API"
```

---

## Task 6: `model.py`

**Files:**
- Create: `src/mlb_edge_finder/model.py`
- Create: `tests/test_model.py`

- [ ] **Step 1: Write the failing test**

`tests/test_model.py`:

```python
"""Smoke tests: model exposes expected public API."""
import inspect


def test_train_signature():
    from mlb_edge_finder import model
    assert callable(model.train)
    sig = inspect.signature(model.train)
    assert "features_df" in sig.parameters


def test_evaluate_signature():
    from mlb_edge_finder import model
    assert callable(model.evaluate)
    sig = inspect.signature(model.evaluate)
    assert "clf" in sig.parameters
    assert "X_test" in sig.parameters
    assert "y_test" in sig.parameters


def test_save_model_signature():
    from mlb_edge_finder import model
    assert callable(model.save_model)
    sig = inspect.signature(model.save_model)
    assert "clf" in sig.parameters
    assert "metrics" in sig.parameters
    assert "game_date" in sig.parameters


def test_load_model_signature():
    from mlb_edge_finder import model
    assert callable(model.load_model)
    sig = inspect.signature(model.load_model)
    assert "game_date" in sig.parameters
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_model.py -v
```

Expected: FAIL — `ImportError` or `AttributeError`

- [ ] **Step 3: Write `model.py`**

`src/mlb_edge_finder/model.py`:

```python
"""Train, evaluate, and persist the XGBoost win-probability model."""
import json
import logging
import pickle
from datetime import date
from typing import Any

import pandas as pd
from xgboost import XGBClassifier

from mlb_edge_finder import config

logger = logging.getLogger(__name__)

# Column that holds the binary win/loss target in the features DataFrame.
TARGET_COL = "home_win"


def train(features_df: pd.DataFrame) -> XGBClassifier:
    """Train an XGBoost classifier to predict home-team win probability.

    Splits features_df into train/test sets (80/20), fits an XGBClassifier
    using config.XGB_N_ESTIMATORS and config.XGB_MAX_DEPTH, and returns
    the trained model. Does not persist — call save_model() separately.

    Args:
        features_df: Output of features.build_features() or load_features().
            Must contain TARGET_COL as the label column.

    Returns:
        Fitted XGBClassifier instance.

    Raises:
        FileNotFoundError: If features_df is empty.
        ValueError: If TARGET_COL is missing from features_df.
    """
    raise NotImplementedError


def evaluate(clf: XGBClassifier, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, Any]:
    """Compute evaluation metrics for a trained classifier.

    Args:
        clf: Fitted XGBClassifier from train().
        X_test: Feature matrix (rows = games, columns = feature columns).
        y_test: True binary labels (1 = home win, 0 = away win).

    Returns:
        Dict with keys: accuracy, roc_auc, log_loss, n_test_samples,
        xgb_n_estimators, xgb_max_depth.
    """
    raise NotImplementedError


def save_model(clf: XGBClassifier, metrics: dict[str, Any], game_date: date) -> None:
    """Persist a trained model and its metrics to disk.

    Writes two files to MODELS_DIR:
      - xgb_YYYY-MM-DD.pkl  — pickled XGBClassifier object
      - metrics_YYYY-MM-DD.json — JSON with eval metrics and hyperparameters

    Args:
        clf: Fitted XGBClassifier to persist.
        metrics: Output of evaluate().
        game_date: Used to name the output files.
    """
    raise NotImplementedError


def load_model(game_date: date) -> XGBClassifier:
    """Load a previously saved XGBClassifier from MODELS_DIR.

    Args:
        game_date: The date whose .pkl file to load.

    Returns:
        Fitted XGBClassifier ready for inference.

    Raises:
        FileNotFoundError: If no model file exists for the given date.
    """
    raise NotImplementedError
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_model.py -v
```

Expected: 4 PASSED

- [ ] **Step 5: Commit**

```bash
git add src/mlb_edge_finder/model.py tests/test_model.py
git commit -m "feat: add model module scaffold with train/evaluate/save/load API"
```

---

## Task 7: `edge_finder.py`

**Files:**
- Create: `src/mlb_edge_finder/edge_finder.py`
- Create: `tests/test_edge_finder.py`

- [ ] **Step 1: Write the failing test**

`tests/test_edge_finder.py`:

```python
"""Smoke tests: edge_finder exposes expected public API and EV math is correct."""
import inspect
import pytest


def test_compute_ev_signature():
    from mlb_edge_finder import edge_finder
    assert callable(edge_finder.compute_ev)
    sig = inspect.signature(edge_finder.compute_ev)
    assert "prob" in sig.parameters
    assert "american_odds" in sig.parameters


def test_compute_ev_favorite():
    """Negative American odds: EV = prob * (100 / abs(odds)) - (1 - prob)."""
    from mlb_edge_finder.edge_finder import compute_ev
    # 60% model prob, -150 line → EV = 0.60*(100/150) - 0.40 = 0.40 - 0.40 = 0.00
    ev = compute_ev(prob=0.60, american_odds=-150)
    assert abs(ev) < 1e-9


def test_compute_ev_underdog():
    """Positive American odds: EV = prob * (odds / 100) - (1 - prob)."""
    from mlb_edge_finder.edge_finder import compute_ev
    # 40% model prob, +150 line → EV = 0.40*(150/100) - 0.60 = 0.60 - 0.60 = 0.00
    ev = compute_ev(prob=0.40, american_odds=150)
    assert abs(ev) < 1e-9


def test_find_edges_signature():
    from mlb_edge_finder import edge_finder
    assert callable(edge_finder.find_edges)
    sig = inspect.signature(edge_finder.find_edges)
    assert "features_df" in sig.parameters
    assert "clf" in sig.parameters
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_edge_finder.py -v
```

Expected: FAIL — `ImportError` or `AttributeError`

- [ ] **Step 3: Write `edge_finder.py`**

`src/mlb_edge_finder/edge_finder.py`:

```python
"""Compute expected value and identify positive-EV betting edges."""
import logging
from datetime import date

import pandas as pd
from xgboost import XGBClassifier

from mlb_edge_finder import config

logger = logging.getLogger(__name__)


def compute_ev(prob: float, american_odds: int) -> float:
    """Compute expected value of a bet given model probability and American odds.

    Formula:
        Favorites (negative odds): EV = prob * (100 / abs(odds)) - (1 - prob)
        Underdogs (positive odds): EV = prob * (odds / 100) - (1 - prob)

    Args:
        prob: Model-predicted win probability for the team (0.0 – 1.0).
        american_odds: Bookmaker's American moneyline for the same team.

    Returns:
        Expected value per unit wagered. Positive = profitable edge.
    """
    if american_odds < 0:
        payout = 100 / abs(american_odds)
    else:
        payout = american_odds / 100
    return prob * payout - (1 - prob)


def find_edges(features_df: pd.DataFrame, clf: XGBClassifier) -> pd.DataFrame:
    """Run inference and return games with positive expected value.

    Uses clf to predict home-win probabilities, computes EV for both sides
    via compute_ev(), then filters to rows where:
      - EV > config.EV_THRESHOLD
      - The relevant team's American odds >= config.MIN_AMERICAN_ODDS

    Logs a warning and returns an empty DataFrame if no edges are found.
    Writes results to DATA_PROCESSED_DIR/edges_YYYY-MM-DD.csv.

    Args:
        features_df: Output of features.load_features() or build_features().
            Must contain home_odds_american and away_odds_american columns.
        clf: Fitted XGBClassifier from model.load_model() or train().

    Returns:
        DataFrame with columns: game_id, home_team, away_team,
        bet_side, american_odds, model_prob, ev — one row per flagged edge.
    """
    raise NotImplementedError
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_edge_finder.py -v
```

Expected: 4 PASSED

- [ ] **Step 5: Commit**

```bash
git add src/mlb_edge_finder/edge_finder.py tests/test_edge_finder.py
git commit -m "feat: add edge_finder module with compute_ev implementation and find_edges scaffold"
```

---

## Task 8: `pipeline.py`

**Files:**
- Create: `src/mlb_edge_finder/pipeline.py`
- (No separate test file — pipeline is orchestration; covered when modules are implemented)

- [ ] **Step 1: Write `pipeline.py`**

`src/mlb_edge_finder/pipeline.py`:

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
      2. Fetch or load team/pitcher stats up to game_date.
      3. Build feature DataFrame from odds + stats.
      4. Load the most recently saved model from MODELS_DIR.
      5. Run edge_finder.find_edges() and return the result.

    Args:
        game_date: Date to run the pipeline for. Defaults to today.

    Returns:
        DataFrame of flagged edges (may be empty if none found).
        Same schema as edge_finder.find_edges().
    """
    raise NotImplementedError
```

- [ ] **Step 2: Verify import is clean**

```bash
python -c "from mlb_edge_finder import pipeline; print('pipeline OK')"
```

Expected: `pipeline OK`

- [ ] **Step 3: Commit**

```bash
git add src/mlb_edge_finder/pipeline.py
git commit -m "feat: add pipeline orchestration scaffold"
```

---

## Task 9: `.env.template` and `README.md`

**Files:**
- Create: `.env.template`
- Create: `README.md`

- [ ] **Step 1: Write `.env.template`**

```
ODDS_API_KEY=
SPORT=baseball_mlb
REGION=us
MARKET=h2h
```

- [ ] **Step 2: Write `README.md`**

```markdown
# MLB Edge Finder

A portfolio project that identifies positive expected-value (EV) opportunities in MLB moneyline betting markets by comparing model-predicted win probabilities against bookmaker-implied probabilities.

## Tech Stack

Python · pandas · pybaseball · The Odds API · scikit-learn · XGBoost · python-dotenv

## Setup

```bash
# 1. Clone and install
pip install -e .

# 2. Configure secrets
cp .env.template .env
# Edit .env and add your Odds API key

# 3. Launch the starter notebook
jupyter notebook notebooks/01_exploration.ipynb
```

## Project Structure

```
src/mlb_edge_finder/
├── config.py           # env loading, path constants, logging
├── odds_ingestion.py   # fetch moneyline odds (The Odds API)
├── stats_ingestion.py  # fetch team/pitcher stats (pybaseball)
├── features.py         # merge odds + stats into feature DataFrame
├── model.py            # train, evaluate, persist XGBoost model
├── edge_finder.py      # compute EV, flag positive-EV bets
└── pipeline.py         # end-to-end orchestration
```

## Edge Definition

A bet is flagged as an edge when:
- Model EV > 5% (configurable via `EV_THRESHOLD` in `config.py`)
- American odds >= -300 (configurable via `MIN_AMERICAN_ODDS`)

## Roadmap

- [ ] Implement all module stubs
- [ ] Kelly criterion bet sizing (`compute_kelly()` in `edge_finder.py`)
- [ ] CLI entry point (`python -m mlb_edge_finder`)
- [ ] Scheduled daily runs (APScheduler)
```

- [ ] **Step 3: Commit**

```bash
git add .env.template README.md
git commit -m "docs: add README skeleton and .env.template"
```

---

## Task 10: Starter Notebook

**Files:**
- Create: `notebooks/01_exploration.ipynb`

- [ ] **Step 1: Write the starter notebook**

`notebooks/01_exploration.ipynb`:

```json
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": ["# MLB Edge Finder — Exploration\n", "\n", "Interactive walkthrough of the pipeline stages."]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import logging\n",
    "from datetime import date\n",
    "\n",
    "from mlb_edge_finder import config\n",
    "\n",
    "config.setup_logging(level=logging.INFO)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": ["## 1. Fetch Odds"]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from mlb_edge_finder import odds_ingestion\n",
    "\n",
    "game_date = date.today()\n",
    "# odds_df = odds_ingestion.fetch_odds(game_date)\n",
    "# odds_df.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": ["## 2. Fetch Stats"]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from mlb_edge_finder import stats_ingestion\n",
    "\n",
    "# stats_df = stats_ingestion.fetch_stats(date(2025, 4, 1), game_date)\n",
    "# stats_df.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": ["## 3. Build Features"]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from mlb_edge_finder import features\n",
    "\n",
    "# features_df = features.build_features(odds_df, stats_df)\n",
    "# features_df.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": ["## 4. Train Model"]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from mlb_edge_finder import model\n",
    "\n",
    "# clf = model.train(features_df)\n",
    "# model.save_model(clf, metrics, game_date)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": ["## 5. Find Edges"]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from mlb_edge_finder import edge_finder\n",
    "\n",
    "# edges = edge_finder.find_edges(features_df, clf)\n",
    "# edges"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.10.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
```

- [ ] **Step 2: Commit**

```bash
git add notebooks/01_exploration.ipynb
git commit -m "feat: add starter exploration notebook"
```

---

## Task 11: Full Test Suite Pass

- [ ] **Step 1: Run the full test suite**

```bash
pytest tests/ -v
```

Expected output (all PASSED):
```
tests/test_config.py::test_config_imports PASSED
tests/test_config.py::test_setup_logging_is_callable PASSED
tests/test_config.py::test_setup_logging_runs PASSED
tests/test_odds_ingestion.py::test_fetch_odds_signature PASSED
tests/test_odds_ingestion.py::test_load_cached_odds_signature PASSED
tests/test_stats_ingestion.py::test_fetch_stats_signature PASSED
tests/test_stats_ingestion.py::test_load_cached_stats_signature PASSED
tests/test_features.py::test_build_features_signature PASSED
tests/test_features.py::test_load_features_signature PASSED
tests/test_model.py::test_train_signature PASSED
tests/test_model.py::test_evaluate_signature PASSED
tests/test_model.py::test_save_model_signature PASSED
tests/test_model.py::test_load_model_signature PASSED
tests/test_edge_finder.py::test_compute_ev_signature PASSED
tests/test_edge_finder.py::test_compute_ev_favorite PASSED
tests/test_edge_finder.py::test_compute_ev_underdog PASSED
tests/test_edge_finder.py::test_find_edges_signature PASSED
```

- [ ] **Step 2: Final commit**

```bash
git add -A
git commit -m "chore: verify full scaffold test suite passes"
```
