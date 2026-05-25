# Historical Backtest Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Simulate the edge-finder's historical performance on the held-out 20% test split to produce a P&L curve that validates the model's edge-finding claims.

**Architecture:** New `backtest.py` module with three functions — `simulate_market_odds`, `run_backtest`, `compute_summary`. Synthetic market odds (−110/−110, standard 4.76% vig, even 50/50 market) are applied to the same test split used in model evaluation (80/20 stratified, `random_state=42`). This answers: "Would we have been profitable betting on games the model had never seen, against a naive −110/−110 market?" A new notebook `notebooks/02_backtest.ipynb` visualises the cumulative P&L curve. No new API calls or external dependencies.

**Tech Stack:** pandas, matplotlib, scikit-learn `train_test_split`, existing `mlb_edge_finder` modules (`model`, `edge_finder`, `training_data`, `config`)

**Assumption documented in notebook:** Synthetic market odds assume every game is priced at −110/−110 (52.38% implied per side, 4.76% vig). Real bookmaker lines vary by game. This backtest tests model skill relative to a naive even-money market, not real-world profitability.

---

## File Structure

| File | Action | Purpose |
|---|---|---|
| `src/mlb_edge_finder/backtest.py` | Create | `simulate_market_odds`, `run_backtest`, `compute_summary` |
| `tests/test_backtest.py` | Create | Unit tests for all three functions |
| `notebooks/02_backtest.ipynb` | Create | Load model, run backtest, print summary, plot P&L curve |

No existing files are modified.

---

## Key Design Decisions

**Test split:** `run_backtest` replicates the exact same 80/20 stratified split from `model._three_way_split()` — specifically the outer `train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)` — so the backtest evaluates the same held-out games used in `model.evaluate()`. No data leakage.

**Metadata recovery:** After splitting, `X_test` only has feature columns (NON_FEATURE_COLS are dropped before splitting). The test row index is preserved, so `training_df.loc[X_test.index, ["game_date", "home_name", "away_name"]]` recovers display columns.

**Market odds:** `simulate_market_odds(home_market_prob=0.5, vig=0.0476)` → `(-110.0, -110.0)`. Both sides get the same odds because the market is assumed 50/50. The vig is split additively: `home_implied = home_market_prob + vig/2`, `away_implied = (1−home_market_prob) + vig/2`.

**American odds conversion:**
```
p >= 0.5 (favorite):  american = −(p / (1−p)) × 100
p < 0.5  (underdog):  american = ((1−p) / p) × 100
```

**P&L per bet:** Winning a bet at American odds `o`:
```
payout = o/100 if o > 0 else 100/abs(o)
pnl_win  = payout × unit
pnl_loss = −unit
```

**`compute_summary` output keys:** `n_bets`, `n_wins`, `win_rate`, `total_pnl`, `roi_pct`, `avg_ev`, `max_drawdown`, `sharpe_ratio`. Sharpe is per-bet (mean / std of individual bet P&Ls); not annualised — document this in the notebook.

---

## Task 1: `backtest.py` module — `simulate_market_odds`

**Files:**
- Create: `src/mlb_edge_finder/backtest.py`
- Create: `tests/test_backtest.py`

- [ ] **Step 1: Write failing tests for `simulate_market_odds`**

Create `tests/test_backtest.py`:

```python
import pytest
from mlb_edge_finder.backtest import simulate_market_odds


def test_simulate_market_odds_default_is_110_110():
    home_american, away_american = simulate_market_odds()
    assert abs(home_american - (-110.0)) < 1.0
    assert abs(away_american - (-110.0)) < 1.0


def test_simulate_market_odds_favored_home():
    home_american, away_american = simulate_market_odds(home_market_prob=0.6)
    assert home_american < 0   # home is favorite
    assert away_american > 0   # away is underdog


def test_simulate_market_odds_implied_probs_sum_to_1_plus_vig():
    """Implied probs must sum to 1 + vig (bookmaker overround)."""
    vig = 0.05
    home_american, away_american = simulate_market_odds(home_market_prob=0.55, vig=vig)

    def to_implied(american: float) -> float:
        if american < 0:
            return abs(american) / (abs(american) + 100)
        return 100 / (american + 100)

    total = to_implied(home_american) + to_implied(away_american)
    assert abs(total - (1 + vig)) < 0.01


def test_simulate_market_odds_even_home_prob():
    home_american, away_american = simulate_market_odds(home_market_prob=0.5, vig=0.0476)
    # Both sides should be equal (symmetric market)
    assert abs(home_american - away_american) < 0.01


def test_simulate_market_odds_zero_vig_gives_fair_odds():
    # At p=0.6, no vig: home = -(0.6/0.4)*100 = -150, away = (0.4/0.6)*100 ≈ +66.7
    home_american, away_american = simulate_market_odds(home_market_prob=0.6, vig=0.0)
    assert abs(home_american - (-150.0)) < 1.0
    assert abs(away_american - (66.67)) < 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/jaydengould/Documents/projects/mlb-edge-finder
pytest tests/test_backtest.py -v
```

Expected: `ModuleNotFoundError: No module named 'mlb_edge_finder.backtest'`

- [ ] **Step 3: Create `backtest.py` with `simulate_market_odds`**

Create `src/mlb_edge_finder/backtest.py`:

```python
"""Backtest the edge-finder against held-out test data using synthetic market odds."""
import logging

import pandas as pd

logger = logging.getLogger(__name__)


def simulate_market_odds(
    home_market_prob: float = 0.5,
    vig: float = 0.0476,
) -> tuple[float, float]:
    """Generate synthetic American odds for both sides of a game.

    Splits the vig additively across home and away implied probabilities,
    then converts each to American odds format.

    At the default home_market_prob=0.5, vig=0.0476, both sides return
    approximately -110.0 (the standard even-money MLB line).

    Args:
        home_market_prob: Market-implied probability that home wins. Default 0.5.
        vig: Bookmaker overround (sum of implied probs minus 1). Default 0.0476.

    Returns:
        (home_american, away_american) as floats.
    """
    home_implied = home_market_prob + vig / 2
    away_implied = (1.0 - home_market_prob) + vig / 2

    def _to_american(p: float) -> float:
        if p >= 0.5:
            return -(p / (1.0 - p)) * 100.0
        return ((1.0 - p) / p) * 100.0

    return _to_american(home_implied), _to_american(away_implied)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_backtest.py -v
```

Expected: 5 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/mlb_edge_finder/backtest.py tests/test_backtest.py
git commit -m "feat: add backtest module with simulate_market_odds"
```

---

## Task 2: `run_backtest` function

**Files:**
- Modify: `src/mlb_edge_finder/backtest.py`
- Modify: `tests/test_backtest.py`

- [ ] **Step 1: Write failing tests for `run_backtest`**

Append to `tests/test_backtest.py`:

```python
import numpy as np
import pandas as pd
from unittest.mock import MagicMock


def _make_training_df(n: int = 200, seed: int = 0) -> pd.DataFrame:
    """Minimal training DataFrame that satisfies run_backtest requirements."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "game_date": pd.date_range("2024-04-01", periods=n, freq="D"),
        "home_name": [f"HomeTeam{i % 15}" for i in range(n)],
        "away_name": [f"AwayTeam{i % 15}" for i in range(n)],
        "home_score": rng.integers(0, 10, n),
        "away_score": rng.integers(0, 10, n),
        "home_abbr": [f"HM{i % 15}" for i in range(n)],
        "away_abbr": [f"AW{i % 15}" for i in range(n)],
        "season": [2024] * n,
        "home_win": rng.integers(0, 2, n),
        "home_starter_name": [None] * n,
        "away_starter_name": [None] * n,
        "home_pitcher_id": [None] * n,
        "away_pitcher_id": [None] * n,
        "feature_a": rng.standard_normal(n),
        "feature_b": rng.standard_normal(n),
        "feature_c": rng.standard_normal(n),
    })
    return df


def _make_mock_clf(n_features: int = 3, home_win_prob: float = 0.58):
    """Mock classifier that always predicts home_win_prob for class 1."""
    clf = MagicMock()
    clf.feature_names_in_ = np.array(["feature_a", "feature_b", "feature_c"])
    clf.predict_proba = MagicMock(
        side_effect=lambda X: np.column_stack([
            np.full(len(X), 1.0 - home_win_prob),
            np.full(len(X), home_win_prob),
        ])
    )
    return clf


from mlb_edge_finder.backtest import run_backtest


def test_run_backtest_returns_dataframe():
    df = _make_training_df(200)
    clf = _make_mock_clf(home_win_prob=0.58)
    result = run_backtest(clf, df)
    assert isinstance(result, pd.DataFrame)


def test_run_backtest_output_columns():
    df = _make_training_df(200)
    clf = _make_mock_clf(home_win_prob=0.58)
    result = run_backtest(clf, df)
    expected_cols = {
        "game_date", "home_name", "away_name", "bet_side",
        "american_odds", "model_prob", "ev", "kelly_fraction",
        "actual_home_win", "won", "pnl", "cumulative_pnl",
    }
    assert expected_cols.issubset(set(result.columns))


def test_run_backtest_no_edges_returns_empty_with_correct_columns():
    """When model predicts 0.50 for all games, no edge clears the 5% EV threshold."""
    df = _make_training_df(200)
    clf = _make_mock_clf(home_win_prob=0.50)
    result = run_backtest(clf, df)
    assert result.empty
    expected_cols = {
        "game_date", "home_name", "away_name", "bet_side",
        "american_odds", "model_prob", "ev", "kelly_fraction",
        "actual_home_win", "won", "pnl", "cumulative_pnl",
    }
    assert expected_cols.issubset(set(result.columns))


def test_run_backtest_high_prob_finds_home_edges():
    """Model predicting 0.65 home win should flag home bets against -110/-110 market."""
    df = _make_training_df(200)
    clf = _make_mock_clf(home_win_prob=0.65)
    result = run_backtest(clf, df)
    assert not result.empty
    assert (result["bet_side"] == "home").any()


def test_run_backtest_cumulative_pnl_is_running_sum():
    df = _make_training_df(200)
    clf = _make_mock_clf(home_win_prob=0.65)
    result = run_backtest(clf, df)
    if not result.empty:
        expected = result["pnl"].cumsum().values
        pd.testing.assert_series_equal(
            result["cumulative_pnl"].reset_index(drop=True),
            pd.Series(expected, name="cumulative_pnl"),
            check_exact=False,
            atol=1e-6,
        )


def test_run_backtest_won_matches_actual_outcome():
    df = _make_training_df(200)
    clf = _make_mock_clf(home_win_prob=0.65)
    result = run_backtest(clf, df)
    if not result.empty:
        home_bets = result[result["bet_side"] == "home"]
        if not home_bets.empty:
            assert (home_bets["won"] == (home_bets["actual_home_win"] == 1)).all()
        away_bets = result[result["bet_side"] == "away"]
        if not away_bets.empty:
            assert (away_bets["won"] == (away_bets["actual_home_win"] == 0)).all()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_backtest.py::test_run_backtest_returns_dataframe -v
```

Expected: `ImportError: cannot import name 'run_backtest'`

- [ ] **Step 3: Implement `run_backtest`**

Append to `src/mlb_edge_finder/backtest.py`:

```python
from typing import Any


def run_backtest(
    clf: Any,
    training_df: pd.DataFrame,
    home_market_prob: float = 0.5,
    vig: float = 0.0476,
    unit: float = 100.0,
) -> pd.DataFrame:
    """Simulate edge-finder performance on the held-out 20% test split.

    Replicates the same 80/20 stratified split used in model._three_way_split()
    (test_size=0.2, random_state=42) so the evaluated games are identical to
    those used in model.evaluate(). No data leakage.

    Synthetic market odds are generated by simulate_market_odds(home_market_prob, vig).
    The default -110/-110 market assumes 50/50 game pricing — the backtest measures
    whether the model outperforms a naive even-money assumption.

    Args:
        clf: Fitted calibrated classifier from model.load_model(). Must have
            feature_names_in_ and predict_proba() attributes.
        training_df: Full training DataFrame from training_data.load_training_set().
            Must contain home_win, game_date, home_name, away_name, and all feature
            columns referenced by clf.feature_names_in_.
        home_market_prob: Market-implied home win probability. Default 0.5.
        vig: Bookmaker overround. Default 0.0476 (≈ -110/-110 standard line).
        unit: Dollar bet size for P&L calculation. Default $100.

    Returns:
        DataFrame sorted by game_date with columns: game_date, home_name, away_name,
        bet_side, american_odds, model_prob, ev, kelly_fraction, actual_home_win,
        won, pnl, cumulative_pnl. Returns empty DataFrame (with those columns) if
        no bets clear the EV threshold.
    """
    from sklearn.model_selection import train_test_split

    from mlb_edge_finder.config import EV_THRESHOLD, MIN_AMERICAN_ODDS
    from mlb_edge_finder.edge_finder import compute_ev, compute_kelly
    from mlb_edge_finder.model import NON_FEATURE_COLS, TARGET_COL

    output_cols = [
        "game_date", "home_name", "away_name", "bet_side",
        "american_odds", "model_prob", "ev", "kelly_fraction",
        "actual_home_win", "won", "pnl", "cumulative_pnl",
    ]

    non_feature = [c for c in NON_FEATURE_COLS if c in training_df.columns]
    X = training_df.drop(columns=non_feature)
    y = training_df[TARGET_COL]

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    feature_names = list(clf.feature_names_in_)
    X_test_aligned = X_test.reindex(columns=feature_names)

    home_probs = clf.predict_proba(X_test_aligned)[:, 1]

    home_odds, away_odds = simulate_market_odds(home_market_prob, vig)
    home_payout = home_odds / 100 if home_odds > 0 else 100 / abs(home_odds)
    away_payout = away_odds / 100 if away_odds > 0 else 100 / abs(away_odds)

    meta = training_df.loc[X_test.index, ["game_date", "home_name", "away_name"]]

    records = []
    for (idx, prob), actual in zip(zip(X_test.index, home_probs), y_test.values):
        row_meta = meta.loc[idx]

        home_ev = compute_ev(float(prob), round(home_odds))
        if home_ev > EV_THRESHOLD and home_odds >= MIN_AMERICAN_ODDS:
            won = int(actual) == 1
            records.append({
                "game_date": row_meta["game_date"],
                "home_name": row_meta["home_name"],
                "away_name": row_meta["away_name"],
                "bet_side": "home",
                "american_odds": round(home_odds),
                "model_prob": round(float(prob), 4),
                "ev": round(home_ev, 4),
                "kelly_fraction": round(compute_kelly(float(prob), round(home_odds)), 4),
                "actual_home_win": int(actual),
                "won": won,
                "pnl": home_payout * unit if won else -unit,
            })

        away_prob = 1.0 - float(prob)
        away_ev = compute_ev(away_prob, round(away_odds))
        if away_ev > EV_THRESHOLD and away_odds >= MIN_AMERICAN_ODDS:
            won = int(actual) == 0
            records.append({
                "game_date": row_meta["game_date"],
                "home_name": row_meta["home_name"],
                "away_name": row_meta["away_name"],
                "bet_side": "away",
                "american_odds": round(away_odds),
                "model_prob": round(away_prob, 4),
                "ev": round(away_ev, 4),
                "kelly_fraction": round(compute_kelly(away_prob, round(away_odds)), 4),
                "actual_home_win": int(actual),
                "won": won,
                "pnl": away_payout * unit if won else -unit,
            })

    if not records:
        logger.warning("No edges found in backtest — model may not outperform even-money market")
        return pd.DataFrame(columns=output_cols)

    result = pd.DataFrame(records).sort_values("game_date").reset_index(drop=True)
    result["cumulative_pnl"] = result["pnl"].cumsum()
    logger.info("Backtest complete: %d bets placed across %d test games", len(result), len(X_test))
    return result
```

- [ ] **Step 4: Run all backtest tests**

```bash
pytest tests/test_backtest.py -v
```

Expected: all tests pass (5 from Task 1 + 6 from this task = 11 total).

- [ ] **Step 5: Commit**

```bash
git add src/mlb_edge_finder/backtest.py tests/test_backtest.py
git commit -m "feat: add run_backtest — synthetic market backtest on held-out test split"
```

---

## Task 3: `compute_summary` function

**Files:**
- Modify: `src/mlb_edge_finder/backtest.py`
- Modify: `tests/test_backtest.py`

- [ ] **Step 1: Write failing tests for `compute_summary`**

Append to `tests/test_backtest.py`:

```python
from mlb_edge_finder.backtest import compute_summary


def _make_backtest_df(pnl_values: list[float]) -> pd.DataFrame:
    """Minimal backtest DataFrame for compute_summary testing."""
    n = len(pnl_values)
    return pd.DataFrame({
        "game_date": pd.date_range("2024-04-01", periods=n),
        "home_name": ["HomeTeam"] * n,
        "away_name": ["AwayTeam"] * n,
        "bet_side": ["home"] * n,
        "american_odds": [-110] * n,
        "model_prob": [0.60] * n,
        "ev": [0.08] * n,
        "kelly_fraction": [0.04] * n,
        "actual_home_win": [1, 0, 1, 0, 1, 1, 0, 1, 1, 0][:n],
        "won": [p > 0 for p in pnl_values],
        "pnl": pnl_values,
        "cumulative_pnl": pd.Series(pnl_values).cumsum().tolist(),
    })


def test_compute_summary_keys():
    df = _make_backtest_df([90.9, -100, 90.9, -100, 90.9])
    result = compute_summary(df, unit=100.0)
    expected_keys = {
        "n_bets", "n_wins", "win_rate", "total_pnl",
        "roi_pct", "avg_ev", "max_drawdown", "sharpe_ratio",
    }
    assert expected_keys == set(result.keys())


def test_compute_summary_n_bets_and_wins():
    pnl = [90.9, -100, 90.9, -100, 90.9]  # 3 wins, 2 losses
    df = _make_backtest_df(pnl)
    result = compute_summary(df, unit=100.0)
    assert result["n_bets"] == 5
    assert result["n_wins"] == 3


def test_compute_summary_win_rate():
    pnl = [90.9, -100, 90.9, -100, 90.9]  # 3/5 = 60%
    df = _make_backtest_df(pnl)
    result = compute_summary(df, unit=100.0)
    assert abs(result["win_rate"] - 0.60) < 0.01


def test_compute_summary_total_pnl():
    pnl = [90.9, -100.0, 90.9]
    df = _make_backtest_df(pnl)
    result = compute_summary(df, unit=100.0)
    assert abs(result["total_pnl"] - sum(pnl)) < 0.01


def test_compute_summary_roi_pct():
    # 2 bets at $100 unit, total pnl = $50 → ROI = 50/200 * 100 = 25%
    pnl = [150.0, -100.0]
    df = _make_backtest_df(pnl)
    result = compute_summary(df, unit=100.0)
    assert abs(result["roi_pct"] - 25.0) < 0.01


def test_compute_summary_max_drawdown():
    # cumulative: 100, 0, 50 → peak=100 at index 0, trough=0 at index 1 → drawdown=100
    pnl = [100.0, -100.0, 50.0]
    df = _make_backtest_df(pnl)
    df["cumulative_pnl"] = pd.Series(pnl).cumsum()
    result = compute_summary(df, unit=100.0)
    assert abs(result["max_drawdown"] - 100.0) < 0.01


def test_compute_summary_empty_df_returns_zeros():
    df = pd.DataFrame(columns=[
        "game_date", "home_name", "away_name", "bet_side",
        "american_odds", "model_prob", "ev", "kelly_fraction",
        "actual_home_win", "won", "pnl", "cumulative_pnl",
    ])
    result = compute_summary(df, unit=100.0)
    assert result["n_bets"] == 0
    assert result["total_pnl"] == 0.0
    assert result["roi_pct"] == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_backtest.py::test_compute_summary_keys -v
```

Expected: `ImportError: cannot import name 'compute_summary'`

- [ ] **Step 3: Implement `compute_summary`**

Append to `src/mlb_edge_finder/backtest.py`:

```python
def compute_summary(backtest_df: pd.DataFrame, unit: float = 100.0) -> dict:
    """Compute headline performance metrics for a completed backtest.

    Args:
        backtest_df: Output of run_backtest(). Must have columns pnl,
            cumulative_pnl, won, ev. Accepts empty DataFrames (all metrics = 0).
        unit: Dollar bet size used in run_backtest(). Used to compute ROI.

    Returns:
        Dict with keys:
            n_bets       — total number of bets placed
            n_wins       — number of winning bets
            win_rate     — n_wins / n_bets (0.0 if no bets)
            total_pnl    — sum of all bet P&Ls
            roi_pct      — total_pnl / (n_bets * unit) * 100
            avg_ev       — mean expected value of flagged bets
            max_drawdown — largest peak-to-trough drop in cumulative P&L
            sharpe_ratio — per-bet Sharpe: mean(pnl) / std(pnl), or 0.0 if std=0
    """
    if backtest_df.empty:
        return {
            "n_bets": 0,
            "n_wins": 0,
            "win_rate": 0.0,
            "total_pnl": 0.0,
            "roi_pct": 0.0,
            "avg_ev": 0.0,
            "max_drawdown": 0.0,
            "sharpe_ratio": 0.0,
        }

    n_bets = len(backtest_df)
    n_wins = int(backtest_df["won"].sum())
    total_pnl = float(backtest_df["pnl"].sum())

    cumulative = backtest_df["cumulative_pnl"]
    running_max = cumulative.cummax()
    max_drawdown = float((running_max - cumulative).max())

    pnl_std = backtest_df["pnl"].std()
    sharpe = float(backtest_df["pnl"].mean() / pnl_std) if pnl_std > 0 else 0.0

    return {
        "n_bets": n_bets,
        "n_wins": n_wins,
        "win_rate": round(n_wins / n_bets, 4),
        "total_pnl": round(total_pnl, 2),
        "roi_pct": round(total_pnl / (n_bets * unit) * 100, 2),
        "avg_ev": round(float(backtest_df["ev"].mean()), 4),
        "max_drawdown": round(max_drawdown, 2),
        "sharpe_ratio": round(sharpe, 4),
    }
```

- [ ] **Step 4: Run all backtest tests**

```bash
pytest tests/test_backtest.py -v
```

Expected: all 18 tests pass.

- [ ] **Step 5: Run full test suite**

```bash
pytest tests/ -v
```

Expected: all existing tests still pass (140+) plus the 18 new backtest tests.

- [ ] **Step 6: Commit**

```bash
git add src/mlb_edge_finder/backtest.py tests/test_backtest.py
git commit -m "feat: add compute_summary to backtest module"
```

---

## Task 4: `notebooks/02_backtest.ipynb`

**Files:**
- Create: `notebooks/02_backtest.ipynb`

This notebook is the portfolio artifact — it loads the saved model, runs the backtest, prints the summary table, and shows the cumulative P&L curve.

- [ ] **Step 1: Create the notebook with all cells**

Create `notebooks/02_backtest.ipynb` as a valid Jupyter notebook with the following cells. Use the `nbformat` library or write the JSON directly.

**Cell 1 — Setup:**
```python
import logging
from datetime import date

import matplotlib.pyplot as plt
import pandas as pd

from mlb_edge_finder import config
from mlb_edge_finder.backtest import compute_summary, run_backtest
from mlb_edge_finder.model import load_model
from mlb_edge_finder.training_data import load_training_set

config.setup_logging()
```

**Cell 2 — Load model and training data:**
```python
# Auto-discover the latest saved model
import glob

model_files = sorted(glob.glob(str(config.MODELS_DIR / "xgb_*.pkl")))
if not model_files:
    raise FileNotFoundError("No saved model found. Run the training cells in 01_exploration.ipynb first.")

latest_model_path = model_files[-1]
model_date_str = latest_model_path.split("xgb_")[-1].replace(".pkl", "")
model_date = date.fromisoformat(model_date_str)

print(f"Loading model: {latest_model_path}")
clf = load_model(model_date)

# Load all training seasons — same set used to train the model
SEASONS = [2019, 2021, 2022, 2023, 2024, 2025]
training_df = load_training_set(SEASONS)
print(f"Training data: {len(training_df):,} games across {SEASONS}")
```

**Cell 3 — Run backtest:**
```python
# Backtest assumption: synthetic market odds of -110/-110 (50/50, 4.76% vig)
# This tests whether the model outperforms a naive even-money market.
# Real bookmaker lines vary by game — this is a lower bound on real-world profitability.
backtest_df = run_backtest(clf, training_df, home_market_prob=0.5, vig=0.0476, unit=100)
print(f"Bets placed: {len(backtest_df)}")
backtest_df.head(10)
```

**Cell 4 — Summary statistics:**
```python
summary = compute_summary(backtest_df, unit=100.0)

print("=" * 40)
print("BACKTEST SUMMARY")
print("=" * 40)
print(f"  Bets placed  : {summary['n_bets']}")
print(f"  Wins         : {summary['n_wins']}")
print(f"  Win rate     : {summary['win_rate']:.1%}")
print(f"  Total P&L    : ${summary['total_pnl']:+,.2f}")
print(f"  ROI          : {summary['roi_pct']:+.2f}%")
print(f"  Avg EV       : {summary['avg_ev']:.2%}")
print(f"  Max drawdown : ${summary['max_drawdown']:,.2f}")
print(f"  Sharpe ratio : {summary['sharpe_ratio']:.3f} (per-bet, not annualised)")
print("=" * 40)
print()
print("Assumptions:")
print("  - Market odds: -110 / -110 (50/50 even market, 4.76% vig)")
print("  - Bet size: $100 flat per flagged edge")
print("  - Test set: held-out 20% (stratified random split, same as model evaluation)")
print("  - Lookahead note: end-of-season team stats used for all games in each season")
```

**Cell 5 — Cumulative P&L plot:**
```python
if backtest_df.empty:
    print("No bets found — model did not outperform the even-money market at the 5% EV threshold.")
else:
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Cumulative P&L
    axes[0].plot(backtest_df.index, backtest_df["cumulative_pnl"], color="steelblue", linewidth=1.5)
    axes[0].axhline(0, color="black", linewidth=0.8, linestyle="--")
    axes[0].fill_between(
        backtest_df.index,
        backtest_df["cumulative_pnl"],
        0,
        where=backtest_df["cumulative_pnl"] >= 0,
        alpha=0.3,
        color="green",
    )
    axes[0].fill_between(
        backtest_df.index,
        backtest_df["cumulative_pnl"],
        0,
        where=backtest_df["cumulative_pnl"] < 0,
        alpha=0.3,
        color="red",
    )
    axes[0].set_title("Cumulative P&L — Backtest on held-out 20% test split ($100/bet, -110/-110 market)")
    axes[0].set_xlabel("Bet number")
    axes[0].set_ylabel("Cumulative P&L ($)")

    # Distribution of individual bet P&Ls
    wins = backtest_df[backtest_df["won"]]["pnl"]
    losses = backtest_df[~backtest_df["won"]]["pnl"]
    axes[1].hist(wins, bins=20, color="green", alpha=0.6, label=f"Wins ({len(wins)})")
    axes[1].hist(losses, bins=20, color="red", alpha=0.6, label=f"Losses ({len(losses)})")
    axes[1].axvline(0, color="black", linewidth=0.8, linestyle="--")
    axes[1].set_title("Distribution of individual bet P&Ls")
    axes[1].set_xlabel("P&L ($)")
    axes[1].set_ylabel("Count")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("notebooks/backtest_pnl.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Plot saved to notebooks/backtest_pnl.png")
```

The notebook JSON structure:

```json
{
 "nbformat": 4,
 "nbformat_minor": 5,
 "metadata": {
  "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
  "language_info": {"name": "python", "version": "3.10.0"}
 },
 "cells": [
  {"cell_type": "code", "id": "setup", "metadata": {}, "execution_count": null, "outputs": [],
   "source": "<Cell 1 source>"},
  {"cell_type": "code", "id": "load", "metadata": {}, "execution_count": null, "outputs": [],
   "source": "<Cell 2 source>"},
  {"cell_type": "code", "id": "run", "metadata": {}, "execution_count": null, "outputs": [],
   "source": "<Cell 3 source>"},
  {"cell_type": "code", "id": "summary", "metadata": {}, "execution_count": null, "outputs": [],
   "source": "<Cell 4 source>"},
  {"cell_type": "code", "id": "plot", "metadata": {}, "execution_count": null, "outputs": [],
   "source": "<Cell 5 source>"}
 ]
}
```

Write the actual notebook file with real cell source content (not placeholders). Use `json.dumps` or write directly. The `source` field for each cell must be a list of strings, one per line, with `\n` at the end of each line except the last.

- [ ] **Step 2: Verify the notebook runs end-to-end**

Run all cells:
```bash
cd /Users/jaydengould/Documents/projects/mlb-edge-finder
jupyter nbconvert --to notebook --execute notebooks/02_backtest.ipynb --output notebooks/02_backtest.ipynb
```

Expected: notebook executes without errors. If `load_training_set` or `load_model` raises `FileNotFoundError`, the cached data is missing — note this in the output but do not fail the task (the user may need to run `01_exploration.ipynb` first to populate caches).

- [ ] **Step 3: Commit**

```bash
git add notebooks/02_backtest.ipynb notebooks/backtest_pnl.png
git commit -m "feat: add backtest notebook with cumulative P&L curve"
```

---

## Self-Review

### Spec coverage
- `simulate_market_odds` — covered Task 1 ✓
- `run_backtest` using same split as model evaluation — covered Task 2 ✓
- `compute_summary` with all required metrics — covered Task 3 ✓
- Notebook with P&L curve and summary table — covered Task 4 ✓
- No new API calls or dependencies — confirmed (only existing modules + matplotlib) ✓
- Empty DataFrame returned when no edges found — handled in both `run_backtest` and `compute_summary` ✓

### Placeholder scan
All code blocks are complete. No TBDs. No "similar to Task N" references. ✓

### Type consistency
- `simulate_market_odds` returns `tuple[float, float]` — consumed correctly by `run_backtest` via `round()` cast ✓
- `run_backtest` returns `pd.DataFrame` with `cumulative_pnl` column — consumed by `compute_summary` ✓
- `compute_summary` takes `unit: float = 100.0` — matches `run_backtest` default ✓
- `compute_ev(float, int)` and `compute_kelly(float, int)` — american odds rounded to int via `round()` ✓
