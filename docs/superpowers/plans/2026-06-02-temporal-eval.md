# Temporal Out-of-Time Evaluation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a temporal holdout evaluation (train 2019–2024, test on 2025) that replaces the random-split backtest numbers and P&L chart on the dashboard.

**Architecture:** Extract `simulate_bets()` from `backtest.py` so the bet loop can be called on pre-split data. Add `temporal_eval.py` that loads the training set, splits by `season`, trains a fresh model on the train slice, and writes a single JSON artifact with model metrics + backtest summary + P&L series. Update `generate_site.py` to read that JSON as its single source of truth for all dashboard evaluation numbers.

**Tech Stack:** XGBoost, scikit-learn, pandas, Chart.js (dashboard), pytest

---

## File Map

| File | Change |
|---|---|
| `src/mlb_edge_finder/backtest.py` | Extract `simulate_bets()` from `run_backtest()`; `run_backtest()` delegates to it |
| `src/mlb_edge_finder/temporal_eval.py` | New module — `_temporal_split()`, `run()`, `__main__` CLI |
| `src/mlb_edge_finder/generate_site.py` | Replace `_load_metrics`+`_load_pnl` with `_load_temporal_eval`; update `generate()` signature; update JS P&L chart data source; add card subtitle |
| `tests/test_backtest.py` | Add `simulate_bets` tests |
| `tests/test_temporal_eval.py` | New test file |
| `tests/test_generate_site.py` | Update `generate()` call sites; replace metrics/pnl tests with temporal eval tests |
| `models/temporal_eval_2025.json` | Generated artifact — committed after running `python -m mlb_edge_finder.temporal_eval` |

---

## Task 1: Extract `simulate_bets()` from `backtest.py`

**Files:**
- Modify: `src/mlb_edge_finder/backtest.py`
- Test: `tests/test_backtest.py`

- [ ] **Step 1: Write failing tests for `simulate_bets`**

Add to the bottom of `tests/test_backtest.py`:

```python
from mlb_edge_finder.backtest import simulate_bets


def _make_aligned_split(n: int = 200, seed: int = 0):
    """Return (clf, X_test, y_test, meta_df) ready for simulate_bets."""
    from sklearn.model_selection import train_test_split
    df = _make_training_df(n, seed)
    from mlb_edge_finder.model import NON_FEATURE_COLS, TARGET_COL
    non_feature = [c for c in NON_FEATURE_COLS if c in df.columns]
    X = df.drop(columns=non_feature)
    y = df[TARGET_COL]
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    meta = df.loc[X_test.index, ["game_date", "home_name", "away_name"]]
    clf = _make_mock_clf(home_win_prob=0.65)
    return clf, X_test, y_test, meta


def test_simulate_bets_returns_dataframe():
    clf, X_test, y_test, meta = _make_aligned_split()
    result = simulate_bets(clf, X_test, y_test, meta)
    assert isinstance(result, pd.DataFrame)


def test_simulate_bets_output_columns():
    clf, X_test, y_test, meta = _make_aligned_split()
    result = simulate_bets(clf, X_test, y_test, meta, ev_threshold=0.05)
    expected = {
        "game_date", "home_name", "away_name", "bet_side",
        "american_odds", "model_prob", "ev", "kelly_fraction",
        "actual_home_win", "won", "pnl", "cumulative_pnl",
    }
    assert expected.issubset(set(result.columns))


def test_simulate_bets_no_edges_returns_empty():
    clf, X_test, y_test, meta = _make_aligned_split()
    clf = _make_mock_clf(home_win_prob=0.50)
    result = simulate_bets(clf, X_test, y_test, meta)
    assert result.empty


def test_simulate_bets_high_prob_finds_edges():
    clf, X_test, y_test, meta = _make_aligned_split()
    result = simulate_bets(clf, X_test, y_test, meta, ev_threshold=0.05)
    assert not result.empty
```

- [ ] **Step 2: Run to confirm tests fail**

```bash
python3 -m pytest tests/test_backtest.py::test_simulate_bets_returns_dataframe -v
```

Expected: `ImportError: cannot import name 'simulate_bets'`

- [ ] **Step 3: Extract `simulate_bets()` and refactor `run_backtest()`**

Replace the full contents of `backtest.py` with the following. Only the per-bet loop logic moves into `simulate_bets`; `run_backtest` becomes a thin wrapper that does the train/test split and calls it.

```python
"""Backtest the edge-finder against held-out test data using synthetic market odds."""
import json
import logging
from pathlib import Path
from typing import Any

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


def simulate_bets(
    clf: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    meta_df: pd.DataFrame,
    home_market_prob: float = 0.5,
    vig: float = 0.0476,
    unit: float = 100.0,
    ev_threshold: float | None = None,
) -> pd.DataFrame:
    """Simulate edge-finder bets on a pre-split test set.

    Runs the EV + Kelly bet-selection loop over every row in X_test/y_test.
    meta_df must be indexed identically to X_test and contain game_date,
    home_name, away_name.

    Args:
        clf: Fitted calibrated classifier. Must have feature_names_in_ and
            predict_proba() attributes.
        X_test: Feature matrix for the test games.
        y_test: True binary labels (1 = home win) for the test games.
        meta_df: DataFrame with game_date, home_name, away_name columns,
            same index as X_test.
        home_market_prob: Market-implied home win probability. Default 0.5.
        vig: Bookmaker overround. Default 0.0476.
        unit: Dollar bet size for P&L. Default $100.
        ev_threshold: Minimum EV to flag a bet. Defaults to config.EV_THRESHOLD.

    Returns:
        DataFrame sorted by game_date with columns: game_date, home_name,
        away_name, bet_side, american_odds, model_prob, ev, kelly_fraction,
        actual_home_win, won, pnl, cumulative_pnl. Returns empty DataFrame
        (with those columns) when no bets clear the thresholds.
    """
    from mlb_edge_finder import config as _config
    from mlb_edge_finder.edge_finder import compute_ev, compute_kelly, market_implied_prob

    _ev_threshold = ev_threshold if ev_threshold is not None else _config.EV_THRESHOLD

    output_cols = [
        "game_date", "home_name", "away_name", "bet_side",
        "american_odds", "model_prob", "ev", "kelly_fraction",
        "actual_home_win", "won", "pnl", "cumulative_pnl",
    ]

    feature_names = list(clf.feature_names_in_)
    X_test_aligned = X_test.reindex(columns=feature_names)
    home_probs = clf.predict_proba(X_test_aligned)[:, 1]

    home_odds_f, away_odds_f = simulate_market_odds(home_market_prob, vig)
    home_odds_i = round(home_odds_f)
    away_odds_i = round(away_odds_f)
    home_payout = home_odds_i / 100 if home_odds_i > 0 else 100 / abs(home_odds_i)
    away_payout = away_odds_i / 100 if away_odds_i > 0 else 100 / abs(away_odds_i)

    records = []
    for (idx, prob), actual in zip(zip(X_test.index, home_probs), y_test.values):
        row_meta = meta_df.loc[idx]

        home_ev = compute_ev(float(prob), home_odds_i)
        if home_ev > _ev_threshold and home_odds_i >= _config.MIN_AMERICAN_ODDS:
            won = int(actual) == 1
            records.append({
                "game_date": row_meta["game_date"],
                "home_name": row_meta["home_name"],
                "away_name": row_meta["away_name"],
                "bet_side": "home",
                "american_odds": home_odds_i,
                "model_prob": round(float(prob), 4),
                "ev": round(home_ev, 4),
                "kelly_fraction": round(compute_kelly(float(prob), home_odds_i), 4),
                "actual_home_win": int(actual),
                "won": won,
                "pnl": home_payout * unit if won else -unit,
            })

        away_prob = 1.0 - float(prob)
        away_ev = compute_ev(away_prob, away_odds_i)
        if away_ev > _ev_threshold and away_odds_i >= _config.MIN_AMERICAN_ODDS:
            won = int(actual) == 0
            records.append({
                "game_date": row_meta["game_date"],
                "home_name": row_meta["home_name"],
                "away_name": row_meta["away_name"],
                "bet_side": "away",
                "american_odds": away_odds_i,
                "model_prob": round(away_prob, 4),
                "ev": round(away_ev, 4),
                "kelly_fraction": round(compute_kelly(away_prob, away_odds_i), 4),
                "actual_home_win": int(actual),
                "won": won,
                "pnl": away_payout * unit if won else -unit,
            })

    if not records:
        logger.warning("No edges found in backtest at EV=%.0f%%", _ev_threshold * 100)
        return pd.DataFrame(columns=output_cols)

    result = pd.DataFrame(records).sort_values("game_date").reset_index(drop=True)
    result["cumulative_pnl"] = result["pnl"].cumsum()
    return result


def run_backtest(
    clf: Any,
    training_df: pd.DataFrame,
    home_market_prob: float = 0.5,
    vig: float = 0.0476,
    unit: float = 100.0,
    ev_threshold: float | None = None,
) -> pd.DataFrame:
    """Simulate edge-finder performance on the held-out 20% test split.

    Replicates the same 80/20 stratified split used in model._three_way_split()
    (test_size=0.2, random_state=42) so the evaluated games are identical to
    those used in model.evaluate(). No data leakage.

    Synthetic market odds are generated by simulate_market_odds(home_market_prob, vig).
    The default -110/-110 market assumes 50/50 game pricing.

    Args:
        clf: Fitted calibrated classifier from model.load_model(). Must have
            feature_names_in_ and predict_proba() attributes.
        training_df: Full training DataFrame from training_data.load_training_set().
            Must contain home_win, game_date, home_name, away_name, and all feature
            columns referenced by clf.feature_names_in_.
        home_market_prob: Market-implied home win probability. Default 0.5.
        vig: Bookmaker overround. Default 0.0476 (approx -110/-110 standard line).
        unit: Dollar bet size for P&L calculation. Default $100.
        ev_threshold: Minimum EV to flag a bet. Defaults to config.EV_THRESHOLD.

    Returns:
        DataFrame sorted by game_date with columns: game_date, home_name, away_name,
        bet_side, american_odds, model_prob, ev, kelly_fraction, actual_home_win,
        won, pnl, cumulative_pnl. Returns empty DataFrame (with those columns) if
        no bets clear the thresholds.
    """
    from sklearn.model_selection import train_test_split

    from mlb_edge_finder.model import NON_FEATURE_COLS, TARGET_COL

    non_feature = [c for c in NON_FEATURE_COLS if c in training_df.columns]
    X = training_df.drop(columns=non_feature)
    y = training_df[TARGET_COL]

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    meta = training_df.loc[X_test.index, ["game_date", "home_name", "away_name"]]

    result = simulate_bets(clf, X_test, y_test, meta, home_market_prob, vig, unit, ev_threshold)
    if not result.empty:
        logger.info(
            "Backtest complete: %d bets across %d test games (EV=%.0f%%)",
            len(result), len(X_test), (ev_threshold or 0.20) * 100,
        )
    return result


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


def sweep_thresholds(
    clf: Any,
    training_df: pd.DataFrame,
    ev_low: float = 0.05,
    ev_high: float = 0.50,
    ev_step: float = 0.05,
    unit: float = 100.0,
) -> pd.DataFrame:
    """Sweep ev_threshold values and rank by Sharpe ratio.

    Runs run_backtest() at each ev_threshold using the synthetic -110/-110 market.
    Combinations that produce 0 bets are excluded from results.

    Args:
        clf: Fitted calibrated classifier.
        training_df: Full training DataFrame from training_data.load_training_set().
        ev_low: Minimum EV threshold to sweep (inclusive). Default 0.05.
        ev_high: Maximum EV threshold to sweep (inclusive). Default 0.50.
        ev_step: Step size for EV threshold sweep. Default 0.05.
        unit: Dollar bet size passed to run_backtest(). Default $100.

    Returns:
        DataFrame sorted by sharpe_ratio descending with columns:
        ev_threshold, n_bets, win_rate, roi_pct, sharpe_ratio, avg_bets_per_day.

    Raises:
        RuntimeError: If every combination produces 0 bets (model is broken).
    """
    n_ev = round((ev_high - ev_low) / ev_step) + 1
    ev_values = [round(ev_low + i * ev_step, 4) for i in range(n_ev)]

    logger.info("Starting threshold sweep: %d EV values", len(ev_values))

    rows = []
    for i, ev_t in enumerate(ev_values):
        if i > 0 and i % 5 == 0:
            logger.info("Threshold sweep: %d/%d complete", i, len(ev_values))

        bt = run_backtest(clf, training_df, ev_threshold=ev_t, unit=unit)
        if bt.empty:
            logger.debug("Skipping EV=%.0f%% — no bets at this threshold", ev_t * 100)
            continue

        summary = compute_summary(bt, unit=unit)
        if summary["n_bets"] == 0:
            continue

        avg_bets_per_day = (
            summary["n_bets"] / bt["game_date"].nunique()
            if bt["game_date"].nunique() > 0 else 0.0
        )
        rows.append({
            "ev_threshold": ev_t,
            "n_bets": summary["n_bets"],
            "win_rate": summary["win_rate"],
            "roi_pct": summary["roi_pct"],
            "sharpe_ratio": summary["sharpe_ratio"],
            "avg_bets_per_day": round(avg_bets_per_day, 2),
        })

    if not rows:
        raise RuntimeError("Threshold sweep produced no valid combinations — model may be broken")

    result = pd.DataFrame(rows).sort_values("sharpe_ratio", ascending=False).reset_index(drop=True)
    best = result.iloc[0]
    logger.info(
        "Optimal: EV=%.0f%% Sharpe=%.3f (%d bets, %.1f/day)",
        best["ev_threshold"] * 100,
        best["sharpe_ratio"],
        best["n_bets"],
        best["avg_bets_per_day"],
    )
    return result


def export_pnl_json(backtest_df: pd.DataFrame, summary: dict, path: Path) -> None:
    """Export cumulative P&L curve and summary stats to a JSON file.

    Args:
        backtest_df: Output of run_backtest(). Must have cumulative_pnl column.
        summary: Output of compute_summary().
        path: Destination path for the JSON file. Parent dirs created if needed.
    """
    data = {
        "cumulative_pnl": [round(v, 2) for v in backtest_df["cumulative_pnl"].tolist()],
        "summary": summary,
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2))
    logger.info("Exported P&L JSON to %s (%d bets)", path, len(backtest_df))
```

- [ ] **Step 4: Run all backtest tests to confirm nothing regressed**

```bash
python3 -m pytest tests/test_backtest.py -v
```

Expected: all previously passing tests still pass, plus the 4 new `simulate_bets` tests.

- [ ] **Step 5: Commit**

```bash
git add src/mlb_edge_finder/backtest.py tests/test_backtest.py
git commit -m "refactor: extract simulate_bets() from run_backtest() for reuse in temporal eval"
```

---

## Task 2: Create `temporal_eval.py`

**Files:**
- Create: `src/mlb_edge_finder/temporal_eval.py`
- Create: `tests/test_temporal_eval.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_temporal_eval.py`:

```python
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


def _make_training_df(n_per_season: int = 60) -> pd.DataFrame:
    """Minimal training DataFrame with two seasons (2023 and 2025)."""
    rng = np.random.default_rng(42)
    n = n_per_season * 2
    seasons = [2023] * n_per_season + [2025] * n_per_season
    return pd.DataFrame({
        "season": seasons,
        "game_date": ["2023-04-01"] * n_per_season + ["2025-04-01"] * n_per_season,
        "home_name": "TeamA",
        "away_name": "TeamB",
        "home_score": rng.integers(0, 10, n),
        "away_score": rng.integers(0, 10, n),
        "home_abbr": "TA",
        "away_abbr": "TB",
        "home_win": rng.integers(0, 2, n),
        "home_starter_name": None,
        "away_starter_name": None,
        "home_pitcher_id": None,
        "away_pitcher_id": None,
        "feature_a": rng.random(n),
        "feature_b": rng.random(n),
    })


def _make_mock_clf(n_features: int = 2) -> MagicMock:
    clf = MagicMock()
    clf.feature_names_in_ = np.array([f"feature_{chr(97+i)}" for i in range(n_features)])
    clf.predict_proba = MagicMock(
        side_effect=lambda X: np.column_stack([np.full(len(X), 0.35), np.full(len(X), 0.65)])
    )
    clf.predict = MagicMock(return_value=np.ones(60, dtype=int))
    return clf


# --- _temporal_split ---

def test_temporal_split_train_has_no_holdout_season():
    from mlb_edge_finder.temporal_eval import _temporal_split
    df = _make_training_df()
    train_df, _ = _temporal_split(df, holdout_season=2025)
    assert (train_df["season"] < 2025).all()


def test_temporal_split_test_is_only_holdout_season():
    from mlb_edge_finder.temporal_eval import _temporal_split
    df = _make_training_df()
    _, test_df = _temporal_split(df, holdout_season=2025)
    assert (test_df["season"] == 2025).all()


def test_temporal_split_raises_if_no_train():
    from mlb_edge_finder.temporal_eval import _temporal_split
    df = pd.DataFrame({"season": [2025] * 10, "home_win": [1] * 10, "f": [0.5] * 10})
    with pytest.raises(RuntimeError, match="No training data"):
        _temporal_split(df, holdout_season=2025)


def test_temporal_split_raises_if_no_test():
    from mlb_edge_finder.temporal_eval import _temporal_split
    df = pd.DataFrame({"season": [2023] * 10, "home_win": [1] * 10, "f": [0.5] * 10})
    with pytest.raises(RuntimeError, match="No test data"):
        _temporal_split(df, holdout_season=2025)


# --- run() ---

def _run_with_mocks(tmp_path: Path, training_df: pd.DataFrame, force: bool = False) -> dict:
    """Call temporal_eval.run() with all expensive operations mocked.

    Patches are on mlb_edge_finder.temporal_eval.* because calibrate, evaluate,
    simulate_bets, compute_summary, and config are all module-level imports there.
    """
    import mlb_edge_finder.temporal_eval as te
    mock_clf = _make_mock_clf()
    empty_backtest = pd.DataFrame(columns=[
        "game_date", "home_name", "away_name", "bet_side", "american_odds",
        "model_prob", "ev", "kelly_fraction", "actual_home_win", "won", "pnl", "cumulative_pnl",
    ])

    with patch.object(te, "_load_training_csv", return_value=training_df), \
         patch("mlb_edge_finder.temporal_eval.XGBClassifier") as MockXGB, \
         patch("mlb_edge_finder.temporal_eval.calibrate", return_value=mock_clf), \
         patch("mlb_edge_finder.temporal_eval.evaluate", return_value={
             "accuracy": 0.57, "roc_auc": 0.60, "log_loss": 0.68, "brier_score": 0.24,
             "n_test_samples": 60,
         }), \
         patch("mlb_edge_finder.temporal_eval.simulate_bets", return_value=empty_backtest), \
         patch("mlb_edge_finder.temporal_eval.compute_summary", return_value={
             "n_bets": 0, "n_wins": 0, "win_rate": 0.0, "total_pnl": 0.0,
             "roi_pct": 0.0, "avg_ev": 0.0, "max_drawdown": 0.0, "sharpe_ratio": 0.0,
         }), \
         patch("mlb_edge_finder.temporal_eval.config") as mock_config:
        mock_config.MODELS_DIR = tmp_path
        mock_config.DATA_PROCESSED_DIR = tmp_path
        mock_config.XGB_N_ESTIMATORS = 10
        mock_config.XGB_MAX_DEPTH = 3
        mock_xgb_instance = MagicMock()
        mock_xgb_instance.feature_names_in_ = np.array(["feature_a", "feature_b"])
        MockXGB.return_value = mock_xgb_instance
        result = te.run(holdout_season=2025, force=force)
    return result


def test_run_writes_json(tmp_path):
    df = _make_training_df()
    _run_with_mocks(tmp_path, df)
    assert (tmp_path / "temporal_eval_2025.json").exists()


def test_run_json_has_required_keys(tmp_path):
    df = _make_training_df()
    result = _run_with_mocks(tmp_path, df)
    required = {
        "holdout_season", "train_seasons", "n_train", "n_test",
        "accuracy", "roc_auc", "log_loss", "brier_score",
        "n_bets", "win_rate", "roi_pct", "sharpe_ratio",
        "total_pnl", "avg_ev", "max_drawdown", "pnl_series",
    }
    assert required.issubset(set(result.keys()))


def test_run_pnl_series_is_list(tmp_path):
    df = _make_training_df()
    result = _run_with_mocks(tmp_path, df)
    assert isinstance(result["pnl_series"], list)


def test_run_skips_if_exists(tmp_path):
    existing = {"holdout_season": 2025, "roc_auc": 0.999, "pnl_series": []}
    (tmp_path / "temporal_eval_2025.json").write_text(json.dumps(existing))
    import mlb_edge_finder.temporal_eval as te
    with patch("mlb_edge_finder.temporal_eval.config") as mock_config:
        mock_config.MODELS_DIR = tmp_path
        result = te.run(holdout_season=2025, force=False)
    assert result["roc_auc"] == 0.999


def test_run_force_overwrites(tmp_path):
    existing = {"holdout_season": 2025, "roc_auc": 0.999, "pnl_series": []}
    (tmp_path / "temporal_eval_2025.json").write_text(json.dumps(existing))
    df = _make_training_df()
    result = _run_with_mocks(tmp_path, df, force=True)
    assert result["roc_auc"] != 0.999


def test_run_raises_if_no_training_csv(tmp_path):
    import mlb_edge_finder.temporal_eval as te
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    with patch("mlb_edge_finder.temporal_eval.config") as mock_config:
        mock_config.MODELS_DIR = tmp_path
        mock_config.DATA_PROCESSED_DIR = empty_dir  # no training_*.csv files
        with pytest.raises(RuntimeError, match="No training set found"):
            te.run(holdout_season=2025)
```

- [ ] **Step 2: Run to confirm tests fail**

```bash
python3 -m pytest tests/test_temporal_eval.py -v
```

Expected: `ModuleNotFoundError: No module named 'mlb_edge_finder.temporal_eval'`

- [ ] **Step 3: Create `src/mlb_edge_finder/temporal_eval.py`**

```python
"""Temporal out-of-time model evaluation: train on prior seasons, test on holdout."""
import json
import logging
from typing import Any

import pandas as pd
from xgboost import XGBClassifier

from mlb_edge_finder import config
from mlb_edge_finder.backtest import compute_summary, simulate_bets
from mlb_edge_finder.model import NON_FEATURE_COLS, TARGET_COL, calibrate, evaluate

logger = logging.getLogger(__name__)


def _temporal_split(
    training_df: pd.DataFrame,
    holdout_season: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split training_df by season column.

    Returns (train_df, test_df) where train contains seasons strictly before
    holdout_season and test contains only holdout_season rows.
    """
    train_df = training_df[training_df["season"] < holdout_season].copy()
    test_df = training_df[training_df["season"] == holdout_season].copy()
    if train_df.empty:
        raise RuntimeError(
            f"No training data found before season {holdout_season} — "
            "check that the training set covers multiple seasons"
        )
    if test_df.empty:
        raise RuntimeError(
            f"No test data found for season {holdout_season} — "
            "check that the training set includes this season"
        )
    return train_df, test_df


def _load_training_csv(data_processed_dir: Any) -> pd.DataFrame:
    """Load the most recently built training CSV from data_processed_dir."""
    from pathlib import Path
    csvs = sorted(Path(data_processed_dir).glob("training_*.csv"))
    if not csvs:
        raise RuntimeError(
            "No training set found — run build_training_set() first"
        )
    return pd.read_csv(csvs[-1])


def run(holdout_season: int = 2025, force: bool = False) -> dict:
    """Train on seasons before holdout_season, evaluate on holdout_season.

    Writes a JSON artifact to MODELS_DIR/temporal_eval_{holdout_season}.json
    containing model metrics + backtest summary + per-bet P&L series.

    Args:
        holdout_season: The season to hold out for testing. Default 2025.
        force: If True, overwrite any existing artifact. Default False.

    Returns:
        The written dict.

    Raises:
        RuntimeError: If no training set CSV is found, or if either the
            train or test slice is empty.
    """
    from sklearn.model_selection import train_test_split

    out_path = config.MODELS_DIR / f"temporal_eval_{holdout_season}.json"
    if out_path.exists() and not force:
        logger.info(
            "Temporal eval already at %s — skipping (use force=True to rerun)", out_path
        )
        return json.loads(out_path.read_text())

    training_df = _load_training_csv(config.DATA_PROCESSED_DIR)
    logger.info("Loaded training set (%d rows)", len(training_df))

    train_df, test_df = _temporal_split(training_df, holdout_season)
    train_seasons = sorted(int(s) for s in train_df["season"].unique())
    logger.info(
        "Temporal split: %d train rows (seasons %s), %d test rows (season %d)",
        len(train_df), train_seasons, len(test_df), holdout_season,
    )

    non_feature = [c for c in NON_FEATURE_COLS if c in train_df.columns]
    X_train_full = train_df.drop(columns=non_feature)
    y_train_full = train_df[TARGET_COL]

    X_test = test_df.drop(columns=non_feature).reindex(columns=X_train_full.columns)
    y_test = test_df[TARGET_COL]

    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.25, random_state=42, stratify=y_train_full
    )

    clf = XGBClassifier(
        n_estimators=config.XGB_N_ESTIMATORS,
        max_depth=config.XGB_MAX_DEPTH,
        eval_metric="logloss",
        random_state=42,
    )
    clf.fit(X_fit, y_fit)
    cal_clf = calibrate(clf, X_val, y_val)
    logger.info("Model trained and calibrated (%d fit, %d val rows)", len(X_fit), len(X_val))

    metrics = evaluate(cal_clf, X_test, y_test)
    logger.info(
        "Holdout metrics: accuracy=%.3f roc_auc=%.3f",
        metrics["accuracy"], metrics["roc_auc"],
    )

    meta_df = test_df[["game_date", "home_name", "away_name"]]
    backtest_df = simulate_bets(cal_clf, X_test, y_test, meta_df)
    summary = compute_summary(backtest_df)

    pnl_series = (
        [
            {"date": str(r["game_date"]), "cumulative_pnl": round(r["cumulative_pnl"], 2)}
            for _, r in backtest_df.iterrows()
        ]
        if not backtest_df.empty
        else []
    )

    result = {
        "holdout_season": holdout_season,
        "train_seasons": train_seasons,
        "n_train": len(train_df),
        "n_test": len(test_df),
        "accuracy": round(metrics["accuracy"], 4),
        "roc_auc": round(metrics["roc_auc"], 4),
        "log_loss": round(metrics["log_loss"], 4),
        "brier_score": round(metrics["brier_score"], 4),
        "n_bets": summary["n_bets"],
        "win_rate": summary["win_rate"],
        "roi_pct": summary["roi_pct"],
        "sharpe_ratio": summary["sharpe_ratio"],
        "total_pnl": summary["total_pnl"],
        "avg_ev": summary["avg_ev"],
        "max_drawdown": summary["max_drawdown"],
        "pnl_series": pnl_series,
    }

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    logger.info(
        "Temporal eval written to %s (ROC-AUC=%.3f, ROI=%.1f%%)",
        out_path, result["roc_auc"], result["roi_pct"],
    )
    return result


if __name__ == "__main__":
    import argparse

    from mlb_edge_finder.config import setup_logging

    setup_logging()
    parser = argparse.ArgumentParser(description="Run temporal out-of-time evaluation")
    parser.add_argument("--holdout-season", type=int, default=2025)
    parser.add_argument("--force", action="store_true", help="Overwrite existing artifact")
    args = parser.parse_args()

    r = run(holdout_season=args.holdout_season, force=args.force)
    print(f"\nTemporal Eval — Holdout Season: {r['holdout_season']}")
    print(f"  Train seasons : {r['train_seasons']}")
    print(f"  Train rows    : {r['n_train']:,}")
    print(f"  Test rows     : {r['n_test']:,}")
    print(f"  ROC-AUC       : {r['roc_auc']:.3f}")
    print(f"  Accuracy      : {r['accuracy']:.3f}")
    print(f"  Bets          : {r['n_bets']:,}")
    print(f"  Win Rate      : {r['win_rate'] * 100:.1f}%")
    print(f"  ROI           : {r['roi_pct']:+.1f}%")
    print(f"  Sharpe        : {r['sharpe_ratio']:.3f}")
    print(f"\nArtifact: models/temporal_eval_{r['holdout_season']}.json")
```

- [ ] **Step 4: Run temporal_eval tests**

```bash
python3 -m pytest tests/test_temporal_eval.py -v
```

Expected: all 9 tests pass.

- [ ] **Step 5: Run full test suite to check for regressions**

```bash
python3 -m pytest tests/ -v --tb=short
```

Expected: all previously passing tests still pass.

- [ ] **Step 6: Commit**

```bash
git add src/mlb_edge_finder/temporal_eval.py tests/test_temporal_eval.py
git commit -m "feat: add temporal_eval module — train on 2019-2024, test on 2025 holdout"
```

---

## Task 3: Update `generate_site.py`

**Files:**
- Modify: `src/mlb_edge_finder/generate_site.py`
- Modify: `tests/test_generate_site.py`

- [ ] **Step 1: Update `tests/test_generate_site.py`**

Replace the full file contents with the following. The `_load_metrics` / `_load_pnl` tests are replaced by `_load_temporal_eval` tests. All `generate()` call sites are updated to the new `(outputs_dir, models_dir, out_path)` signature.

```python
import json
import csv
from datetime import date
from pathlib import Path

import pytest


def _write_edges_csv(path: Path, rows: list[dict]) -> None:
    cols = ["game_id", "home_team", "away_team", "bet_side",
            "american_odds", "model_prob", "ev", "kelly_fraction", "high_confidence"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def _write_header_only_csv(path: Path) -> None:
    path.write_text(
        "game_id,home_team,away_team,bet_side,american_odds,model_prob,ev,kelly_fraction,high_confidence\n"
    )


def _write_temporal_eval_json(path: Path, **overrides) -> dict:
    data = {
        "holdout_season": 2025,
        "train_seasons": [2019, 2021, 2022, 2023, 2024],
        "n_train": 12000,
        "n_test": 2400,
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
        "max_drawdown": 420.0,
        "pnl_series": [
            {"date": "2025-04-01", "cumulative_pnl": 95.45},
            {"date": "2025-04-02", "cumulative_pnl": 185.90},
        ],
    }
    data.update(overrides)
    path.write_text(json.dumps(data))
    return data


# --- _load_edges_data ---

def test_load_edges_data_today_rows(tmp_path):
    from mlb_edge_finder.generate_site import _load_edges_data
    today = date.today().isoformat()
    _write_edges_csv(tmp_path / f"edges_{today}.csv", [
        {"game_id": "abc", "home_team": "Giants", "away_team": "Dodgers",
         "bet_side": "home", "american_odds": -110, "model_prob": 0.7,
         "ev": 0.55, "kelly_fraction": 0.25, "high_confidence": False},
    ])
    today_rows, history = _load_edges_data(tmp_path)
    assert len(today_rows) == 1
    assert today_rows[0]["home_team"] == "Giants"


def test_load_edges_data_history_count(tmp_path):
    from mlb_edge_finder.generate_site import _load_edges_data
    _write_edges_csv(tmp_path / "edges_2026-05-20.csv", [
        {"game_id": "a", "home_team": "X", "away_team": "Y", "bet_side": "home",
         "american_odds": -110, "model_prob": 0.6, "ev": 0.5, "kelly_fraction": 0.2, "high_confidence": False},
        {"game_id": "b", "home_team": "X", "away_team": "Z", "bet_side": "away",
         "american_odds": 120, "model_prob": 0.6, "ev": 0.5, "kelly_fraction": 0.2, "high_confidence": False},
    ])
    _write_header_only_csv(tmp_path / "edges_2026-05-21.csv")
    _, history = _load_edges_data(tmp_path)
    counts = {h["date"]: h["count"] for h in history}
    assert counts["2026-05-20"] == 2
    assert counts["2026-05-21"] == 0


def test_load_edges_data_empty_outputs_dir(tmp_path):
    from mlb_edge_finder.generate_site import _load_edges_data
    today_rows, history = _load_edges_data(tmp_path)
    assert today_rows == []
    assert history == []


def test_load_edges_data_caps_at_30_days(tmp_path):
    from mlb_edge_finder.generate_site import _load_edges_data
    for i in range(35):
        _write_header_only_csv(tmp_path / f"edges_2026-04-{i+1:02d}.csv")
    _, history = _load_edges_data(tmp_path)
    assert len(history) <= 30


# --- _load_temporal_eval ---

def test_load_temporal_eval_returns_dict(tmp_path):
    from mlb_edge_finder.generate_site import _load_temporal_eval
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    _write_temporal_eval_json(models_dir / "temporal_eval_2025.json")
    result = _load_temporal_eval(models_dir)
    assert result is not None
    assert result["roc_auc"] == 0.601


def test_load_temporal_eval_returns_none_when_missing(tmp_path):
    from mlb_edge_finder.generate_site import _load_temporal_eval
    assert _load_temporal_eval(tmp_path) is None


def test_load_temporal_eval_picks_most_recent(tmp_path):
    from mlb_edge_finder.generate_site import _load_temporal_eval
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    _write_temporal_eval_json(models_dir / "temporal_eval_2024.json", roc_auc=0.55)
    _write_temporal_eval_json(models_dir / "temporal_eval_2025.json", roc_auc=0.601)
    result = _load_temporal_eval(models_dir)
    assert result["roc_auc"] == 0.601


# --- generate() integration ---

def test_generate_creates_index_html(tmp_path):
    from mlb_edge_finder.generate_site import generate
    today = date.today().isoformat()
    outputs_dir = tmp_path / "outputs"
    outputs_dir.mkdir()
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    _write_edges_csv(outputs_dir / f"edges_{today}.csv", [
        {"game_id": "abc", "home_team": "Giants", "away_team": "Dodgers",
         "bet_side": "home", "american_odds": -110, "model_prob": 0.7,
         "ev": 0.55, "kelly_fraction": 0.25, "high_confidence": False},
    ])
    out = tmp_path / "docs" / "index.html"
    generate(outputs_dir=outputs_dir, models_dir=models_dir, out_path=out)
    assert out.exists()
    html = out.read_text()
    assert "Giants" in html
    assert "MLB Edge Finder" in html


def test_generate_empty_state_when_no_edges_today(tmp_path):
    from mlb_edge_finder.generate_site import generate
    outputs_dir = tmp_path / "outputs"
    outputs_dir.mkdir()
    today = date.today().isoformat()
    _write_header_only_csv(outputs_dir / f"edges_{today}.csv")
    out = tmp_path / "docs" / "index.html"
    generate(outputs_dir=outputs_dir, models_dir=tmp_path, out_path=out)
    html = out.read_text()
    assert "No edges found today" in html


def test_generate_includes_stats_when_temporal_eval_present(tmp_path):
    from mlb_edge_finder.generate_site import generate
    outputs_dir = tmp_path / "outputs"
    outputs_dir.mkdir()
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    _write_temporal_eval_json(models_dir / "temporal_eval_2025.json")
    out = tmp_path / "docs" / "index.html"
    generate(outputs_dir=outputs_dir, models_dir=models_dir, out_path=out)
    html = out.read_text()
    assert "60.3%" in html    # win_rate
    assert "15.1%" in html    # roi_pct
    assert "0.601" in html    # roc_auc
    assert "2025 holdout" in html


def test_generate_still_works_when_temporal_eval_missing(tmp_path):
    from mlb_edge_finder.generate_site import generate
    outputs_dir = tmp_path / "outputs"
    outputs_dir.mkdir()
    out = tmp_path / "docs" / "index.html"
    generate(outputs_dir=outputs_dir, models_dir=tmp_path, out_path=out)
    assert out.exists()
    html = out.read_text()
    assert "<!DOCTYPE html>" in html


def test_generate_high_confidence_shows_star_badge(tmp_path):
    from mlb_edge_finder.generate_site import generate
    today = date.today().isoformat()
    outputs_dir = tmp_path / "outputs"
    outputs_dir.mkdir()
    _write_edges_csv(outputs_dir / f"edges_{today}.csv", [
        {"game_id": "x", "home_team": "A", "away_team": "B",
         "bet_side": "away", "american_odds": 150, "model_prob": 0.75,
         "ev": 0.6, "kelly_fraction": 0.3, "high_confidence": True},
    ])
    out = tmp_path / "docs" / "index.html"
    generate(outputs_dir=outputs_dir, models_dir=tmp_path, out_path=out)
    html = out.read_text()
    assert "★" in html
    assert "⚠" not in html
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
python3 -m pytest tests/test_generate_site.py -v
```

Expected: `_load_temporal_eval` tests fail with `ImportError`; `generate()` tests fail with wrong number of arguments.

- [ ] **Step 3: Replace `generate_site.py` with the updated version**

Replace the full file contents:

```python
"""Generate the static GitHub Pages dashboard at docs/index.html."""
import json
import logging
from datetime import date
from html import escape
from pathlib import Path

import pandas as pd

from mlb_edge_finder.stats_ingestion import ODDS_NAME_TO_ABBR

logger = logging.getLogger(__name__)


def _team_abbr(full_name: str) -> str:
    """Return the 3-letter abbreviation for a full Odds API team name."""
    return ODDS_NAME_TO_ABBR.get(full_name, full_name)


_ROOT = Path(__file__).resolve().parents[2]
OUTPUTS_DIR: Path = _ROOT / "outputs"
DOCS_DIR: Path = _ROOT / "docs"


def _load_edges_data(outputs_dir: Path) -> tuple[list[dict], list[dict]]:
    """Load today's edges and per-day edge counts from outputs/ CSVs.

    Returns:
        (today_rows, history) where history is a list of {date, count} dicts
        covering the last 30 available days sorted oldest-first.
    """
    if not outputs_dir.exists():
        return [], []
    today = date.today().isoformat()
    csv_files = sorted(outputs_dir.glob("edges_*.csv"))[-30:]

    history: list[dict] = []
    today_rows: list[dict] = []

    for csv_path in csv_files:
        file_date = csv_path.stem[len("edges_"):]
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            logger.warning("Skipping malformed CSV: %s", csv_path)
            continue
        history.append({"date": file_date, "count": len(df)})
        if file_date == today:
            today_rows = df.to_dict(orient="records")

    return today_rows, history


def _load_temporal_eval(models_dir: Path) -> dict | None:
    """Load the most recent temporal_eval_*.json from models_dir, or None."""
    files = sorted(Path(models_dir).glob("temporal_eval_*.json"))
    if not files:
        return None
    return json.loads(files[-1].read_text())


def _render_stats_html(te_data: dict | None) -> str:
    """Render the backtest performance stats card, or '' if no data."""
    if not te_data:
        return ""
    rows = []
    if "win_rate" in te_data:
        rows.append(
            f'<div class="stat-row"><span class="stat-label">Win Rate</span>'
            f'<span class="stat-value">{te_data["win_rate"] * 100:.1f}%</span></div>'
        )
    if "roi_pct" in te_data:
        roi = te_data["roi_pct"]
        roi_prefix = "+" if roi >= 0 else ""
        roi_class = "stat-value green" if roi >= 0 else "stat-value"
        rows.append(
            f'<div class="stat-row"><span class="stat-label">Backtest ROI</span>'
            f'<span class="{roi_class}">{roi_prefix}{roi:.1f}%</span></div>'
        )
    if "sharpe_ratio" in te_data:
        rows.append(
            f'<div class="stat-row"><span class="stat-label">Sharpe</span>'
            f'<span class="stat-value">{te_data["sharpe_ratio"]:.3f}</span></div>'
        )
    if "roc_auc" in te_data:
        rows.append(
            f'<div class="stat-row"><span class="stat-label">ROC-AUC</span>'
            f'<span class="stat-value neutral">{te_data["roc_auc"]:.3f}</span></div>'
        )
    if "n_test" in te_data:
        rows.append(
            f'<div class="stat-row"><span class="stat-label">Holdout games</span>'
            f'<span class="stat-value neutral">{te_data["n_test"]:,}</span></div>'
        )
    if not rows:
        return ""
    train_seasons = te_data.get("train_seasons", [])
    holdout = te_data.get("holdout_season", "")
    subtitle = ""
    if train_seasons and holdout:
        subtitle = (
            f'<div class="card-subtitle">Trained {train_seasons[0]}&ndash;'
            f'{train_seasons[-1]} &middot; {holdout} holdout</div>'
        )
    return (
        '<div class="card"><div class="card-title">Backtest Performance</div>'
        + subtitle
        + "".join(rows)
        + "</div>"
    )


def _render_pnl_html(te_data: dict | None) -> str:
    """Render the P&L chart card, or '' if no data."""
    if te_data is None:
        return ""
    return (
        '<div class="card">'
        '<div class="card-title">Backtest P&amp;L Curve</div>'
        '<div class="chart-wrap-sm"><canvas id="pnl-chart"></canvas></div>'
        "</div>"
    )


def _render_edges_html(today_rows: list[dict]) -> str:
    """Render the today's edges table as static HTML."""
    if not today_rows:
        return (
            '<div class="empty-state">No edges found today &mdash; model found no '
            "+EV opportunities meeting the current thresholds.</div>"
        )
    rows_html = ""
    for r in today_rows:
        home = escape(str(r.get("home_team", "")))
        away = escape(str(r.get("away_team", "")))
        side = escape(str(r.get("bet_side", "")))
        sc = "side-home" if r.get("bet_side") == "home" else "side-away"
        raw_home = str(r.get("home_team", ""))
        raw_away = str(r.get("away_team", ""))
        bet_team = escape(_team_abbr(raw_home if r.get("bet_side") == "home" else raw_away))
        high_conf = r.get("high_confidence")
        is_high_conf = high_conf is True or str(high_conf).strip() == "True"
        star = "★ " if is_high_conf else ""
        odds_int = int(r.get("american_odds", 0) or 0)
        odds_str = f"+{odds_int}" if odds_int > 0 else str(odds_int)
        model_prob = float(r.get("model_prob", 0))
        ev = float(r.get("ev", 0))
        ev_str = f"+{ev * 100:.1f}%" if ev >= 0 else f"{ev * 100:.1f}%"
        kelly = float(r.get("kelly_fraction", 0))
        rows_html += (
            f"<tr>"
            f"<td>{home} vs {away}</td>"
            f'<td><span class="side-badge {sc}">{star}{bet_team}</span></td>'
            f"<td>{odds_str}</td>"
            f"<td>{model_prob * 100:.1f}%</td>"
            f'<td class="ev-val">{ev_str}</td>'
            f"<td>{kelly * 100:.1f}%</td>"
            f"</tr>"
        )
    return (
        "<table><thead><tr>"
        "<th>Matchup</th><th>Bet On</th><th>Odds</th>"
        "<th>Model Prob</th><th>EV</th><th>Kelly</th>"
        f"</tr></thead><tbody>{rows_html}</tbody></table>"
    )


def _render_html(
    today_rows: list[dict],
    history: list[dict],
    te_data: dict | None,
    updated: str,
) -> str:
    """Return the complete HTML page as a string."""
    edges_table_html = _render_edges_html(today_rows)
    history_json = json.dumps(history)
    te_json = json.dumps(te_data) if te_data else "null"
    stats_html = _render_stats_html(te_data)
    pnl_chart_html = _render_pnl_html(te_data)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>MLB Edge Finder — Daily Dashboard</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
  <style>
    *{{margin:0;padding:0;box-sizing:border-box}}
    body{{background:#1a1713;color:#d6cfc4;font-family:-apple-system,'Segoe UI',sans-serif;min-height:100vh}}
    a{{color:#FD5A1E}}
    .page{{max-width:1100px;margin:0 auto;padding:24px 20px}}
    .header{{border-bottom:1px solid #3d3930;padding-bottom:16px;margin-bottom:24px;display:flex;align-items:flex-start;justify-content:space-between;flex-wrap:wrap;gap:8px}}
    .header-title{{font-size:22px;font-weight:800;letter-spacing:-0.02em;color:#fff}}
    .header-sub{{font-size:13px;color:#8a8070;margin-top:4px}}
    .badge{{font-size:11px;font-weight:700;background:#FD5A1E22;color:#FD5A1E;border:1px solid #FD5A1E44;padding:3px 10px;border-radius:12px;white-space:nowrap;margin-top:4px}}
    .main-layout{{display:flex;gap:24px;align-items:flex-start}}
    .col-main{{flex:1;min-width:0}}
    .col-sidebar{{width:240px;flex-shrink:0;display:flex;flex-direction:column;gap:16px}}
    .section{{margin-bottom:28px}}
    .section-title{{font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;color:#8a8070;margin-bottom:12px}}
    .table-wrap{{overflow-x:auto;border-radius:8px;border:1px solid #3d3930}}
    table{{width:100%;border-collapse:collapse;font-size:13px}}
    thead tr{{background:#27251F}}
    th{{padding:10px 12px;text-align:left;font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:0.07em;color:#8a8070;white-space:nowrap}}
    td{{padding:10px 12px;border-bottom:1px solid #2e2b25;color:#d6cfc4;white-space:nowrap}}
    tbody tr:last-child td{{border-bottom:none}}
    tbody tr:hover td{{background:#27251F55}}
    .ev-val{{color:#FD5A1E;font-weight:700}}
    .side-badge{{display:inline-block;font-size:11px;font-weight:600;padding:2px 8px;border-radius:4px;white-space:nowrap}}
    .side-home{{background:#FD5A1E22;color:#FD5A1E;border:1px solid #FD5A1E44}}
    .side-away{{background:#ffffff11;color:#8a8070;border:1px solid #3d3930}}
    .empty-state{{text-align:center;padding:32px;color:#8a8070;font-size:14px;line-height:1.6}}
    .card{{background:#27251F;border:1px solid #3d3930;border-radius:8px;padding:16px}}
    .card-title{{font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;color:#8a8070;margin-bottom:4px}}
    .card-subtitle{{font-size:10px;color:#8a8070;margin-bottom:10px;opacity:0.75}}
    .stat-row{{display:flex;justify-content:space-between;align-items:baseline;margin-bottom:8px}}
    .stat-row:last-child{{margin-bottom:0}}
    .stat-label{{font-size:12px;color:#8a8070}}
    .stat-value{{font-size:14px;font-weight:700;color:#FD5A1E}}
    .stat-value.green{{color:#4ade80}}
    .stat-value.neutral{{color:#d6cfc4}}
    .chart-wrap{{position:relative;height:120px}}
    .chart-wrap-sm{{position:relative;height:100px}}
    .updated{{font-size:11px;color:#8a8070;text-align:right}}
    @media(max-width:700px){{.main-layout{{flex-direction:column}}.col-sidebar{{width:100%}}}}
  </style>
</head>
<body>
<div class="page">
  <div class="header">
    <div>
      <div class="header-title">&#9918; MLB Edge Finder</div>
      <div class="header-sub">XGBoost model identifying positive expected-value MLB moneyline bets</div>
    </div>
    <span class="badge">Updated {updated}</span>
  </div>
  <div class="main-layout">
    <div class="col-main">
      <div class="section">
        <div class="section-title">Today&#39;s Edges</div>
        <div class="table-wrap" id="edges-table">{edges_table_html}</div>
      </div>
      <div class="section">
        <div class="section-title">Edge History &mdash; Last 30 Days</div>
        <div class="chart-wrap"><canvas id="history-chart"></canvas></div>
      </div>
    </div>
    <div class="col-sidebar">
      <div class="updated">Updated {updated}</div>
      {stats_html}
      {pnl_chart_html}
    </div>
  </div>
</div>
<script>
const HISTORY={history_json};
const TE={te_json};
</script>
<script>
(function(){{
  if(!HISTORY||HISTORY.length===0)return;
  var ctx=document.getElementById('history-chart').getContext('2d');
  new Chart(ctx,{{type:'bar',data:{{labels:HISTORY.map(function(d){{return d.date.slice(5)}}),datasets:[{{data:HISTORY.map(function(d){{return d.count}}),backgroundColor:'#FD5A1E99',borderColor:'#FD5A1E',borderWidth:1,borderRadius:2}}]}},options:{{responsive:true,maintainAspectRatio:false,plugins:{{legend:{{display:false}},tooltip:{{callbacks:{{title:function(t){{return HISTORY[t[0].dataIndex].date}},label:function(t){{return t.raw+' edge'+(t.raw!==1?'s':'')}}}}}}}},scales:{{x:{{grid:{{color:'#3d393044'}},ticks:{{color:'#8a8070',font:{{size:10}}}}}},y:{{grid:{{color:'#3d393044'}},ticks:{{color:'#8a8070',font:{{size:10}},stepSize:1}},beginAtZero:true}}}}}}}});
}})();
(function(){{
  var el=document.getElementById('pnl-chart');
  if(!el||!TE||!TE.pnl_series||TE.pnl_series.length===0)return;
  var ctx=el.getContext('2d');
  new Chart(ctx,{{type:'line',data:{{labels:TE.pnl_series.map(function(d){{return d.date.slice(5)}}),datasets:[{{data:TE.pnl_series.map(function(d){{return d.cumulative_pnl}}),borderColor:'#FD5A1E',borderWidth:2,pointRadius:0,tension:0.1,fill:false}}]}},options:{{responsive:true,maintainAspectRatio:false,plugins:{{legend:{{display:false}},tooltip:{{callbacks:{{label:function(t){{return'$'+t.raw.toFixed(0)}},title:function(t){{return TE.pnl_series[t[0].dataIndex].date}}}}}}}},scales:{{x:{{display:false}},y:{{grid:{{color:'#3d393044'}},ticks:{{color:'#8a8070',font:{{size:10}},callback:function(v){{return'$'+v}}}}}}}}}}}});
}})();
</script>
</body>
</html>"""


def generate(
    outputs_dir: Path,
    models_dir: Path,
    out_path: Path,
) -> None:
    """Generate docs/index.html from outputs CSVs and temporal eval artifact.

    Never raises — degrades gracefully if the temporal eval file is missing.

    Args:
        outputs_dir: Directory containing edges_YYYY-MM-DD.csv files.
        models_dir: Directory containing temporal_eval_*.json files.
        out_path: Destination for the generated index.html.
    """
    today_rows, history = _load_edges_data(outputs_dir)
    te_data = _load_temporal_eval(models_dir)

    updated = date.today().strftime("%B %d, %Y").replace(" 0", " ")
    html = _render_html(today_rows, history, te_data, updated)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    logger.info(
        "Dashboard written to %s (%d edges today, %d history days)",
        out_path, len(today_rows), len(history),
    )


if __name__ == "__main__":
    from mlb_edge_finder import config as _config

    generate(
        outputs_dir=_ROOT / "outputs",
        models_dir=_config.MODELS_DIR,
        out_path=DOCS_DIR / "index.html",
    )
```

- [ ] **Step 4: Run generate_site tests**

```bash
python3 -m pytest tests/test_generate_site.py -v
```

Expected: all 12 tests pass.

- [ ] **Step 5: Run full test suite**

```bash
python3 -m pytest tests/ -v --tb=short
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/mlb_edge_finder/generate_site.py tests/test_generate_site.py
git commit -m "feat: update dashboard to use temporal eval JSON as single data source"
```

---

## Task 4: Run Temporal Eval, Commit Artifacts, Update CLAUDE.md

**Files:**
- Generate: `models/temporal_eval_2025.json`
- Regenerate: `docs/index.html`
- Modify: `CLAUDE.md`

- [ ] **Step 1: Run temporal evaluation (takes ~2–3 minutes)**

```bash
python3 -m mlb_edge_finder.temporal_eval --force
```

Expected output:
```
Temporal Eval — Holdout Season: 2025
  Train seasons : [2019, 2021, 2022, 2023, 2024]
  Train rows    : 12,xxx
  Test rows     : 2,xxx
  ROC-AUC       : 0.xxx
  Accuracy      : 0.xxx
  Bets          : x,xxx
  Win Rate      : xx.x%
  ROI           : +xx.x%
  Sharpe        : 0.xxx

Artifact: models/temporal_eval_2025.json
```

- [ ] **Step 2: Regenerate the dashboard**

```bash
python3 -m mlb_edge_finder.generate_site
```

Expected: `docs/index.html` updated with temporal eval numbers in the stats card.

- [ ] **Step 3: Verify the dashboard locally**

Open `docs/index.html` in a browser. Confirm:
- Stats card shows Win Rate, ROI, Sharpe, ROC-AUC, Holdout games
- Subtitle reads "Trained 2019–2024 · 2025 holdout" (or the actual train season range)
- P&L chart renders with a curve (not blank)

- [ ] **Step 4: Update `CLAUDE.md`**

Add an entry under the completed phases list (after the dashboard entry) describing the temporal eval feature, similar in format to existing entries. Include: the new module `temporal_eval.py`, the `simulate_bets()` extraction from `backtest.py`, the updated `generate_site.py` signature, the JSON artifact schema, and the observed ROC-AUC / ROI numbers.

Also update the test count in the "Running Tests" section to reflect the new total.

- [ ] **Step 5: Commit all artifacts**

```bash
git add models/temporal_eval_2025.json docs/index.html CLAUDE.md
git commit -m "feat: add temporal out-of-time eval — train 2019-2024, test 2025 holdout"
```

- [ ] **Step 6: Push to remote**

```bash
git push
```
