# ROI Reframe + Market-Efficiency Sensitivity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the misleading synthetic-odds ROI as the project's headline with an honest market-efficiency sensitivity analysis (where the edge dies as the market gets informed), and reframe the dashboard + README around methodology and limitations.

**Architecture:** Add a `sweep_market_efficiency()` analysis to `backtest.py` (built on an extracted `_run_bet_loop` helper so per-game odds are supported). `temporal_eval.run()` calls it and writes `market_efficiency_sweep` + `break_even_alpha` into its JSON artifact (dropping `pnl_series`). `generate_site.py` swaps the P&L chart for an efficiency chart and reorders the stats card. README/CLAUDE docs are reframed.

**Tech Stack:** Python 3.10+, pandas, numpy, scikit-learn, XGBoost, Chart.js (CDN), pytest.

**Spec:** `docs/superpowers/specs/2026-06-02-roi-reframe-design.md`

**Branch:** `feat/roi-reframe` (already created).

---

### Task 1: Refactor `backtest.py` — promote `_prob_to_american`, extract `_run_bet_loop`

**Goal:** Make the per-bet loop accept per-game odds, without changing `simulate_bets` behavior. This is a pure refactor; all existing `backtest` tests must still pass.

**Files:**
- Modify: `src/mlb_edge_finder/backtest.py`
- Test: `tests/test_backtest.py`

- [ ] **Step 1: Add the numpy import**

At the top of `src/mlb_edge_finder/backtest.py`, the current imports are:
```python
"""Backtest the edge-finder against held-out test data using synthetic market odds."""
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
```
Add `import numpy as np` after `import pandas as pd`:
```python
import numpy as np
import pandas as pd
```

- [ ] **Step 2: Promote `_to_american` to a module-level helper**

The current `simulate_market_odds` (lines ~12-39) contains an inner `_to_american` closure. Replace the whole function with this version that uses a module-level helper:

```python
def _prob_to_american(p: float) -> float:
    """Convert an implied probability to American odds."""
    if p >= 0.5:
        return -(p / (1.0 - p)) * 100.0
    return ((1.0 - p) / p) * 100.0


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
    return _prob_to_american(home_implied), _prob_to_american(away_implied)
```

- [ ] **Step 3: Run the existing market-odds tests to confirm the refactor is behavior-preserving**

Run: `python3 -m pytest tests/test_backtest.py -k simulate_market_odds -v`
Expected: PASS (5 tests: default_is_110_110, favored_home, implied_probs_sum, even_home_prob, zero_vig).

- [ ] **Step 4: Extract `_run_bet_loop` and make `simulate_bets` delegate to it**

Replace the entire current `simulate_bets` function (lines ~42-142) with the following two functions:

```python
def _run_bet_loop(
    clf: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    meta_df: pd.DataFrame,
    home_odds: pd.Series,
    away_odds: pd.Series,
    unit: float = 100.0,
    ev_threshold: float | None = None,
) -> pd.DataFrame:
    """Run the EV + Kelly bet-selection loop with per-game American odds.

    home_odds / away_odds are integer American odds Series indexed identically
    to X_test. meta_df must share that index and contain game_date, home_name,
    away_name. Returns the per-bet DataFrame (empty with correct columns if no
    bets clear the thresholds). Does not log — callers decide.
    """
    from mlb_edge_finder import config as _config
    from mlb_edge_finder.edge_finder import compute_ev, compute_kelly

    _ev_threshold = ev_threshold if ev_threshold is not None else _config.EV_THRESHOLD

    output_cols = [
        "game_date", "home_name", "away_name", "bet_side",
        "american_odds", "model_prob", "ev", "kelly_fraction",
        "actual_home_win", "won", "pnl", "cumulative_pnl",
    ]

    feature_names = list(clf.feature_names_in_)
    X_test_aligned = X_test.reindex(columns=feature_names)
    home_probs = clf.predict_proba(X_test_aligned)[:, 1]

    records = []
    for (idx, prob), actual in zip(zip(X_test.index, home_probs), y_test.values):
        row_meta = meta_df.loc[idx]
        h_odds = int(home_odds.loc[idx])
        a_odds = int(away_odds.loc[idx])
        h_payout = h_odds / 100 if h_odds > 0 else 100 / abs(h_odds)
        a_payout = a_odds / 100 if a_odds > 0 else 100 / abs(a_odds)

        home_ev = compute_ev(float(prob), h_odds)
        if home_ev > _ev_threshold and h_odds >= _config.MIN_AMERICAN_ODDS:
            won = int(actual) == 1
            records.append({
                "game_date": row_meta["game_date"],
                "home_name": row_meta["home_name"],
                "away_name": row_meta["away_name"],
                "bet_side": "home",
                "american_odds": h_odds,
                "model_prob": round(float(prob), 4),
                "ev": round(home_ev, 4),
                "kelly_fraction": round(compute_kelly(float(prob), h_odds), 4),
                "actual_home_win": int(actual),
                "won": won,
                "pnl": h_payout * unit if won else -unit,
            })

        away_prob = 1.0 - float(prob)
        away_ev = compute_ev(away_prob, a_odds)
        if away_ev > _ev_threshold and a_odds >= _config.MIN_AMERICAN_ODDS:
            won = int(actual) == 0
            records.append({
                "game_date": row_meta["game_date"],
                "home_name": row_meta["home_name"],
                "away_name": row_meta["away_name"],
                "bet_side": "away",
                "american_odds": a_odds,
                "model_prob": round(away_prob, 4),
                "ev": round(away_ev, 4),
                "kelly_fraction": round(compute_kelly(away_prob, a_odds), 4),
                "actual_home_win": int(actual),
                "won": won,
                "pnl": a_payout * unit if won else -unit,
            })

    if not records:
        return pd.DataFrame(columns=output_cols)

    result = pd.DataFrame(records).sort_values("game_date").reset_index(drop=True)
    result["cumulative_pnl"] = result["pnl"].cumsum()
    return result


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
    """Simulate edge-finder bets on a pre-split test set using a flat market.

    Generates one synthetic odds pair via simulate_market_odds(home_market_prob,
    vig), applies it to every game, and delegates the bet loop to _run_bet_loop.
    meta_df must be indexed identically to X_test and contain game_date,
    home_name, away_name.

    Args:
        clf: Fitted calibrated classifier with feature_names_in_ and predict_proba.
        X_test: Feature matrix for the test games.
        y_test: True binary labels (1 = home win) for the test games.
        meta_df: DataFrame with game_date, home_name, away_name; same index as X_test.
        home_market_prob: Market-implied home win probability. Default 0.5.
        vig: Bookmaker overround. Default 0.0476.
        unit: Dollar bet size for P&L. Default $100.
        ev_threshold: Minimum EV to flag a bet. Defaults to config.EV_THRESHOLD.

    Returns:
        DataFrame sorted by game_date with columns: game_date, home_name,
        away_name, bet_side, american_odds, model_prob, ev, kelly_fraction,
        actual_home_win, won, pnl, cumulative_pnl. Empty (with those columns)
        when no bets clear the thresholds.
    """
    from mlb_edge_finder import config as _config

    _ev_threshold = ev_threshold if ev_threshold is not None else _config.EV_THRESHOLD

    home_odds_f, away_odds_f = simulate_market_odds(home_market_prob, vig)
    home_odds = pd.Series(round(home_odds_f), index=X_test.index)
    away_odds = pd.Series(round(away_odds_f), index=X_test.index)

    result = _run_bet_loop(clf, X_test, y_test, meta_df, home_odds, away_odds, unit, _ev_threshold)
    if result.empty:
        logger.warning("No edges found in backtest at EV=%.0f%%", _ev_threshold * 100)
    return result
```

- [ ] **Step 5: Run the full backtest test file to confirm no regression**

Run: `python3 -m pytest tests/test_backtest.py -v`
Expected: PASS (all existing tests — simulate_bets and run_backtest tests unchanged in behavior).

- [ ] **Step 6: Commit**

```bash
git add src/mlb_edge_finder/backtest.py
git commit -m "refactor: extract _run_bet_loop and _prob_to_american in backtest"
```

---

### Task 2: Add `sweep_market_efficiency()` to `backtest.py`

**Goal:** New analysis function that sweeps market efficiency α from 0 (naive 50/50) to 1 (market = model's own prediction) and returns ROI at each point.

**Files:**
- Modify: `src/mlb_edge_finder/backtest.py`
- Test: `tests/test_backtest.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_backtest.py` (the helpers `_make_aligned_split` and `_make_mock_clf` already exist in this file):

```python
# --- sweep_market_efficiency ---

def test_sweep_returns_expected_columns():
    from mlb_edge_finder.backtest import sweep_market_efficiency
    clf, X_test, y_test, meta = _make_aligned_split()
    result = sweep_market_efficiency(clf, X_test, y_test, meta, ev_threshold=0.05)
    assert set(["alpha", "roi_pct", "n_bets", "win_rate"]).issubset(result.columns)


def test_sweep_one_row_per_grid_point():
    from mlb_edge_finder.backtest import sweep_market_efficiency
    clf, X_test, y_test, meta = _make_aligned_split()
    grid = [0.0, 0.25, 0.5, 0.75, 1.0]
    result = sweep_market_efficiency(clf, X_test, y_test, meta, alpha_grid=grid, ev_threshold=0.05)
    assert len(result) == len(grid)
    assert list(result["alpha"]) == grid


def test_sweep_roi_decreases_with_efficiency():
    from mlb_edge_finder.backtest import sweep_market_efficiency
    clf, X_test, y_test, meta = _make_aligned_split()
    result = sweep_market_efficiency(clf, X_test, y_test, meta, ev_threshold=0.05)
    roi_at_0 = result.loc[result["alpha"] == 0.0, "roi_pct"].iloc[0]
    roi_at_1 = result.loc[result["alpha"] == 1.0, "roi_pct"].iloc[0]
    assert roi_at_0 >= roi_at_1


def test_sweep_alpha_one_has_no_positive_edge():
    # At alpha=1 the market equals the model's own prob (+vig), so EV<=0 and no
    # bets clear the threshold -> 0 bets, roi 0.
    from mlb_edge_finder.backtest import sweep_market_efficiency
    clf, X_test, y_test, meta = _make_aligned_split()
    result = sweep_market_efficiency(clf, X_test, y_test, meta, ev_threshold=0.05)
    roi_at_1 = result.loc[result["alpha"] == 1.0, "roi_pct"].iloc[0]
    assert roi_at_1 <= 0.0


def test_sweep_handles_no_bets():
    # A 0.50 model never clears EV>0.20 against any vigged line.
    from mlb_edge_finder.backtest import sweep_market_efficiency
    clf, X_test, y_test, meta = _make_aligned_split()
    flat_clf = _make_mock_clf(home_win_prob=0.50)
    result = sweep_market_efficiency(flat_clf, X_test, y_test, meta, ev_threshold=0.20)
    assert (result["n_bets"] == 0).all()
    assert (result["roi_pct"] == 0.0).all()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m pytest tests/test_backtest.py -k sweep -v`
Expected: FAIL with `ImportError: cannot import name 'sweep_market_efficiency'`.

- [ ] **Step 3: Implement `sweep_market_efficiency`**

Insert this function in `src/mlb_edge_finder/backtest.py` immediately after `simulate_bets` (and before `run_backtest`):

```python
def sweep_market_efficiency(
    clf: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    meta_df: pd.DataFrame,
    alpha_grid: list[float] | None = None,
    vig: float = 0.0476,
    ev_threshold: float | None = None,
    unit: float = 100.0,
) -> pd.DataFrame:
    """Sweep market efficiency from naive (alpha=0) to model-sharp (alpha=1).

    At each alpha, each game's market-implied home probability is set to
    0.5*(1-alpha) + model_prob*alpha, the vig is added, the result is converted
    to per-game American odds, and the bet loop is run against it.

    alpha=0 reproduces the naive 50/50 market (today's headline ROI). alpha=1
    sets the market equal to the model's own prediction, leaving no informational
    edge (ROI collapses to roughly -vig).

    Args:
        clf: Fitted calibrated classifier with feature_names_in_ and predict_proba.
        X_test: Feature matrix for the test games.
        y_test: True binary labels (1 = home win).
        meta_df: DataFrame with game_date, home_name, away_name; same index as X_test.
        alpha_grid: Efficiency points to evaluate. Default: 0.0..1.0 step 0.05.
        vig: Bookmaker overround applied at every alpha. Default 0.0476.
        ev_threshold: Minimum EV to flag a bet. Defaults to config.EV_THRESHOLD.
        unit: Dollar bet size. Default $100.

    Returns:
        DataFrame (one row per alpha, ascending) with columns:
        alpha, roi_pct, n_bets, win_rate.
    """
    if alpha_grid is None:
        alpha_grid = [round(float(a), 4) for a in np.arange(0.0, 1.0001, 0.05)]

    feature_names = list(clf.feature_names_in_)
    X_aligned = X_test.reindex(columns=feature_names)
    home_probs = clf.predict_proba(X_aligned)[:, 1]

    rows = []
    for alpha in alpha_grid:
        market_home = 0.5 * (1.0 - alpha) + home_probs * alpha
        home_imp = np.clip(market_home + vig / 2, 1e-6, 1 - 1e-6)
        away_imp = np.clip((1.0 - market_home) + vig / 2, 1e-6, 1 - 1e-6)
        home_odds = pd.Series([round(_prob_to_american(p)) for p in home_imp], index=X_test.index)
        away_odds = pd.Series([round(_prob_to_american(p)) for p in away_imp], index=X_test.index)

        bt = _run_bet_loop(clf, X_test, y_test, meta_df, home_odds, away_odds, unit, ev_threshold)
        summary = compute_summary(bt, unit=unit)
        rows.append({
            "alpha": alpha,
            "roi_pct": summary["roi_pct"],
            "n_bets": summary["n_bets"],
            "win_rate": summary["win_rate"],
        })

    return pd.DataFrame(rows)
```

Note: `compute_summary` is defined later in the same module; Python resolves it at call time, so ordering is fine.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -m pytest tests/test_backtest.py -k sweep -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Run the full backtest file**

Run: `python3 -m pytest tests/test_backtest.py -v`
Expected: PASS (all).

- [ ] **Step 6: Commit**

```bash
git add src/mlb_edge_finder/backtest.py tests/test_backtest.py
git commit -m "feat: add sweep_market_efficiency to backtest"
```

---

### Task 3: Wire the sweep into `temporal_eval.py` and reshape the JSON

**Goal:** `run()` computes the efficiency sweep + break-even α and writes them to the artifact; `pnl_series` is dropped.

**Files:**
- Modify: `src/mlb_edge_finder/temporal_eval.py`
- Test: `tests/test_temporal_eval.py`

- [ ] **Step 1: Update the failing tests first**

In `tests/test_temporal_eval.py`, update `_run_with_mocks` to also patch the new sweep call. Replace the `with` block's patch list (lines ~89-101) — add a `sweep_market_efficiency` patch returning a small fake DataFrame. The new helper body:

```python
def _run_with_mocks(tmp_path: Path, training_df: pd.DataFrame, force: bool = False) -> dict:
    """Call temporal_eval.run() with all expensive operations mocked."""
    import mlb_edge_finder.temporal_eval as te
    mock_clf = _make_mock_clf()
    empty_backtest = pd.DataFrame(columns=[
        "game_date", "home_name", "away_name", "bet_side", "american_odds",
        "model_prob", "ev", "kelly_fraction", "actual_home_win", "won", "pnl", "cumulative_pnl",
    ])
    fake_sweep = pd.DataFrame({
        "alpha": [0.0, 0.5, 1.0],
        "roi_pct": [18.0, 4.0, -4.0],
        "n_bets": [200, 90, 8],
        "win_rate": [0.61, 0.55, 0.50],
    })

    with patch.object(te, "_load_training_csv", return_value=training_df), \
         patch("mlb_edge_finder.temporal_eval.XGBClassifier") as MockXGB, \
         patch("mlb_edge_finder.temporal_eval.calibrate", return_value=mock_clf), \
         patch("mlb_edge_finder.temporal_eval.evaluate", return_value={
             "accuracy": 0.57, "roc_auc": 0.60, "log_loss": 0.68, "brier_score": 0.24,
             "n_test_samples": 60,
         }), \
         patch("mlb_edge_finder.temporal_eval.simulate_bets", return_value=empty_backtest), \
         patch("mlb_edge_finder.temporal_eval.sweep_market_efficiency", return_value=fake_sweep), \
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
```

Then update the keys test (replace `test_run_json_has_required_keys` and `test_run_pnl_series_is_list`):

```python
def test_run_json_has_required_keys(tmp_path):
    df = _make_training_df()
    result = _run_with_mocks(tmp_path, df)
    required = {
        "holdout_season", "train_seasons", "n_train", "n_test",
        "accuracy", "roc_auc", "log_loss", "brier_score",
        "n_bets", "win_rate", "roi_pct", "sharpe_ratio",
        "total_pnl", "avg_ev", "max_drawdown",
        "market_efficiency_sweep", "break_even_alpha",
    }
    assert required.issubset(set(result.keys()))
    assert "pnl_series" not in result


def test_run_market_efficiency_sweep_is_list(tmp_path):
    df = _make_training_df()
    result = _run_with_mocks(tmp_path, df)
    sweep = result["market_efficiency_sweep"]
    assert isinstance(sweep, list)
    assert sweep and set(sweep[0].keys()) == {"alpha", "roi_pct", "n_bets"}


def test_run_break_even_alpha_interpolated(tmp_path):
    # fake_sweep crosses 0 between alpha=0.5 (roi 4.0) and alpha=1.0 (roi -4.0)
    # crossing at 0.5 + 0.5 * (4.0 / (4.0 - -4.0)) = 0.75
    df = _make_training_df()
    result = _run_with_mocks(tmp_path, df)
    assert result["break_even_alpha"] == 0.75
```

Add a direct unit test for `_break_even_alpha`:

```python
def test_break_even_alpha_returns_none_when_never_negative():
    from mlb_edge_finder.temporal_eval import _break_even_alpha
    sweep = pd.DataFrame({"alpha": [0.0, 0.5, 1.0], "roi_pct": [10.0, 5.0, 1.0]})
    assert _break_even_alpha(sweep) is None
```

Also update the two tests that hand-write an existing JSON dict containing `pnl_series` (`test_run_skips_if_exists`, `test_run_force_overwrites`) — change `"pnl_series": []` to `"market_efficiency_sweep": []` in their `existing` dicts (harmless, keeps them representative):

```python
def test_run_skips_if_exists(tmp_path):
    existing = {"holdout_season": 2025, "roc_auc": 0.999, "market_efficiency_sweep": []}
    (tmp_path / "temporal_eval_2025.json").write_text(json.dumps(existing))
    import mlb_edge_finder.temporal_eval as te
    with patch("mlb_edge_finder.temporal_eval.config") as mock_config:
        mock_config.MODELS_DIR = tmp_path
        result = te.run(holdout_season=2025, force=False)
    assert result["roc_auc"] == 0.999


def test_run_force_overwrites(tmp_path):
    existing = {"holdout_season": 2025, "roc_auc": 0.999, "market_efficiency_sweep": []}
    (tmp_path / "temporal_eval_2025.json").write_text(json.dumps(existing))
    df = _make_training_df()
    result = _run_with_mocks(tmp_path, df, force=True)
    assert result["roc_auc"] != 0.999
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m pytest tests/test_temporal_eval.py -v`
Expected: FAIL — `sweep_market_efficiency` not importable in `temporal_eval`, and `_break_even_alpha` undefined.

- [ ] **Step 3: Update the import in `temporal_eval.py`**

In `src/mlb_edge_finder/temporal_eval.py`, the current import (line ~10) is:
```python
from mlb_edge_finder.backtest import compute_summary, simulate_bets
```
Replace with:
```python
from mlb_edge_finder.backtest import compute_summary, simulate_bets, sweep_market_efficiency
```

- [ ] **Step 4: Add the `_break_even_alpha` helper**

Insert this function in `src/mlb_edge_finder/temporal_eval.py` after `_load_training_csv` (before `run`):

```python
def _break_even_alpha(sweep_df: pd.DataFrame) -> float | None:
    """Interpolate the alpha where ROI first crosses from >= 0 to < 0.

    Returns None if ROI stays non-negative across the whole grid.
    """
    rows = sweep_df.sort_values("alpha").reset_index(drop=True)
    for i in range(1, len(rows)):
        r0 = rows.loc[i - 1, "roi_pct"]
        r1 = rows.loc[i, "roi_pct"]
        if r0 >= 0 and r1 < 0:
            a0 = rows.loc[i - 1, "alpha"]
            a1 = rows.loc[i, "alpha"]
            alpha_star = a0 + (a1 - a0) * (r0 / (r0 - r1))
            return round(float(alpha_star), 4)
    return None
```

- [ ] **Step 5: Call the sweep and reshape the result dict in `run()`**

In `run()`, find this block (lines ~124-154):
```python
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
```
Replace it with:
```python
    meta_df = test_df[["game_date", "home_name", "away_name"]]
    backtest_df = simulate_bets(cal_clf, X_test, y_test, meta_df)
    summary = compute_summary(backtest_df)

    sweep_df = sweep_market_efficiency(cal_clf, X_test, y_test, meta_df)
    break_even = _break_even_alpha(sweep_df)
    market_efficiency_sweep = [
        {
            "alpha": round(float(r["alpha"]), 4),
            "roi_pct": float(r["roi_pct"]),
            "n_bets": int(r["n_bets"]),
        }
        for _, r in sweep_df.iterrows()
    ]
    logger.info(
        "Market-efficiency sweep: break-even alpha=%s",
        f"{break_even:.3f}" if break_even is not None else "none",
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
        "break_even_alpha": break_even,
        "market_efficiency_sweep": market_efficiency_sweep,
    }
```

- [ ] **Step 6: Add break-even to the CLI summary print**

In the `if __name__ == "__main__"` block, find the line:
```python
    print(f"  ROI           : {r['roi_pct']:+.1f}%")
    print(f"  Sharpe        : {r['sharpe_ratio']:.3f}")
```
Insert a break-even line after the Sharpe line:
```python
    print(f"  ROI           : {r['roi_pct']:+.1f}%")
    print(f"  Sharpe        : {r['sharpe_ratio']:.3f}")
    be = r["break_even_alpha"]
    print(f"  Break-even α  : {be:.3f}" if be is not None else "  Break-even α  : none (ROI stays positive)")
```

- [ ] **Step 7: Run the temporal-eval tests**

Run: `python3 -m pytest tests/test_temporal_eval.py -v`
Expected: PASS (all, including the new sweep/break-even tests).

- [ ] **Step 8: Commit**

```bash
git add src/mlb_edge_finder/temporal_eval.py tests/test_temporal_eval.py
git commit -m "feat: write market_efficiency_sweep + break_even_alpha in temporal eval"
```

---

### Task 4: Reframe the dashboard in `generate_site.py`

**Goal:** Swap the P&L chart for an efficiency chart; reorder the stats card to lead with ROC-AUC + break-even and tag the betting numbers as naive-market.

**Files:**
- Modify: `src/mlb_edge_finder/generate_site.py`
- Test: `tests/test_generate_site.py`

- [ ] **Step 1: Update the test fixture and write failing assertions**

In `tests/test_generate_site.py`, update `_write_temporal_eval_json` (lines ~24-48) to emit the new keys instead of `pnl_series`:

```python
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
        "max_drawdown": 420.0,
        "avg_ev": 0.28,
        "break_even_alpha": 0.1,
        "market_efficiency_sweep": [
            {"alpha": 0.0, "roi_pct": 15.1, "n_bets": 1800},
            {"alpha": 0.5, "roi_pct": 4.0, "n_bets": 700},
            {"alpha": 1.0, "roi_pct": -4.8, "n_bets": 40},
        ],
    }
    data.update(overrides)
    path.write_text(json.dumps(data))
    return data
```

Append new tests at the end of `tests/test_generate_site.py`:

```python
def test_generate_renders_efficiency_chart(tmp_path):
    outputs_dir = tmp_path / "outputs"
    outputs_dir.mkdir()
    (outputs_dir / "edges_2025-01-01.csv").write_text(
        "game_id,home_team,away_team,bet_side,american_odds,model_prob,ev,kelly_fraction,high_confidence\n"
    )
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    _write_temporal_eval_json(models_dir / "temporal_eval_2025.json")
    out = tmp_path / "index.html"
    generate(outputs_dir=outputs_dir, models_dir=models_dir, out_path=out)
    html = out.read_text()
    assert "efficiency-chart" in html
    assert "market_efficiency_sweep" in html


def test_generate_no_pnl_chart(tmp_path):
    outputs_dir = tmp_path / "outputs"
    outputs_dir.mkdir()
    (outputs_dir / "edges_2025-01-01.csv").write_text(
        "game_id,home_team,away_team,bet_side,american_odds,model_prob,ev,kelly_fraction,high_confidence\n"
    )
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    _write_temporal_eval_json(models_dir / "temporal_eval_2025.json")
    out = tmp_path / "index.html"
    generate(outputs_dir=outputs_dir, models_dir=models_dir, out_path=out)
    html = out.read_text()
    assert "pnl-chart" not in html
    assert "pnl_series" not in html


def test_generate_stats_card_shows_roc_and_break_even(tmp_path):
    outputs_dir = tmp_path / "outputs"
    outputs_dir.mkdir()
    (outputs_dir / "edges_2025-01-01.csv").write_text(
        "game_id,home_team,away_team,bet_side,american_odds,model_prob,ev,kelly_fraction,high_confidence\n"
    )
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    _write_temporal_eval_json(models_dir / "temporal_eval_2025.json")
    out = tmp_path / "index.html"
    generate(outputs_dir=outputs_dir, models_dir=models_dir, out_path=out)
    html = out.read_text()
    assert "0.601" in html            # ROC-AUC present
    assert "ROC-AUC" in html
    assert "naive market" in html.lower()   # ROI labeled as naive
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m pytest tests/test_generate_site.py -k "efficiency or pnl or break_even or roc" -v`
Expected: FAIL (efficiency-chart/market_efficiency_sweep absent; pnl-chart still present).

- [ ] **Step 3: Rename `_render_pnl_html` → `_render_efficiency_html`**

In `src/mlb_edge_finder/generate_site.py`, replace the `_render_pnl_html` function (lines ~113-122) with:

```python
def _render_efficiency_html(te_data: dict | None) -> str:
    """Render the market-efficiency sensitivity chart card, or '' if no data."""
    if te_data is None or not te_data.get("market_efficiency_sweep"):
        return ""
    return (
        '<div class="card">'
        '<div class="card-title">Edge vs Market Efficiency</div>'
        '<div class="chart-wrap-sm"><canvas id="efficiency-chart"></canvas></div>'
        '<div class="card-caption">Synthetic-market stress test &mdash; betting ROI as the '
        "market becomes as informed as the model (0 = ignores matchup, 1 = as sharp as the model).</div>"
        "</div>"
    )
```

- [ ] **Step 4: Reorder and relabel the stats card**

Replace the `_render_stats_html` function (lines ~62-110) with:

```python
def _render_stats_html(te_data: dict | None) -> str:
    """Render the holdout-evaluation stats card, or '' if no data."""
    if not te_data:
        return ""
    rows = []
    if "roc_auc" in te_data:
        rows.append(
            f'<div class="stat-row"><span class="stat-label">ROC-AUC</span>'
            f'<span class="stat-value neutral">{te_data["roc_auc"]:.3f}</span></div>'
        )
    if te_data.get("break_even_alpha") is not None:
        rows.append(
            f'<div class="stat-row"><span class="stat-label">Break-even efficiency</span>'
            f'<span class="stat-value">&alpha; &approx; {te_data["break_even_alpha"]:.2f}</span></div>'
        )
    elif "break_even_alpha" in te_data:
        rows.append(
            '<div class="stat-row"><span class="stat-label">Break-even efficiency</span>'
            '<span class="stat-value neutral">none in range</span></div>'
        )
    if "accuracy" in te_data:
        rows.append(
            f'<div class="stat-row"><span class="stat-label">Accuracy</span>'
            f'<span class="stat-value neutral">{te_data["accuracy"] * 100:.1f}%</span></div>'
        )
    if "n_test" in te_data:
        rows.append(
            f'<div class="stat-row"><span class="stat-label">Holdout games</span>'
            f'<span class="stat-value neutral">{te_data["n_test"]:,}</span></div>'
        )
    if "win_rate" in te_data:
        rows.append(
            f'<div class="stat-row"><span class="stat-label">Win Rate (naive market)</span>'
            f'<span class="stat-value">{te_data["win_rate"] * 100:.1f}%</span></div>'
        )
    if "roi_pct" in te_data:
        roi = te_data["roi_pct"]
        roi_prefix = "+" if roi >= 0 else ""
        roi_class = "stat-value green" if roi >= 0 else "stat-value"
        rows.append(
            f'<div class="stat-row"><span class="stat-label">ROI (naive market)</span>'
            f'<span class="{roi_class}">{roi_prefix}{roi:.1f}%</span></div>'
        )
    if not rows:
        return ""
    train_seasons = te_data.get("train_seasons", [])
    holdout = te_data.get("holdout_season", "")
    subtitle = ""
    if train_seasons and holdout:
        subtitle = (
            f'<div class="card-subtitle">Trained {train_seasons[0]}&ndash;'
            f'{train_seasons[-1]} &middot; {holdout} holdout &middot; synthetic-market stress test</div>'
        )
    return (
        '<div class="card"><div class="card-title">Holdout Evaluation</div>'
        + subtitle
        + "".join(rows)
        + "</div>"
    )
```

- [ ] **Step 5: Update `_render_html` to use the renamed renderer and add the caption CSS**

In `_render_html`, change the line (around line 179):
```python
    pnl_chart_html = _render_pnl_html(te_data)
```
to:
```python
    efficiency_chart_html = _render_efficiency_html(te_data)
```

In the same function's template, change the sidebar line (around line 252):
```python
      {pnl_chart_html}
```
to:
```python
      {efficiency_chart_html}
```

Add a `.card-caption` style. Find the `.card-subtitle` CSS line (around line 216):
```python
    .card-subtitle{{font-size:10px;color:#8a8070;margin-bottom:10px;opacity:0.75}}
```
Add immediately after it:
```python
    .card-caption{{font-size:10px;color:#8a8070;margin-top:8px;line-height:1.4;opacity:0.8}}
```

- [ ] **Step 6: Replace the P&L Chart.js IIFE with the efficiency chart**

In `_render_html`'s `<script>` block, replace the second IIFE (the one reading `TE.pnl_series`, lines ~266-271) with:

```python
(function(){{
  var el=document.getElementById('efficiency-chart');
  if(!el||!TE||!TE.market_efficiency_sweep||TE.market_efficiency_sweep.length===0)return;
  var S=TE.market_efficiency_sweep;
  var ctx=el.getContext('2d');
  new Chart(ctx,{{type:'line',data:{{labels:S.map(function(d){{return d.alpha}}),datasets:[{{data:S.map(function(d){{return d.roi_pct}}),borderColor:'#FD5A1E',borderWidth:2,pointRadius:0,tension:0.1,fill:false}}]}},options:{{responsive:true,maintainAspectRatio:false,plugins:{{legend:{{display:false}},tooltip:{{callbacks:{{title:function(t){{return'α = '+S[t[0].dataIndex].alpha}},label:function(t){{return'ROI '+t.raw.toFixed(1)+'%'}}}}}}}},scales:{{x:{{grid:{{color:'#3d393044'}},ticks:{{color:'#8a8070',font:{{size:9}}}}}},y:{{grid:{{color:'#3d393044'}},ticks:{{color:'#8a8070',font:{{size:10}},callback:function(v){{return v+'%'}}}}}}}}}}}});
}})();
```

Note the brace discipline: this is inside a Python f-string, so every literal JS `{` is `{{` and `}` is `}}`. The IIFE opens with `(function(){{` and closes with `}})();`. Count the closing sequence after `}});` for the `new Chart(...)` call carefully — it mirrors the working history-chart IIFE above it. After editing, Step 8 verifies the rendered JS is balanced.

- [ ] **Step 7: Run the generate-site tests**

Run: `python3 -m pytest tests/test_generate_site.py -v`
Expected: PASS (all, including the 3 new tests; pre-existing `test_generate_includes_stats_when_temporal_eval_present` still passes because 0.601 / 15.1% / 60.3% / "2025 holdout" remain present).

- [ ] **Step 8: Verify the rendered dashboard JS is syntactically balanced**

Run:
```bash
python3 -c "
import json, re
from pathlib import Path
import mlb_edge_finder.generate_site as gs
html = gs._render_html([], [{'date':'2025-01-01','count':1}], json.loads(Path('models/temporal_eval_2025.json').read_text()) if Path('models/temporal_eval_2025.json').exists() else {'roc_auc':0.6,'accuracy':0.55,'n_test':2400,'win_rate':0.6,'roi_pct':15.0,'break_even_alpha':0.1,'train_seasons':[2019,2024],'holdout_season':2025,'market_efficiency_sweep':[{'alpha':0.0,'roi_pct':15.0,'n_bets':100},{'alpha':1.0,'roi_pct':-4.0,'n_bets':5}]}, 'now')
js = '\n'.join(re.findall(r'<script>(.*?)</script>', html, re.S))
assert js.count('{') == js.count('}'), (js.count('{'), js.count('}'))
assert 'efficiency-chart' in html and 'pnl-chart' not in html
print('OK: braces balanced, efficiency chart present, pnl chart gone')
"
```
Expected: `OK: braces balanced, efficiency chart present, pnl chart gone`

- [ ] **Step 9: Commit**

```bash
git add src/mlb_edge_finder/generate_site.py tests/test_generate_site.py
git commit -m "feat: dashboard shows market-efficiency chart, ROC-AUC-led stats card"
```

---

### Task 5: Fix the stale `model.calibrate()` docstring

**Files:**
- Modify: `src/mlb_edge_finder/model.py:105-126`

- [ ] **Step 1: Replace the inaccurate sentence**

In `calibrate()`'s docstring, replace:
```python
    """Wrap a fitted classifier with isotonic probability calibration.

    Uses sklearn's CalibratedClassifierCV with cv='prefit' so the underlying
    model is not retrained — only the calibration layer is fit on X_val/y_val.
    X_val must be held out from the data used to train clf.
```
with:
```python
    """Wrap a fitted classifier with isotonic probability calibration.

    Wraps clf in a FrozenEstimator (sklearn 1.6+) so the underlying model is
    not retrained — only the isotonic calibration layer is fit on X_val/y_val
    via CalibratedClassifierCV. X_val must be held out from the data used to
    train clf.
```

- [ ] **Step 2: Confirm nothing broke**

Run: `python3 -m pytest tests/test_model.py -v`
Expected: PASS (docstring-only change).

- [ ] **Step 3: Commit**

```bash
git add src/mlb_edge_finder/model.py
git commit -m "docs: fix stale calibrate() docstring (FrozenEstimator, not cv=prefit)"
```

---

### Task 6: Regenerate artifacts and run the full suite

**Goal:** Produce the real `temporal_eval_2025.json` (with the sweep) and the rebuilt `docs/index.html`, then confirm everything passes.

**Files:**
- Modify (generated): `models/temporal_eval_2025.json`, `docs/index.html`

- [ ] **Step 1: Regenerate the temporal-eval artifact with the sweep**

Run: `python3 -m mlb_edge_finder.temporal_eval --force`
Expected: prints a summary table including a `Break-even α` line; writes `models/temporal_eval_2025.json`. Requires `data/processed/training_2019-2026.csv` (present on disk).

- [ ] **Step 2: Inspect the new artifact**

Run:
```bash
python3 -c "
import json
d=json.load(open('models/temporal_eval_2025.json'))
print('keys:', sorted(d.keys()))
print('break_even_alpha:', d['break_even_alpha'])
print('sweep points:', len(d['market_efficiency_sweep']))
print('alpha=0 roi:', d['market_efficiency_sweep'][0])
print('alpha=1 roi:', d['market_efficiency_sweep'][-1])
assert 'pnl_series' not in d
print('OK')
"
```
Expected: `market_efficiency_sweep` has 21 points, alpha=0 roi ≈ the old +18%, alpha=1 roi ≤ 0, `pnl_series` absent, `break_even_alpha` is a float or null. **Record the actual `break_even_alpha` value — it is needed for README copy in Task 7.**

- [ ] **Step 3: Regenerate the dashboard**

Run: `python3 -m mlb_edge_finder.generate_site`
Expected: writes `docs/index.html`, logs "Dashboard written".

- [ ] **Step 4: Confirm the dashboard reflects the new data**

Run:
```bash
python3 -c "
html=open('docs/index.html').read()
assert 'efficiency-chart' in html
assert 'pnl-chart' not in html
assert 'Holdout Evaluation' in html
assert html.count('{')==html.count('}') or True  # full-page braces include CSS; JS checked in Task 4
print('OK: dashboard regenerated with efficiency chart')
"
```
Expected: `OK: dashboard regenerated with efficiency chart`

- [ ] **Step 5: Run the entire test suite**

Run: `python3 -m pytest tests/ -q`
Expected: PASS — all tests (was 220; now higher with the ~10 new tests).

- [ ] **Step 6: Commit the regenerated artifacts**

```bash
git add models/temporal_eval_2025.json docs/index.html
git commit -m "chore: regenerate temporal eval + dashboard with market-efficiency sweep"
```

---

### Task 7: Reframe the README and update CLAUDE.md

**Goal:** Honest narrative — lead with methodology, tag ROI as illustrative, report break-even α, add a Limitations section. Update CLAUDE.md and fix its two stale lines.

**Files:**
- Modify: `README.md`, `CLAUDE.md`

- [ ] **Step 1: Reframe the README Dashboard section**

In `README.md`, replace the Dashboard paragraph (the line beginning "Updated daily by GitHub Actions at 9:30 AM EDT...") with:

```markdown
Updated daily by GitHub Actions at 9:30 AM EDT. Shows today's recommended edges (★ marks high-confidence picks), a 30-day edge history, and an honest evaluation of the model on a **true temporal holdout** (trained on 2019–2024, tested blind on the full 2025 season). The headline chart is a **market-efficiency stress test**: it shows how quickly the model's apparent betting edge disappears as the synthetic market is made more informed — because the edge against a naive market is not a real-world edge. See [Limitations](#limitations--what-id-do-next).
```

- [ ] **Step 2: Reframe the Model section's temporal table and add break-even**

In `README.md`, replace the temporal-holdout table and the paragraph after it (the block starting "**Temporal holdout performance (trained 2019–2024..." through the paragraph ending "...the dashboard's headline numbers.") with:

```markdown
**Temporal holdout performance (trained 2019–2024, tested on full 2025 season, n=2,444):**
| Metric | Value |
|---|---|
| ROC-AUC | 0.555 |
| Accuracy | 54.7% |
| Brier score | 0.248 |
| Win rate — vs naive synthetic market (illustrative) | 62.0% |
| ROI — vs naive synthetic market (illustrative) | +18.3% |

The temporal holdout is the credible evaluation — the model never saw 2025 during training. **The ROC-AUC of 0.555 is the honest headline: a weak-but-positive ranking signal, not a profitable system.** The win rate and ROI are computed against a *naive synthetic 50/50 market* and are illustrative only — a real sportsbook prices the favorite, which erases most of that apparent edge. The market-efficiency sweep on the dashboard quantifies exactly how fragile it is: the edge breaks even once the synthetic market is only **α ≈ <FILL_BREAK_EVEN>** of the way to being as informed as the model itself.
```

Replace `<FILL_BREAK_EVEN>` with the actual `break_even_alpha` recorded in Task 6 Step 2 (e.g. `0.10`). If it was `null`, instead write: "the edge persists across the full sweep — see the Limitations note on why this synthetic test is still optimistic."

- [ ] **Step 3: Add a Limitations section**

In `README.md`, insert this section immediately before the `## Running Tests` heading:

```markdown
## Limitations & What I'd Do Next

This is a portfolio project on a genuinely hard problem; the value is the end-to-end system and a rigorous, self-critical evaluation — not a claim of beating the market. Known limitations:

- **Synthetic odds, not real lines.** The backtest prices every game against a synthetic market, not real historical bookmaker odds (which require a paid Odds API plan). A real book moves the line toward the favorite, so the naive-market ROI overstates any real edge.
- **The market-efficiency sweep uses the model as its own "sharp market."** Interpolating toward the model's own prediction is a principled proxy for an informed market, not a claim of equivalence to live lines — a real market could be sharper or duller.
- **Weak signal.** ROC-AUC 0.555 is barely above chance. The model also slightly underperforms its own predictions on the bets it makes (predicts ~66%, realizes ~62%), a sign of calibration drift under the 2024→2025 temporal shift.
- **Look-ahead in team stats.** Pitcher stats are time-matched to each game; team batting/pitching stats still use end-of-season values (a remaining, smaller source of leakage).

**What I'd do next:** a real-odds backtest against historical closing lines; time-matched team-stat snapshots (mirroring the pitcher-snapshot approach); and richer features (rest days, travel, ballpark factors, weather).
```

- [ ] **Step 4: Update the Backtest section's caveat**

In `README.md`, in the `## Backtest` section, replace the final paragraph (starting "The dashboard shows the **temporal holdout** results...") with:

```markdown
The dashboard shows the **temporal holdout** results (trained 2019–2024, tested on 2025) and a **market-efficiency sweep** rather than a raw P&L curve. The sweep interpolates each game's synthetic market probability from naive (50/50) toward the model's own prediction and plots betting ROI at each step; the break-even point is the headline. The notebook retains the original random-split P&L results for comparison.
```

- [ ] **Step 5: Update the test count line**

In `README.md`, find:
```markdown
220 smoke + integration tests. All pass.
```
Replace `220` with the actual count printed by `python3 -m pytest tests/ -q` in Task 6 Step 5 (e.g. `230`).

- [ ] **Step 6: Update CLAUDE.md — add the reframe bullet and fix stale lines**

In `CLAUDE.md`:

(a) Fix the stale `_load_training_csv` description. Find the phrase:
```
`_load_training_csv()` globs for most data-rich `training_*.csv` (largest file by size).
```
Replace with:
```
`_load_training_csv()` globs `training_*.csv` and picks the file with the earliest start season then latest end season (parsed from the filename).
```

(b) Append a new bullet after the temporal-eval bullet (the one ending "8 new tests (220 total passing)."):
```
- **ROI reframe + market-efficiency sweep complete:** `backtest._prob_to_american` promoted to module level; `backtest._run_bet_loop(clf, X_test, y_test, meta_df, home_odds, away_odds, ...)` extracted from `simulate_bets` to accept per-game odds; `simulate_bets` now delegates to it (behavior unchanged). `backtest.sweep_market_efficiency(clf, X_test, y_test, meta_df, alpha_grid=None, vig=0.0476, ...)` sweeps market efficiency α∈[0,1] — each game's market-implied home prob set to `0.5*(1-α) + model_prob*α`, vig added, converted to per-game American odds, bet loop run — returning `alpha, roi_pct, n_bets, win_rate`. `temporal_eval._break_even_alpha(sweep_df)` linearly interpolates the α where ROI crosses 0 (None if never). `temporal_eval.run()` writes `market_efficiency_sweep` + `break_even_alpha` to the JSON and drops `pnl_series`. `generate_site.py`: `_render_pnl_html`→`_render_efficiency_html` (canvas `efficiency-chart` reading `market_efficiency_sweep`); stats card renamed "Holdout Evaluation", leads with ROC-AUC + break-even α, tags ROI/win-rate as "(naive market)". `model.calibrate()` docstring corrected (FrozenEstimator). Dashboard now tells an honest story: weak positive signal (ROC-AUC 0.555) whose naive-market ROI collapses as the market gets informed. README gains a "Limitations & What I'd Do Next" section. N new tests (NNN total passing).
```
Replace `N` and `NNN` with the actual new-test count and total from Task 6 Step 5.

(c) Update the test-count line near the bottom of CLAUDE.md:
```
220 smoke + integration tests. All pass.
```
Replace `220` with the actual total.

- [ ] **Step 7: Commit**

```bash
git add README.md CLAUDE.md
git commit -m "docs: reframe README/CLAUDE around honest evaluation + market-efficiency sweep"
```

---

## Final verification

- [ ] Run the full suite once more: `python3 -m pytest tests/ -q` → all pass.
- [ ] Confirm `git status` is clean and the branch contains the seven commits above.
- [ ] Hand off to `superpowers:finishing-a-development-branch`.
