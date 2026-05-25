# Threshold Sweep & Market-Edge Filter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `market_implied_prob()` helper and `MIN_PROB_EDGE` filter to cut daily bet volume, then sweep `(EV_THRESHOLD, MIN_PROB_EDGE)` over the synthetic backtest to find the Sharpe-optimal combination and hard-code it in `config.py`.

**Architecture:** `market_implied_prob()` is added to `edge_finder.py`; `find_edges()` gains a `min_prob_edge` parameter that filters out bets where the model doesn't meaningfully disagree with the bookmaker. `run_backtest()` gains explicit `ev_threshold` and `min_prob_edge` parameters so the new `sweep_thresholds()` function can drive them without mutating global config. The winning pair is committed to `config.py` and the notebook updated to show the sweep results.

**Tech Stack:** Python 3.10+, pandas, numpy, XGBoost, pytest, Jupyter

---

## File Map

| Action | File |
|---|---|
| Modify | `src/mlb_edge_finder/edge_finder.py` |
| Modify | `src/mlb_edge_finder/config.py` |
| Modify | `src/mlb_edge_finder/backtest.py` |
| Modify | `tests/test_edge_finder.py` |
| Modify | `tests/test_backtest.py` |
| Modify | `notebooks/02_backtest.ipynb` |

---

## Task 1: `market_implied_prob()` in `edge_finder.py`

**Files:**
- Modify: `src/mlb_edge_finder/edge_finder.py`
- Modify: `tests/test_edge_finder.py`

- [ ] **Step 1: Write failing tests**

Add these tests to `tests/test_edge_finder.py` after the existing `test_compute_ev_*` tests:

```python
def test_market_implied_prob_favourite():
    """Negative odds -110: implied = 110/210 ≈ 0.5238."""
    from mlb_edge_finder.edge_finder import market_implied_prob
    result = market_implied_prob(-110)
    assert abs(result - 110 / 210) < 1e-6


def test_market_implied_prob_underdog():
    """Positive odds +130: implied = 100/230 ≈ 0.4348."""
    from mlb_edge_finder.edge_finder import market_implied_prob
    result = market_implied_prob(130)
    assert abs(result - 100 / 230) < 1e-6


def test_market_implied_prob_even_money():
    """+100 odds: implied = 100/200 = 0.50."""
    from mlb_edge_finder.edge_finder import market_implied_prob
    result = market_implied_prob(100)
    assert abs(result - 0.50) < 1e-6


def test_market_implied_prob_zero_odds_returns_half():
    """Degenerate input odds=0 → 0.5 with a warning (don't crash)."""
    from mlb_edge_finder.edge_finder import market_implied_prob
    result = market_implied_prob(0)
    assert result == 0.5
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd /Users/jaydengould/Documents/projects/mlb-edge-finder
pytest tests/test_edge_finder.py::test_market_implied_prob_favourite tests/test_edge_finder.py::test_market_implied_prob_underdog tests/test_edge_finder.py::test_market_implied_prob_even_money tests/test_edge_finder.py::test_market_implied_prob_zero_odds_returns_half -v
```

Expected: 4 errors — `ImportError: cannot import name 'market_implied_prob'`

- [ ] **Step 3: Implement `market_implied_prob()`**

Add this function to `src/mlb_edge_finder/edge_finder.py` after `compute_ev()` and before `compute_kelly()`:

```python
def market_implied_prob(american_odds: int) -> float:
    """Convert American odds to raw bookmaker-implied probability (vig included).

    Args:
        american_odds: Bookmaker's American moneyline. 0 is treated as even money.

    Returns:
        Implied probability in [0.0, 1.0].
    """
    if american_odds == 0:
        logger.warning("market_implied_prob received odds=0, returning 0.5")
        return 0.5
    if american_odds < 0:
        return abs(american_odds) / (abs(american_odds) + 100)
    return 100 / (american_odds + 100)
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
pytest tests/test_edge_finder.py::test_market_implied_prob_favourite tests/test_edge_finder.py::test_market_implied_prob_underdog tests/test_edge_finder.py::test_market_implied_prob_even_money tests/test_edge_finder.py::test_market_implied_prob_zero_odds_returns_half -v
```

Expected: 4 PASSED

- [ ] **Step 5: Commit**

```bash
git add src/mlb_edge_finder/edge_finder.py tests/test_edge_finder.py
git commit -m "feat: add market_implied_prob() to edge_finder"
```

---

## Task 2: `MIN_PROB_EDGE` in `config.py` and filter in `find_edges()`

**Files:**
- Modify: `src/mlb_edge_finder/config.py`
- Modify: `src/mlb_edge_finder/edge_finder.py`
- Modify: `tests/test_edge_finder.py`

- [ ] **Step 1: Add `MIN_PROB_EDGE` to config**

In `src/mlb_edge_finder/config.py`, add after `MIN_AMERICAN_ODDS`:

```python
MIN_PROB_EDGE: float = 0.0  # Updated after threshold sweep
```

- [ ] **Step 2: Write failing tests for the prob-edge filter**

Add to `tests/test_edge_finder.py` after the existing `find_edges` tests:

```python
def test_find_edges_min_prob_edge_filters_weak_disagreement(tmp_path):
    """Bet is excluded when model_prob - market_implied_prob <= min_prob_edge.

    home_prob=0.60, home_odds=+110 → market_implied=100/210≈0.476
    disagreement = 0.60 - 0.476 = 0.124
    With min_prob_edge=0.15, 0.124 < 0.15 → excluded.
    """
    from mlb_edge_finder.edge_finder import find_edges
    features_df = _make_features(home_odds=110, away_odds=-140)
    clf = _make_clf(home_proba=0.60)

    with patch("mlb_edge_finder.edge_finder.config.DATA_PROCESSED_DIR", tmp_path), \
         patch("mlb_edge_finder.edge_finder.config.EV_THRESHOLD", 0.05), \
         patch("mlb_edge_finder.edge_finder.config.MIN_AMERICAN_ODDS", -300), \
         patch("mlb_edge_finder.edge_finder.config.MIN_PROB_EDGE", 0.0):
        result = find_edges(features_df, clf, GAME_DATE, min_prob_edge=0.15)

    assert result.empty


def test_find_edges_min_prob_edge_passes_strong_disagreement(tmp_path):
    """Bet is included when model_prob - market_implied_prob > min_prob_edge.

    home_prob=0.75, home_odds=+110 → market_implied≈0.476
    disagreement = 0.75 - 0.476 = 0.274
    With min_prob_edge=0.15, 0.274 > 0.15 → included.
    """
    from mlb_edge_finder.edge_finder import find_edges
    features_df = _make_features(home_odds=110, away_odds=-140)
    clf = _make_clf(home_proba=0.75)

    with patch("mlb_edge_finder.edge_finder.config.DATA_PROCESSED_DIR", tmp_path), \
         patch("mlb_edge_finder.edge_finder.config.EV_THRESHOLD", 0.05), \
         patch("mlb_edge_finder.edge_finder.config.MIN_AMERICAN_ODDS", -300), \
         patch("mlb_edge_finder.edge_finder.config.MIN_PROB_EDGE", 0.0):
        result = find_edges(features_df, clf, GAME_DATE, min_prob_edge=0.15)

    assert len(result) == 1
    assert result.iloc[0]["bet_side"] == "home"


def test_find_edges_logs_prob_edge_filter_count(tmp_path, caplog):
    """find_edges logs how many edges the prob-edge filter kept."""
    import logging
    from mlb_edge_finder.edge_finder import find_edges
    features_df = _make_features(home_odds=110, away_odds=-140)
    clf = _make_clf(home_proba=0.75)

    with patch("mlb_edge_finder.edge_finder.config.DATA_PROCESSED_DIR", tmp_path), \
         patch("mlb_edge_finder.edge_finder.config.EV_THRESHOLD", 0.05), \
         patch("mlb_edge_finder.edge_finder.config.MIN_AMERICAN_ODDS", -300), \
         patch("mlb_edge_finder.edge_finder.config.MIN_PROB_EDGE", 0.0), \
         caplog.at_level(logging.INFO, logger="mlb_edge_finder.edge_finder"):
        find_edges(features_df, clf, GAME_DATE, min_prob_edge=0.15)

    assert any("prob-edge filter" in r.message for r in caplog.records)
```

- [ ] **Step 3: Run failing tests**

```bash
pytest tests/test_edge_finder.py::test_find_edges_min_prob_edge_filters_weak_disagreement tests/test_edge_finder.py::test_find_edges_min_prob_edge_passes_strong_disagreement tests/test_edge_finder.py::test_find_edges_logs_prob_edge_filter_count -v
```

Expected: 3 FAILED — `find_edges() got unexpected keyword argument 'min_prob_edge'`

- [ ] **Step 4: Update `find_edges()` to accept and apply `min_prob_edge`**

Replace the `find_edges` signature and filter logic in `src/mlb_edge_finder/edge_finder.py`. The full updated function:

```python
def find_edges(
    features_df: pd.DataFrame,
    clf: XGBClassifier,
    game_date: date,
    min_prob_edge: float | None = None,
) -> pd.DataFrame:
    """Run inference and return games with positive expected value.

    Uses clf.feature_names_in_ to select exactly the columns the model was
    trained on, then runs two passes (home, away) to find bets where:
      - EV > config.EV_THRESHOLD
      - The relevant team's American odds >= config.MIN_AMERICAN_ODDS
      - model_prob - market_implied_prob(odds) > min_prob_edge

    Logs a warning and returns an empty DataFrame (with correct columns) if no
    edges are found. Writes results to DATA_PROCESSED_DIR/edges_YYYY-MM-DD.csv.

    Args:
        features_df: Output of features.load_features() or build_features().
            Must contain all columns in clf.feature_names_in_, plus
            game_id, home_team, away_team, home_odds_american, away_odds_american.
        clf: Fitted XGBClassifier from model.load_model() or train().
        game_date: Used to name the output CSV.
        min_prob_edge: Minimum required gap between model_prob and
            market_implied_prob. Defaults to config.MIN_PROB_EDGE.

    Returns:
        DataFrame with columns: game_id, home_team, away_team,
        bet_side, american_odds, model_prob, ev, kelly_fraction, prob_flag —
        one row per flagged edge. prob_flag=True when model_prob > 0.80.

    Raises:
        ValueError: If features_df is missing any column in clf.feature_names_in_.
    """
    if min_prob_edge is None:
        min_prob_edge = config.MIN_PROB_EDGE

    output_cols = [
        "game_id", "home_team", "away_team", "bet_side",
        "american_odds", "model_prob", "ev", "kelly_fraction", "prob_flag",
    ]

    if features_df.empty:
        logger.warning("No features available for %s — returning empty edges", game_date)
        return pd.DataFrame(columns=output_cols)

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
    home_implied = np.array(
        [market_implied_prob(int(o)) for o in df["home_odds_american"]]
    )
    home_mask = (
        (home_ev > config.EV_THRESHOLD)
        & (df["home_odds_american"] >= config.MIN_AMERICAN_ODDS)
        & ((home_prob - home_implied) > min_prob_edge)
    )
    home_edges = df.loc[home_mask, ["game_id", "home_team", "away_team"]].copy()
    home_edges["bet_side"] = "home"
    home_edges["american_odds"] = df.loc[home_mask, "home_odds_american"].values
    home_edges["model_prob"] = home_prob[home_mask.values]
    home_edges["ev"] = home_ev[home_mask].values
    home_edges["kelly_fraction"] = [
        compute_kelly(float(p), int(o))
        for p, o in zip(home_prob[home_mask.values], df.loc[home_mask, "home_odds_american"].values)
    ]
    home_edges["prob_flag"] = home_prob[home_mask.values] > 0.80

    # Away pass
    away_ev = pd.Series(
        [compute_ev(float(p), int(o)) for p, o in zip(away_prob, df["away_odds_american"])]
    )
    away_implied = np.array(
        [market_implied_prob(int(o)) for o in df["away_odds_american"]]
    )
    away_mask = (
        (away_ev > config.EV_THRESHOLD)
        & (df["away_odds_american"] >= config.MIN_AMERICAN_ODDS)
        & ((away_prob - away_implied) > min_prob_edge)
    )
    away_edges = df.loc[away_mask, ["game_id", "home_team", "away_team"]].copy()
    away_edges["bet_side"] = "away"
    away_edges["american_odds"] = df.loc[away_mask, "away_odds_american"].values
    away_edges["model_prob"] = away_prob[away_mask.values]
    away_edges["ev"] = away_ev[away_mask].values
    away_edges["kelly_fraction"] = [
        compute_kelly(float(p), int(o))
        for p, o in zip(away_prob[away_mask.values], df.loc[away_mask, "away_odds_american"].values)
    ]
    away_edges["prob_flag"] = away_prob[away_mask.values] > 0.80

    edges = pd.concat([home_edges[output_cols], away_edges[output_cols]], ignore_index=True)

    n_before = home_mask.sum() + away_mask.sum()
    logger.info(
        "Prob-edge filter (%.0f%%): %d edges kept after all filters for %s",
        min_prob_edge * 100,
        len(edges),
        game_date,
    )

    if edges.empty:
        logger.warning("No edges found for %s", game_date)
        return pd.DataFrame(columns=output_cols)

    config.DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.DATA_PROCESSED_DIR / f"edges_{game_date}.csv"
    edges.to_csv(out_path, index=False)
    logger.info("Found %d edge(s) for %s → %s", len(edges), game_date, out_path)

    return edges
```

Note: add `import numpy as np` at the top of `edge_finder.py` if not already present.

- [ ] **Step 5: Run new tests**

```bash
pytest tests/test_edge_finder.py::test_find_edges_min_prob_edge_filters_weak_disagreement tests/test_edge_finder.py::test_find_edges_min_prob_edge_passes_strong_disagreement tests/test_edge_finder.py::test_find_edges_logs_prob_edge_filter_count -v
```

Expected: 3 PASSED

- [ ] **Step 6: Run full edge_finder test suite (regression)**

```bash
pytest tests/test_edge_finder.py -v
```

Expected: all previously passing tests still PASS. If any fail due to the new `min_prob_edge` parameter, check that `config.MIN_PROB_EDGE = 0.0` is set and that patching `config.EV_THRESHOLD` still works (the new code reads `config.MIN_PROB_EDGE` at call time via the default `None` path).

- [ ] **Step 7: Commit**

```bash
git add src/mlb_edge_finder/config.py src/mlb_edge_finder/edge_finder.py tests/test_edge_finder.py
git commit -m "feat: add MIN_PROB_EDGE filter to find_edges and config"
```

---

## Task 3: Update `run_backtest()` to accept explicit thresholds

**Files:**
- Modify: `src/mlb_edge_finder/backtest.py`
- Modify: `tests/test_backtest.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_backtest.py`:

```python
def test_run_backtest_explicit_ev_threshold_filters_bets():
    """A very high explicit ev_threshold produces fewer bets than a low one."""
    df = _make_training_df(200)
    clf = _make_mock_clf(home_win_prob=0.65)
    result_low = run_backtest(clf, df, ev_threshold=0.05, min_prob_edge=0.0)
    result_high = run_backtest(clf, df, ev_threshold=0.50, min_prob_edge=0.0)
    assert len(result_high) <= len(result_low)


def test_run_backtest_explicit_min_prob_edge_filters_bets():
    """A high min_prob_edge filters out bets where model barely disagrees with market."""
    df = _make_training_df(200)
    clf = _make_mock_clf(home_win_prob=0.65)
    result_no_filter = run_backtest(clf, df, ev_threshold=0.05, min_prob_edge=0.0)
    result_filtered = run_backtest(clf, df, ev_threshold=0.05, min_prob_edge=0.30)
    assert len(result_filtered) <= len(result_no_filter)


def test_run_backtest_defaults_unchanged():
    """run_backtest() with no threshold args still runs (default config values)."""
    df = _make_training_df(200)
    clf = _make_mock_clf(home_win_prob=0.65)
    result = run_backtest(clf, df)
    assert isinstance(result, pd.DataFrame)
```

- [ ] **Step 2: Run failing tests**

```bash
pytest tests/test_backtest.py::test_run_backtest_explicit_ev_threshold_filters_bets tests/test_backtest.py::test_run_backtest_explicit_min_prob_edge_filters_bets tests/test_backtest.py::test_run_backtest_defaults_unchanged -v
```

Expected: 2 FAILED (`unexpected keyword argument 'ev_threshold'`), 1 PASSED (`defaults_unchanged`)

- [ ] **Step 3: Update `run_backtest()` signature and filtering logic**

Replace the `run_backtest` function in `src/mlb_edge_finder/backtest.py` with this updated version (keep `simulate_market_odds` and `compute_summary` unchanged):

```python
def run_backtest(
    clf: Any,
    training_df: pd.DataFrame,
    home_market_prob: float = 0.5,
    vig: float = 0.0476,
    unit: float = 100.0,
    ev_threshold: float | None = None,
    min_prob_edge: float | None = None,
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
        min_prob_edge: Minimum gap between model_prob and market_implied_prob.
            Defaults to config.MIN_PROB_EDGE.

    Returns:
        DataFrame sorted by game_date with columns: game_date, home_name, away_name,
        bet_side, american_odds, model_prob, ev, kelly_fraction, actual_home_win,
        won, pnl, cumulative_pnl. Returns empty DataFrame (with those columns) if
        no bets clear the thresholds.
    """
    from sklearn.model_selection import train_test_split

    from mlb_edge_finder import config as _config
    from mlb_edge_finder.edge_finder import compute_ev, compute_kelly, market_implied_prob
    from mlb_edge_finder.model import NON_FEATURE_COLS, TARGET_COL

    _ev_threshold = ev_threshold if ev_threshold is not None else _config.EV_THRESHOLD
    _min_prob_edge = min_prob_edge if min_prob_edge is not None else _config.MIN_PROB_EDGE

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

    home_odds_f, away_odds_f = simulate_market_odds(home_market_prob, vig)
    home_odds_i = round(home_odds_f)
    away_odds_i = round(away_odds_f)
    home_payout = home_odds_i / 100 if home_odds_i > 0 else 100 / abs(home_odds_i)
    away_payout = away_odds_i / 100 if away_odds_i > 0 else 100 / abs(away_odds_i)

    home_market_implied = market_implied_prob(home_odds_i)
    away_market_implied = market_implied_prob(away_odds_i)

    meta = training_df.loc[X_test.index, ["game_date", "home_name", "away_name"]]

    records = []
    for (idx, prob), actual in zip(zip(X_test.index, home_probs), y_test.values):
        row_meta = meta.loc[idx]

        home_ev = compute_ev(float(prob), home_odds_i)
        home_edge = float(prob) - home_market_implied
        if (home_ev > _ev_threshold
                and home_odds_i >= _config.MIN_AMERICAN_ODDS
                and home_edge > _min_prob_edge):
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
        away_edge = away_prob - away_market_implied
        if (away_ev > _ev_threshold
                and away_odds_i >= _config.MIN_AMERICAN_ODDS
                and away_edge > _min_prob_edge):
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
        logger.warning("No edges found in backtest at EV=%.0f%% edge=%.0f%%",
                       _ev_threshold * 100, _min_prob_edge * 100)
        return pd.DataFrame(columns=output_cols)

    result = pd.DataFrame(records).sort_values("game_date").reset_index(drop=True)
    result["cumulative_pnl"] = result["pnl"].cumsum()
    logger.info(
        "Backtest complete: %d bets across %d test games (EV=%.0f%% edge=%.0f%%)",
        len(result), len(X_test), _ev_threshold * 100, _min_prob_edge * 100,
    )
    return result
```

- [ ] **Step 4: Run new tests**

```bash
pytest tests/test_backtest.py::test_run_backtest_explicit_ev_threshold_filters_bets tests/test_backtest.py::test_run_backtest_explicit_min_prob_edge_filters_bets tests/test_backtest.py::test_run_backtest_defaults_unchanged -v
```

Expected: 3 PASSED

- [ ] **Step 5: Run full backtest regression suite**

```bash
pytest tests/test_backtest.py -v
```

Expected: all previously passing tests still PASS.

- [ ] **Step 6: Commit**

```bash
git add src/mlb_edge_finder/backtest.py tests/test_backtest.py
git commit -m "feat: add ev_threshold and min_prob_edge params to run_backtest"
```

---

## Task 4: `sweep_thresholds()` in `backtest.py`

**Files:**
- Modify: `src/mlb_edge_finder/backtest.py`
- Modify: `tests/test_backtest.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_backtest.py`:

```python
from mlb_edge_finder.backtest import sweep_thresholds


def test_sweep_thresholds_returns_dataframe():
    df = _make_training_df(200)
    clf = _make_mock_clf(home_win_prob=0.65)
    result = sweep_thresholds(clf, df, ev_low=0.05, ev_high=0.15, ev_step=0.05,
                               prob_edge_low=0.0, prob_edge_high=0.10, prob_edge_step=0.05)
    assert isinstance(result, pd.DataFrame)


def test_sweep_thresholds_output_columns():
    df = _make_training_df(200)
    clf = _make_mock_clf(home_win_prob=0.65)
    result = sweep_thresholds(clf, df, ev_low=0.05, ev_high=0.15, ev_step=0.05,
                               prob_edge_low=0.0, prob_edge_high=0.10, prob_edge_step=0.05)
    expected = {"ev_threshold", "min_prob_edge", "n_bets", "win_rate",
                "roi_pct", "sharpe_ratio", "avg_bets_per_day"}
    assert expected.issubset(set(result.columns))


def test_sweep_thresholds_sorted_by_sharpe_descending():
    df = _make_training_df(200)
    clf = _make_mock_clf(home_win_prob=0.65)
    result = sweep_thresholds(clf, df, ev_low=0.05, ev_high=0.15, ev_step=0.05,
                               prob_edge_low=0.0, prob_edge_high=0.10, prob_edge_step=0.05)
    if len(result) > 1:
        assert result["sharpe_ratio"].iloc[0] >= result["sharpe_ratio"].iloc[1]


def test_sweep_thresholds_excludes_zero_bet_combinations():
    """Threshold combinations that produce 0 bets should not appear in output."""
    df = _make_training_df(200)
    clf = _make_mock_clf(home_win_prob=0.65)
    # ev_high=0.99 — very few or no bets at such a high threshold
    result = sweep_thresholds(clf, df, ev_low=0.95, ev_high=0.99, ev_step=0.05,
                               prob_edge_low=0.0, prob_edge_high=0.0, prob_edge_step=0.05)
    # Either empty (all filtered) or every row has n_bets > 0
    if not result.empty:
        assert (result["n_bets"] > 0).all()


def test_sweep_thresholds_best_row_logged(caplog):
    """sweep_thresholds logs the optimal threshold combination."""
    import logging
    df = _make_training_df(200)
    clf = _make_mock_clf(home_win_prob=0.65)
    with caplog.at_level(logging.INFO, logger="mlb_edge_finder.backtest"):
        sweep_thresholds(clf, df, ev_low=0.05, ev_high=0.10, ev_step=0.05,
                         prob_edge_low=0.0, prob_edge_high=0.05, prob_edge_step=0.05)
    assert any("Optimal" in r.message for r in caplog.records)
```

- [ ] **Step 2: Run failing tests**

```bash
pytest tests/test_backtest.py::test_sweep_thresholds_returns_dataframe tests/test_backtest.py::test_sweep_thresholds_output_columns tests/test_backtest.py::test_sweep_thresholds_sorted_by_sharpe_descending tests/test_backtest.py::test_sweep_thresholds_excludes_zero_bet_combinations tests/test_backtest.py::test_sweep_thresholds_best_row_logged -v
```

Expected: 5 errors — `ImportError: cannot import name 'sweep_thresholds'`

- [ ] **Step 3: Implement `sweep_thresholds()`**

Add to the end of `src/mlb_edge_finder/backtest.py`:

```python
def sweep_thresholds(
    clf: Any,
    training_df: pd.DataFrame,
    ev_low: float = 0.05,
    ev_high: float = 0.50,
    ev_step: float = 0.05,
    prob_edge_low: float = 0.00,
    prob_edge_high: float = 0.30,
    prob_edge_step: float = 0.05,
    unit: float = 100.0,
) -> pd.DataFrame:
    """Sweep (ev_threshold, min_prob_edge) combinations and rank by Sharpe ratio.

    Runs run_backtest() at each combination using the synthetic -110/-110 market.
    Combinations that produce 0 bets are excluded from results.

    Args:
        clf: Fitted calibrated classifier.
        training_df: Full training DataFrame from training_data.load_training_set().
        ev_low: Minimum EV threshold to sweep (inclusive). Default 0.05.
        ev_high: Maximum EV threshold to sweep (inclusive). Default 0.50.
        ev_step: Step size for EV threshold sweep. Default 0.05.
        prob_edge_low: Minimum prob-edge to sweep (inclusive). Default 0.00.
        prob_edge_high: Maximum prob-edge to sweep (inclusive). Default 0.30.
        prob_edge_step: Step size for prob-edge sweep. Default 0.05.
        unit: Dollar bet size passed to run_backtest(). Default $100.

    Returns:
        DataFrame sorted by sharpe_ratio descending with columns:
        ev_threshold, min_prob_edge, n_bets, win_rate, roi_pct,
        sharpe_ratio, avg_bets_per_day.
        Empty DataFrame if all combinations produce 0 bets.

    Raises:
        RuntimeError: If every combination produces 0 bets (model is broken).
    """
    import numpy as np

    ev_values = [round(v, 4) for v in
                 list(float(x) for x in
                      [ev_low + i * ev_step for i in
                       range(round((ev_high - ev_low) / ev_step) + 1)])]
    edge_values = [round(v, 4) for v in
                   list(float(x) for x in
                        [prob_edge_low + i * prob_edge_step for i in
                         range(round((prob_edge_high - prob_edge_low) / prob_edge_step) + 1)])]

    total = len(ev_values) * len(edge_values)
    logger.info("Starting threshold sweep: %d combinations", total)

    n_unique_dates = training_df["game_date"].nunique() * 0.2  # approx test-set dates

    rows = []
    for i, (ev_t, edge_t) in enumerate(
        (ev, edge) for ev in ev_values for edge in edge_values
    ):
        if i > 0 and i % 10 == 0:
            logger.info("Threshold sweep: %d/%d complete", i, total)

        bt = run_backtest(clf, training_df, ev_threshold=ev_t,
                          min_prob_edge=edge_t, unit=unit)
        if bt.empty:
            logger.debug(
                "Skipping EV=%.0f%% edge=%.0f%% — no bets at this threshold",
                ev_t * 100, edge_t * 100,
            )
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
            "min_prob_edge": edge_t,
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
        "Optimal: EV=%.0f%% MIN_PROB_EDGE=%.0f%% Sharpe=%.3f (%d bets, %.1f/day)",
        best["ev_threshold"] * 100,
        best["min_prob_edge"] * 100,
        best["sharpe_ratio"],
        best["n_bets"],
        best["avg_bets_per_day"],
    )
    return result
```

- [ ] **Step 4: Run new tests**

```bash
pytest tests/test_backtest.py::test_sweep_thresholds_returns_dataframe tests/test_backtest.py::test_sweep_thresholds_output_columns tests/test_backtest.py::test_sweep_thresholds_sorted_by_sharpe_descending tests/test_backtest.py::test_sweep_thresholds_excludes_zero_bet_combinations tests/test_backtest.py::test_sweep_thresholds_best_row_logged -v
```

Expected: 5 PASSED

- [ ] **Step 5: Run full test suite**

```bash
pytest tests/ -v
```

Expected: all 158+ tests PASS. Note the count — it will be higher after Tasks 1–4.

- [ ] **Step 6: Commit**

```bash
git add src/mlb_edge_finder/backtest.py tests/test_backtest.py
git commit -m "feat: add sweep_thresholds() to backtest"
```

---

## Task 5: Run the sweep, update `config.py`

**Files:**
- Modify: `src/mlb_edge_finder/config.py`

This task runs the sweep interactively and hard-codes the result. No new tests needed — the config values are validated by the existing suite.

- [ ] **Step 1: Run the sweep**

```bash
cd /Users/jaydengould/Documents/projects/mlb-edge-finder
python3 - << 'EOF'
import logging
from mlb_edge_finder.config import setup_logging
setup_logging()

from mlb_edge_finder.training_data import load_training_set
from mlb_edge_finder.model import load_model
from mlb_edge_finder.backtest import sweep_thresholds

training_df = load_training_set([2019, 2021, 2022, 2023, 2024, 2025])
clf = load_model()  # auto-discovers latest model

result = sweep_thresholds(clf, training_df)
print(result.head(15).to_string(index=False))
print(f"\nBest EV_THRESHOLD:  {result.iloc[0]['ev_threshold']}")
print(f"Best MIN_PROB_EDGE: {result.iloc[0]['min_prob_edge']}")
EOF
```

Wait for it to finish (70 backtest runs over ~3,000 test games — should take under 60s).

- [ ] **Step 2: Record the winner**

Note the printed `Best EV_THRESHOLD` and `Best MIN_PROB_EDGE` values from the output above.

- [ ] **Step 3: Update `config.py` with the winning values**

In `src/mlb_edge_finder/config.py`, update the two threshold lines (replace the placeholder values with the actual sweep results):

```python
EV_THRESHOLD: float = <best_ev_threshold>   # Sharpe-optimal from threshold sweep 2026-05-24
MIN_PROB_EDGE: float = <best_min_prob_edge>  # Sharpe-optimal from threshold sweep 2026-05-24
```

For example, if the sweep returned EV=0.15 and edge=0.10:
```python
EV_THRESHOLD: float = 0.15
MIN_PROB_EDGE: float = 0.10
```

- [ ] **Step 4: Run full test suite to confirm nothing broke**

```bash
pytest tests/ -v
```

Expected: all tests PASS. (Tests that patch `config.EV_THRESHOLD` are unaffected because they override the value at call time.)

- [ ] **Step 5: Commit**

```bash
git add src/mlb_edge_finder/config.py
git commit -m "config: set Sharpe-optimal EV_THRESHOLD and MIN_PROB_EDGE from threshold sweep"
```

---

## Task 6: Update `notebooks/02_backtest.ipynb`

**Files:**
- Modify: `notebooks/02_backtest.ipynb`

- [ ] **Step 1: Open the notebook and run all existing cells**

```bash
cd /Users/jaydengould/Documents/projects/mlb-edge-finder
jupyter nbconvert --to notebook --execute --inplace notebooks/02_backtest.ipynb
```

Expected: exits 0. If any cell errors, investigate before continuing.

- [ ] **Step 2: Add threshold sweep section**

Append the following cells to `notebooks/02_backtest.ipynb` using `NotebookEdit` or by opening Jupyter and inserting manually. Add a markdown cell then a code cell:

Markdown cell:
```markdown
## Threshold Sweep — Finding Sharpe-Optimal Filters

Sweeps EV threshold (5%–50%) × MIN_PROB_EDGE (0%–30%) over the synthetic backtest
to find the combination with the best risk-adjusted return (Sharpe ratio).
```

Code cell:
```python
from mlb_edge_finder.backtest import sweep_thresholds

sweep_results = sweep_thresholds(clf_calibrated, training_df)
print(f"Top 10 threshold combinations by Sharpe ratio:")
print(sweep_results.head(10).to_string(index=False))

best = sweep_results.iloc[0]
print(f"\n✓ Optimal: EV_THRESHOLD={best['ev_threshold']:.0%}  "
      f"MIN_PROB_EDGE={best['min_prob_edge']:.0%}  "
      f"Sharpe={best['sharpe_ratio']:.3f}  "
      f"Bets/day={best['avg_bets_per_day']:.1f}")
```

- [ ] **Step 3: Add heatmap cell**

Add a code cell after the sweep results:

```python
import matplotlib.pyplot as plt
import numpy as np

pivot = sweep_results.pivot(index="min_prob_edge", columns="ev_threshold", values="sharpe_ratio")

fig, ax = plt.subplots(figsize=(10, 5))
im = ax.imshow(pivot.values, aspect="auto", origin="lower", cmap="RdYlGn")
ax.set_xticks(range(len(pivot.columns)))
ax.set_xticklabels([f"{v:.0%}" for v in pivot.columns])
ax.set_yticks(range(len(pivot.index)))
ax.set_yticklabels([f"{v:.0%}" for v in pivot.index])
ax.set_xlabel("EV Threshold")
ax.set_ylabel("Min Prob Edge")
ax.set_title("Sharpe Ratio by Threshold Combination")
plt.colorbar(im, ax=ax, label="Sharpe Ratio")

# Mark the best cell
best_ev_idx = list(pivot.columns).index(best["ev_threshold"])
best_edge_idx = list(pivot.index).index(best["min_prob_edge"])
ax.plot(best_ev_idx, best_edge_idx, "k*", markersize=15, label="Optimal")
ax.legend()
plt.tight_layout()
plt.savefig("notebooks/threshold_heatmap.png", dpi=150)
plt.show()
print("Saved: notebooks/threshold_heatmap.png")
```

- [ ] **Step 4: Add optimal-threshold backtest cell**

Add a code cell that re-runs the backtest at the chosen thresholds and prints a clean summary:

```python
from mlb_edge_finder.backtest import run_backtest, compute_summary
import matplotlib.pyplot as plt

bt_optimal = run_backtest(
    clf_calibrated, training_df,
    ev_threshold=best["ev_threshold"],
    min_prob_edge=best["min_prob_edge"],
)

summary_optimal = compute_summary(bt_optimal)
print("=== Optimal-threshold backtest (held-out 20% test split) ===")
for k, v in summary_optimal.items():
    print(f"  {k:20s}: {v}")

fig, axes = plt.subplots(1, 2, figsize=(14, 4))
axes[0].plot(bt_optimal["cumulative_pnl"].values)
axes[0].set_title(f"Cumulative P&L — EV={best['ev_threshold']:.0%} Edge={best['min_prob_edge']:.0%}")
axes[0].set_xlabel("Bet #")
axes[0].set_ylabel("P&L ($)")
axes[0].axhline(0, color="gray", linestyle="--")

axes[1].hist(bt_optimal["pnl"], bins=20, edgecolor="black")
axes[1].set_title("Bet P&L Distribution")
axes[1].set_xlabel("P&L ($)")
axes[1].set_ylabel("Count")

plt.tight_layout()
plt.savefig("notebooks/backtest_optimal_pnl.png", dpi=150)
plt.show()
print("Saved: notebooks/backtest_optimal_pnl.png")
```

- [ ] **Step 5: Execute updated notebook end-to-end**

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/02_backtest.ipynb
```

Expected: exits 0, all cells run without error.

- [ ] **Step 6: Commit**

```bash
git add notebooks/02_backtest.ipynb notebooks/threshold_heatmap.png notebooks/backtest_optimal_pnl.png
git commit -m "notebooks: add threshold sweep and optimal-threshold backtest to 02_backtest.ipynb"
```

---

## Task 7: Final regression check and CLAUDE.md update

- [ ] **Step 1: Run the full test suite one last time**

```bash
pytest tests/ -v --tb=short 2>&1 | tail -20
```

Expected: all tests PASS. Note the final count.

- [ ] **Step 2: Update CLAUDE.md**

In `CLAUDE.md`, update the **Current Phase** section and **Roadmap** to reflect:

- Mark this feature complete:
  ```
  - [x] Threshold sweep & market-edge filter — `market_implied_prob()` in `edge_finder.py`, `MIN_PROB_EDGE` in `config.py`, `sweep_thresholds()` in `backtest.py`. Sharpe-optimal `(EV_THRESHOLD, MIN_PROB_EDGE)` found by 70-combination grid search over synthetic backtest and committed to config.
  ```

- In the current phase summary, add:
  - `market_implied_prob(american_odds)` converts bookmaker odds to implied probability (vig-included).
  - `find_edges()` gains `min_prob_edge` parameter (default `config.MIN_PROB_EDGE`); filters bets where `model_prob − market_implied_prob ≤ min_prob_edge`.
  - `run_backtest()` gains `ev_threshold` and `min_prob_edge` parameters (both default to config values).
  - `sweep_thresholds(clf, training_df, ...)` in `backtest.py` — 70-combination `(ev_threshold, min_prob_edge)` grid, returns Sharpe-sorted DataFrame.

- [ ] **Step 3: Final commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md with threshold sweep and market-edge filter"
```
