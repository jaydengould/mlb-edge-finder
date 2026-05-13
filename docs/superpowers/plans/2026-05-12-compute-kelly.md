# compute_kelly() Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `compute_kelly(prob, american_odds) -> float` to `edge_finder.py` and wire it into `find_edges()` so every flagged edge gets a `kelly_fraction` column.

**Architecture:** `compute_kelly` is a pure function added after `compute_ev` in `edge_finder.py` — same input shape, same module, no new dependencies. `find_edges` calls it in both the home and away passes, appending the result as a column before building the output DataFrame.

**Tech Stack:** Python 3.10+, pandas, pytest

---

## File Map

| File | Change |
|---|---|
| `src/mlb_edge_finder/edge_finder.py` | Add `compute_kelly`; update `find_edges` output_cols + both passes |
| `tests/test_edge_finder.py` | Add 5 new unit tests; update 4 existing tests to expect `kelly_fraction` in columns |

---

### Task 1: `compute_kelly` — tests first, then implementation

**Files:**
- Modify: `tests/test_edge_finder.py`
- Modify: `src/mlb_edge_finder/edge_finder.py`

- [ ] **Step 1: Write failing tests for `compute_kelly`**

Add these tests at the end of `tests/test_edge_finder.py`:

```python
# ---- compute_kelly tests ----

def test_compute_kelly_signature():
    from mlb_edge_finder import edge_finder
    assert callable(edge_finder.compute_kelly)
    sig = inspect.signature(edge_finder.compute_kelly)
    assert "prob" in sig.parameters
    assert "american_odds" in sig.parameters


def test_compute_kelly_zero_ev_returns_zero():
    """Zero EV → Kelly fraction is 0.0.

    prob=0.60, -150: b=100/150=0.6667, ev=0.60*0.6667-0.40=0.0 → kelly=0.0
    """
    from mlb_edge_finder.edge_finder import compute_kelly
    result = compute_kelly(prob=0.60, american_odds=-150)
    assert abs(result) < 1e-9


def test_compute_kelly_positive_ev_underdog():
    """Positive EV underdog returns correct half-Kelly fraction.

    prob=0.55, +110: b=1.10, ev=0.55*1.10-0.45=0.155
    full_kelly=0.155/1.10=0.14091, half_kelly=0.07045
    """
    from mlb_edge_finder.edge_finder import compute_kelly
    result = compute_kelly(prob=0.55, american_odds=110)
    assert abs(result - 0.0705) < 1e-3


def test_compute_kelly_negative_ev_returns_zero():
    """Negative EV → Kelly fraction is 0.0 (clamp, don't bet).

    prob=0.40, -150: b=0.6667, ev=0.40*0.6667-0.60=-0.333 → kelly=0.0
    """
    from mlb_edge_finder.edge_finder import compute_kelly
    result = compute_kelly(prob=0.40, american_odds=-150)
    assert result == 0.0


def test_compute_kelly_result_in_valid_range():
    """Result is always in [0.0, 1.0] for valid inputs."""
    from mlb_edge_finder.edge_finder import compute_kelly
    for prob, odds in [(0.99, 100), (0.55, 200), (0.60, -120), (0.45, -110)]:
        result = compute_kelly(prob=prob, american_odds=odds)
        assert 0.0 <= result <= 1.0, f"Out of range for prob={prob}, odds={odds}: {result}"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python3 -m pytest tests/test_edge_finder.py::test_compute_kelly_signature tests/test_edge_finder.py::test_compute_kelly_zero_ev_returns_zero tests/test_edge_finder.py::test_compute_kelly_positive_ev_underdog tests/test_edge_finder.py::test_compute_kelly_negative_ev_returns_zero tests/test_edge_finder.py::test_compute_kelly_result_in_valid_range -v
```

Expected: all 5 **FAIL** with `AttributeError: module 'mlb_edge_finder.edge_finder' has no attribute 'compute_kelly'`

- [ ] **Step 3: Implement `compute_kelly` in `edge_finder.py`**

Add this function immediately after `compute_ev` (after line 31, before `def find_edges`):

```python
def compute_kelly(prob: float, american_odds: int) -> float:
    """Compute half-Kelly bet size as a fraction of bankroll.

    Uses the same payout derivation as compute_ev, then applies:
        full_kelly = EV / payout
        half_kelly = full_kelly / 2

    Returns 0.0 for zero or negative EV (no edge, don't bet).
    Result is clamped to [0.0, 1.0].

    Args:
        prob: Model-predicted win probability for the team (0.0 – 1.0).
        american_odds: Bookmaker's American moneyline for the same team.

    Returns:
        Fraction of bankroll to wager (0.0 = no bet, 1.0 = full bankroll).
    """
    if american_odds < 0:
        b = 100 / abs(american_odds)
    else:
        b = american_odds / 100
    ev = prob * b - (1 - prob)
    if ev <= 0:
        return 0.0
    return min(ev / b / 2, 1.0)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python3 -m pytest tests/test_edge_finder.py::test_compute_kelly_signature tests/test_edge_finder.py::test_compute_kelly_zero_ev_returns_zero tests/test_edge_finder.py::test_compute_kelly_positive_ev_underdog tests/test_edge_finder.py::test_compute_kelly_negative_ev_returns_zero tests/test_edge_finder.py::test_compute_kelly_result_in_valid_range -v
```

Expected: all 5 **PASS**

- [ ] **Step 5: Commit**

```bash
git add src/mlb_edge_finder/edge_finder.py tests/test_edge_finder.py
git commit -m "feat: add compute_kelly() half-Kelly bet sizing"
```

---

### Task 2: Integrate `kelly_fraction` into `find_edges`

**Files:**
- Modify: `tests/test_edge_finder.py`
- Modify: `src/mlb_edge_finder/edge_finder.py`

- [ ] **Step 1: Write a failing integration test**

Add this test at the end of `tests/test_edge_finder.py`:

```python
def test_find_edges_includes_kelly_fraction(tmp_path):
    """find_edges output contains kelly_fraction column with a positive value."""
    from mlb_edge_finder.edge_finder import find_edges
    # home_prob=0.75, home_odds=+110 → positive EV → positive Kelly fraction
    features_df = _make_features(home_odds=110, away_odds=-140)
    clf = _make_clf(home_proba=0.75)

    with patch("mlb_edge_finder.edge_finder.config.DATA_PROCESSED_DIR", tmp_path), \
         patch("mlb_edge_finder.edge_finder.config.EV_THRESHOLD", 0.05), \
         patch("mlb_edge_finder.edge_finder.config.MIN_AMERICAN_ODDS", -300):
        result = find_edges(features_df, clf, GAME_DATE)

    assert "kelly_fraction" in result.columns
    assert result.iloc[0]["kelly_fraction"] > 0.0
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python3 -m pytest tests/test_edge_finder.py::test_find_edges_includes_kelly_fraction -v
```

Expected: **FAIL** — `AssertionError: 'kelly_fraction' not in columns`

- [ ] **Step 3: Update `find_edges` in `edge_finder.py`**

Replace the entire `find_edges` function body with the updated version below. The changes are:
1. `output_cols` gains `"kelly_fraction"`
2. Both home and away passes compute and assign `kelly_fraction`

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
        bet_side, american_odds, model_prob, ev, kelly_fraction —
        one row per flagged edge.

    Raises:
        ValueError: If features_df is missing any column in clf.feature_names_in_.
    """
    output_cols = [
        "game_id", "home_team", "away_team", "bet_side",
        "american_odds", "model_prob", "ev", "kelly_fraction",
    ]

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
    home_edges["kelly_fraction"] = [
        compute_kelly(float(p), int(o))
        for p, o in zip(home_prob[home_mask.values], df.loc[home_mask, "home_odds_american"].values)
    ]

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
    away_edges["kelly_fraction"] = [
        compute_kelly(float(p), int(o))
        for p, o in zip(away_prob[away_mask.values], df.loc[away_mask, "away_odds_american"].values)
    ]

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

- [ ] **Step 4: Update the 4 existing tests that check `result.columns`**

In `tests/test_edge_finder.py`, update these four tests to include `"kelly_fraction"` in the expected column sets:

**`test_find_edges_returns_home_edge`** — change the `assert set(result.columns)` line:
```python
assert set(result.columns) == {
    "game_id", "home_team", "away_team", "bet_side",
    "american_odds", "model_prob", "ev", "kelly_fraction",
}
```

**`test_find_edges_filters_min_odds`** — change the `assert set(result.columns)` line:
```python
assert set(result.columns) == {
    "game_id", "home_team", "away_team", "bet_side",
    "american_odds", "model_prob", "ev", "kelly_fraction",
}
```

**`test_find_edges_empty_when_no_edges`** — change the `assert set(result.columns)` line:
```python
assert set(result.columns) == {
    "game_id", "home_team", "away_team", "bet_side",
    "american_odds", "model_prob", "ev", "kelly_fraction",
}
```

**`test_find_edges_missing_feature_column`** — no column assertion in this test, no change needed.

- [ ] **Step 5: Run the full test suite**

```bash
python3 -m pytest tests/test_edge_finder.py -v
```

Expected: all **11 tests PASS** (6 original + 5 new)

- [ ] **Step 6: Run all 116+ tests to confirm no regressions**

```bash
python3 -m pytest tests/ -v
```

Expected: all tests **PASS**

- [ ] **Step 7: Commit**

```bash
git add src/mlb_edge_finder/edge_finder.py tests/test_edge_finder.py
git commit -m "feat: add kelly_fraction column to find_edges() output"
```
