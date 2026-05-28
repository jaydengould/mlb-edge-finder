# Threshold Rebalance + High-Confidence Badge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Lower EV_THRESHOLD to 0.20, remove MIN_PROB_EDGE entirely, and replace the `prob_flag` warning column with a `high_confidence` positive badge so the dashboard shows regular output and highlights the strongest edges.

**Architecture:** Config drives all threshold logic — four constant changes cascade into edge_finder, backtest, generate_site, and their tests. `find_edges()` loses the `min_prob_edge` parameter entirely; `high_confidence` is computed inline from the already-present `home_implied`/`away_implied` arrays. `sweep_thresholds()` becomes a 1D sweep over `ev_threshold` only.

**Tech Stack:** Python 3.10+, XGBoost, pandas, pytest, Chart.js (dashboard)

---

## Files

| File | Change |
|---|---|
| `src/mlb_edge_finder/config.py` | Change `EV_THRESHOLD`, remove `MIN_PROB_EDGE`, add `HIGH_CONFIDENCE_EV` + `HIGH_CONFIDENCE_PROB_EDGE` |
| `src/mlb_edge_finder/edge_finder.py` | Remove `min_prob_edge` param/filter, replace `prob_flag` with `high_confidence` |
| `src/mlb_edge_finder/backtest.py` | Remove `min_prob_edge` from `run_backtest`, simplify `sweep_thresholds` to 1D |
| `src/mlb_edge_finder/generate_site.py` | Replace ⚠ `prob_flag` badge with ★ `high_confidence` inline prefix, drop Flag column |
| `tests/test_edge_finder.py` | Drop `MIN_PROB_EDGE` patches, replace 3 `prob_flag` tests with 2 `high_confidence` tests, update column assertions |
| `tests/test_backtest.py` | Remove `min_prob_edge` filter test, update sweep column assertion |
| `tests/test_generate_site.py` | Update all `prob_flag` references to `high_confidence`, update badge assertion |
| `notebooks/02_backtest.ipynb` | Update column references and sweep output display |

---

## Task 1: Update config constants

**Files:**
- Modify: `src/mlb_edge_finder/config.py:28-31`

- [ ] **Step 1: Replace the edge-finding threshold block**

Replace the current block:
```python
# --- Edge-finding thresholds ---
EV_THRESHOLD: float = 0.50  # Sharpe-optimal from threshold sweep 2026-05-24
MIN_AMERICAN_ODDS: int = -300
MIN_PROB_EDGE: float = 0.30  # Sharpe-optimal from threshold sweep 2026-05-24
RETRAIN_THRESHOLD: int = 15  # retrain after this many new games since last model date
```

With:
```python
# --- Edge-finding thresholds ---
EV_THRESHOLD: float = 0.20
MIN_AMERICAN_ODDS: int = -300
HIGH_CONFIDENCE_EV: float = 0.40
HIGH_CONFIDENCE_PROB_EDGE: float = 0.15
RETRAIN_THRESHOLD: int = 15  # retrain after this many new games since last model date
```

- [ ] **Step 2: Verify the constants load correctly**

```bash
cd /Users/jaydengould/Documents/projects/mlb-edge-finder
python -c "from mlb_edge_finder.config import EV_THRESHOLD, HIGH_CONFIDENCE_EV, HIGH_CONFIDENCE_PROB_EDGE, MIN_AMERICAN_ODDS; print(EV_THRESHOLD, HIGH_CONFIDENCE_EV, HIGH_CONFIDENCE_PROB_EDGE, MIN_AMERICAN_ODDS)"
```

Expected output: `0.2 0.4 0.15 -300`

Also verify `MIN_PROB_EDGE` is gone:
```bash
python -c "from mlb_edge_finder import config; print(hasattr(config, 'MIN_PROB_EDGE'))"
```

Expected output: `False`

- [ ] **Step 3: Commit**

```bash
git add src/mlb_edge_finder/config.py
git commit -m "feat: lower EV_THRESHOLD to 0.20, remove MIN_PROB_EDGE, add HIGH_CONFIDENCE_EV/PROB_EDGE constants"
```

---

## Task 2: Write failing tests for edge_finder changes

**Files:**
- Modify: `tests/test_edge_finder.py`

- [ ] **Step 1: Remove all MIN_PROB_EDGE patches and update output column assertions**

There are four tests that patch `MIN_PROB_EDGE` and three that assert the output columns include `prob_flag`. Make all seven changes now so they reflect the new world — these will fail until `edge_finder.py` is updated in Task 3.

In `test_find_edges_returns_home_edge` (around line 113): remove the `patch("mlb_edge_finder.edge_finder.config.MIN_PROB_EDGE", 0.0)` line and update the `with` block, and change `"prob_flag"` → `"high_confidence"` in the column assertion:

```python
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
    assert set(result.columns) == {
        "game_id", "home_team", "away_team", "bet_side",
        "american_odds", "model_prob", "ev", "kelly_fraction", "high_confidence",
    }
```

In `test_find_edges_filters_min_odds` (around line 130): change `"prob_flag"` → `"high_confidence"` in the column assertion only (no MIN_PROB_EDGE patch in this test):

```python
    assert set(result.columns) == {
        "game_id", "home_team", "away_team", "bet_side",
        "american_odds", "model_prob", "ev", "kelly_fraction", "high_confidence",
    }
```

In `test_find_edges_empty_when_no_edges` (around line 150): change `"prob_flag"` → `"high_confidence"` in the column assertion:

```python
    assert set(result.columns) == {
        "game_id", "home_team", "away_team", "bet_side",
        "american_odds", "model_prob", "ev", "kelly_fraction", "high_confidence",
    }
```

In `test_find_edges_both_sides` (around line 169): remove the `patch("mlb_edge_finder.edge_finder.config.MIN_PROB_EDGE", 0.0)` line:

```python
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
```

In `test_find_edges_includes_kelly_fraction` (around line 236): remove the `patch("mlb_edge_finder.edge_finder.config.MIN_PROB_EDGE", 0.0)` line:

```python
    with patch("mlb_edge_finder.edge_finder.config.DATA_PROCESSED_DIR", tmp_path), \
         patch("mlb_edge_finder.edge_finder.config.EV_THRESHOLD", 0.05), \
         patch("mlb_edge_finder.edge_finder.config.MIN_AMERICAN_ODDS", -300):
        result = find_edges(features_df, clf, GAME_DATE)
```

- [ ] **Step 2: Replace the three prob_flag tests with two high_confidence tests**

Delete `test_find_edges_prob_flag_true_when_model_prob_above_0_80`, `test_find_edges_prob_flag_false_when_model_prob_at_or_below_0_80`, and `test_find_edges_prob_flag_boundary_exactly_0_80` (lines 268–313). Replace them with:

```python
def test_find_edges_high_confidence_true_when_both_thresholds_met(tmp_path):
    """high_confidence=True when EV > HIGH_CONFIDENCE_EV and prob_gap > HIGH_CONFIDENCE_PROB_EDGE."""
    from mlb_edge_finder.edge_finder import find_edges
    # home_prob=0.75, home_odds=+110
    # EV = 0.75*1.10 - 0.25 = 0.575 > 0.40 ✓
    # market_implied(+110) = 100/210 ≈ 0.476
    # prob_gap = 0.75 - 0.476 = 0.274 > 0.15 ✓
    features_df = _make_features(home_odds=110, away_odds=-140)
    clf = _make_clf(home_proba=0.75)

    with patch("mlb_edge_finder.edge_finder.config.DATA_PROCESSED_DIR", tmp_path), \
         patch("mlb_edge_finder.edge_finder.config.EV_THRESHOLD", 0.05), \
         patch("mlb_edge_finder.edge_finder.config.MIN_AMERICAN_ODDS", -300), \
         patch("mlb_edge_finder.edge_finder.config.HIGH_CONFIDENCE_EV", 0.40), \
         patch("mlb_edge_finder.edge_finder.config.HIGH_CONFIDENCE_PROB_EDGE", 0.15):
        result = find_edges(features_df, clf, GAME_DATE)

    assert len(result) == 1
    assert result.iloc[0]["high_confidence"] == True


def test_find_edges_high_confidence_false_when_ev_below_badge_threshold(tmp_path):
    """high_confidence=False when EV does not exceed HIGH_CONFIDENCE_EV."""
    from mlb_edge_finder.edge_finder import find_edges
    # home_prob=0.55, home_odds=+110
    # EV = 0.55*1.10 - 0.45 = 0.155 — passes EV_THRESHOLD=0.05 but < HIGH_CONFIDENCE_EV=0.40
    # market_implied(+110) = 100/210 ≈ 0.476
    # prob_gap = 0.55 - 0.476 = 0.074 < 0.15 ✗
    features_df = _make_features(home_odds=110, away_odds=-140)
    clf = _make_clf(home_proba=0.55)

    with patch("mlb_edge_finder.edge_finder.config.DATA_PROCESSED_DIR", tmp_path), \
         patch("mlb_edge_finder.edge_finder.config.EV_THRESHOLD", 0.05), \
         patch("mlb_edge_finder.edge_finder.config.MIN_AMERICAN_ODDS", -300), \
         patch("mlb_edge_finder.edge_finder.config.HIGH_CONFIDENCE_EV", 0.40), \
         patch("mlb_edge_finder.edge_finder.config.HIGH_CONFIDENCE_PROB_EDGE", 0.15):
        result = find_edges(features_df, clf, GAME_DATE)

    assert len(result) == 1
    assert result.iloc[0]["high_confidence"] == False
```

- [ ] **Step 3: Run tests to verify they fail for the right reason**

```bash
cd /Users/jaydengould/Documents/projects/mlb-edge-finder
pytest tests/test_edge_finder.py -v 2>&1 | tail -30
```

Expected: failures referencing `high_confidence` column not found and `prob_flag` column missing. Tests for `MIN_PROB_EDGE` patches may also error with `AttributeError` since the config constant no longer exists.

---

## Task 3: Implement edge_finder.py changes

**Files:**
- Modify: `src/mlb_edge_finder/edge_finder.py`

- [ ] **Step 1: Update find_edges signature and output_cols**

Remove the `min_prob_edge` parameter from the function signature and update `output_cols`:

```python
def find_edges(
    features_df: pd.DataFrame,
    clf: XGBClassifier,
    game_date: date,
) -> pd.DataFrame:
```

Update `output_cols` (line ~116):
```python
output_cols = [
    "game_id", "home_team", "away_team", "bet_side",
    "american_odds", "model_prob", "ev", "kelly_fraction", "high_confidence",
]
```

Remove the docstring lines for `min_prob_edge` and update the Returns section to say `high_confidence` instead of `prob_flag`. The new docstring for the relevant section:
```
        min_prob_edge filter is removed — EV_THRESHOLD already implies a sufficient
        prob gap on standard MLB lines.

    Returns:
        DataFrame with columns: game_id, home_team, away_team,
        bet_side, american_odds, model_prob, ev, kelly_fraction, high_confidence —
        one row per flagged edge. high_confidence=True when both EV > config.HIGH_CONFIDENCE_EV
        and (model_prob - market_implied_prob) > config.HIGH_CONFIDENCE_PROB_EDGE.
```

Also remove these two lines near the top of the function body (around line 113):
```python
    if min_prob_edge is None:
        min_prob_edge = config.MIN_PROB_EDGE
```

- [ ] **Step 2: Update the home pass mask and badge**

The home mask currently reads:
```python
home_mask = (
    (home_ev > config.EV_THRESHOLD)
    & (df["home_odds_american"] >= config.MIN_AMERICAN_ODDS)
    & ((home_prob - home_implied) > min_prob_edge)
)
```

Replace with (remove the third condition):
```python
home_mask = (
    (home_ev > config.EV_THRESHOLD)
    & (df["home_odds_american"] >= config.MIN_AMERICAN_ODDS)
)
```

Then replace:
```python
home_edges["prob_flag"] = home_prob[home_mask.values] > 0.80
```

With:
```python
home_edges["high_confidence"] = (
    (home_ev[home_mask].values > config.HIGH_CONFIDENCE_EV)
    & ((home_prob[home_mask.values] - home_implied[home_mask.values]) > config.HIGH_CONFIDENCE_PROB_EDGE)
)
```

- [ ] **Step 3: Update the away pass mask and badge**

The away mask currently reads:
```python
away_mask = (
    (away_ev > config.EV_THRESHOLD)
    & (df["away_odds_american"] >= config.MIN_AMERICAN_ODDS)
    & ((away_prob - away_implied) > min_prob_edge)
)
```

Replace with:
```python
away_mask = (
    (away_ev > config.EV_THRESHOLD)
    & (df["away_odds_american"] >= config.MIN_AMERICAN_ODDS)
)
```

Then replace:
```python
away_edges["prob_flag"] = away_prob[away_mask.values] > 0.80
```

With:
```python
away_edges["high_confidence"] = (
    (away_ev[away_mask].values > config.HIGH_CONFIDENCE_EV)
    & ((away_prob[away_mask.values] - away_implied[away_mask.values]) > config.HIGH_CONFIDENCE_PROB_EDGE)
)
```

- [ ] **Step 4: Update the log message**

Find the log line that references `min_prob_edge * 100` (around line 183):
```python
    logger.info(
        "prob-edge filter (%.0f%%): %d edges kept after all filters for %s",
        min_prob_edge * 100,
        len(edges),
        game_date,
    )
```

Replace with:
```python
    logger.info(
        "%d edge(s) found for %s (EV_THRESHOLD=%.2f)",
        len(edges),
        game_date,
        config.EV_THRESHOLD,
    )
```

- [ ] **Step 5: Run edge_finder tests to verify they pass**

```bash
cd /Users/jaydengould/Documents/projects/mlb-edge-finder
pytest tests/test_edge_finder.py -v 2>&1 | tail -20
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/mlb_edge_finder/edge_finder.py tests/test_edge_finder.py
git commit -m "feat: replace prob_flag with high_confidence badge, remove min_prob_edge filter"
```

---

## Task 4: Update backtest.py and its tests

**Files:**
- Modify: `src/mlb_edge_finder/backtest.py`
- Modify: `tests/test_backtest.py`

- [ ] **Step 1: Write a failing test for the updated sweep_thresholds columns**

In `tests/test_backtest.py`, find the test that asserts sweep output columns (around line 270). The current assertion includes `"min_prob_edge"`. Update it:

```python
def test_sweep_thresholds_returns_sorted_dataframe():
    """sweep_thresholds returns a DataFrame sorted by sharpe_ratio with expected columns."""
    from mlb_edge_finder.backtest import sweep_thresholds
    df = _make_training_df(300)
    clf = _make_mock_clf(home_win_prob=0.65)
    result = sweep_thresholds(clf, df, ev_low=0.05, ev_high=0.15, ev_step=0.05)
    assert isinstance(result, pd.DataFrame)
    assert not result.empty
    expected = {"ev_threshold", "n_bets", "win_rate",
                "roi_pct", "sharpe_ratio", "avg_bets_per_day"}
    assert expected.issubset(set(result.columns))
    assert "min_prob_edge" not in result.columns
    assert result["sharpe_ratio"].is_monotonic_decreasing or len(result) == 1
```

Also delete `test_run_backtest_explicit_min_prob_edge_filters_bets` (around line 158–164) — this test verifies a filter that no longer exists.

- [ ] **Step 2: Run the backtest tests to verify the sweep test fails**

```bash
cd /Users/jaydengould/Documents/projects/mlb-edge-finder
pytest tests/test_backtest.py::test_sweep_thresholds_returns_sorted_dataframe -v
```

Expected: FAIL — `sweep_thresholds` still has `min_prob_edge` in its output.

- [ ] **Step 3: Update run_backtest in backtest.py**

Remove `min_prob_edge: float | None = None` from the signature and all related logic. The updated function signature and relevant section:

```python
def run_backtest(
    clf: Any,
    training_df: pd.DataFrame,
    home_market_prob: float = 0.5,
    vig: float = 0.0476,
    unit: float = 100.0,
    ev_threshold: float | None = None,
) -> pd.DataFrame:
```

Remove this line from the function body:
```python
    _min_prob_edge = min_prob_edge if min_prob_edge is not None else _config.MIN_PROB_EDGE
```

Remove `home_edge` and `away_edge` variables and their filter conditions. The home check becomes:
```python
        home_ev = compute_ev(float(prob), home_odds_i)
        if (home_ev > _ev_threshold
                and home_odds_i >= _config.MIN_AMERICAN_ODDS):
```

The away check becomes:
```python
        away_ev = compute_ev(away_prob, away_odds_i)
        if (away_ev > _ev_threshold
                and away_odds_i >= _config.MIN_AMERICAN_ODDS):
```

Update the warning log message (around line 164):
```python
        logger.warning("No edges found in backtest at EV=%.0f%%", _ev_threshold * 100)
```

Update the info log message (around line 171):
```python
    logger.info(
        "Backtest complete: %d bets across %d test games (EV=%.0f%%)",
        len(result), len(X_test), _ev_threshold * 100,
    )
```

- [ ] **Step 4: Update sweep_thresholds to 1D**

Replace the entire `sweep_thresholds` function with:

```python
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
```

- [ ] **Step 5: Run backtest tests to verify they pass**

```bash
cd /Users/jaydengould/Documents/projects/mlb-edge-finder
pytest tests/test_backtest.py -v 2>&1 | tail -20
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/mlb_edge_finder/backtest.py tests/test_backtest.py
git commit -m "feat: remove min_prob_edge from run_backtest, simplify sweep_thresholds to 1D"
```

---

## Task 5: Update generate_site.py and its tests

**Files:**
- Modify: `src/mlb_edge_finder/generate_site.py`
- Modify: `tests/test_generate_site.py`

- [ ] **Step 1: Write a failing test for the high_confidence badge**

In `tests/test_generate_site.py`, replace `test_generate_prob_flag_shows_warning_badge` (around line 167) with:

```python
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
    generate(outputs_dir=outputs_dir, metrics_path=None, pnl_path=None, out_path=out)
    html = out.read_text()
    assert "★" in html
    assert "⚠" not in html
```

Also update all fixture dicts in `tests/test_generate_site.py` that use `"prob_flag": False` → `"high_confidence": False` and `"prob_flag": True` → `"high_confidence": True`. Find them with:

```bash
grep -n "prob_flag" tests/test_generate_site.py
```

Update the `_write_edges_csv` column list at the top of the file and all dict literals to use `high_confidence`.

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd /Users/jaydengould/Documents/projects/mlb-edge-finder
pytest tests/test_generate_site.py::test_generate_high_confidence_shows_star_badge -v
```

Expected: FAIL — `★` not found in HTML.

- [ ] **Step 3: Update generate_site.py**

In `_render_edges_html` (around line 128), replace the `prob_flag` block:

```python
        prob_flag = r.get("prob_flag")
        flagged = prob_flag is True or str(prob_flag).strip() == "True"
        flag = '<span title="Model probability >80% — review manually">⚠</span>' if flagged else ""
```

With:
```python
        high_conf = r.get("high_confidence")
        is_high_conf = high_conf is True or str(high_conf).strip() == "True"
        badge = "★ " if is_high_conf else ""
```

Update the row template to use the badge inline on the bet_side cell and remove the separate flag cell:

```python
        rows_html += (
            f"<tr>"
            f"<td>{home} vs {away}</td>"
            f'<td><span class="side-badge {sc}">{badge}{side}</span></td>'
            f"<td>{odds_str}</td>"
            f"<td>{model_prob * 100:.1f}%</td>"
            f'<td class="ev-val">{ev_str}</td>'
            f"<td>{kelly * 100:.1f}%</td>"
            f"</tr>"
        )
```

Update the table header (line 154–156) to remove the "Flag" column:

```python
    return (
        "<table><thead><tr>"
        "<th>Matchup</th><th>Side</th><th>Odds</th>"
        "<th>Model Prob</th><th>EV</th><th>Kelly</th>"
        f"</tr></thead><tbody>{rows_html}</tbody></table>"
    )
```

- [ ] **Step 4: Run generate_site tests**

```bash
cd /Users/jaydengould/Documents/projects/mlb-edge-finder
pytest tests/test_generate_site.py -v 2>&1 | tail -20
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/mlb_edge_finder/generate_site.py tests/test_generate_site.py
git commit -m "feat: replace prob_flag warning badge with high_confidence star badge on dashboard"
```

---

## Task 6: Full test suite + notebook update + final commit

**Files:**
- Modify: `notebooks/02_backtest.ipynb`

- [ ] **Step 1: Run the full test suite**

```bash
cd /Users/jaydengould/Documents/projects/mlb-edge-finder
pytest tests/ -v 2>&1 | tail -30
```

Expected: all tests pass. Note the new total count (should be ~210, net change ≈ -3 deleted tests +2 new high_confidence tests -1 deleted backtest test).

- [ ] **Step 2: Update notebooks/02_backtest.ipynb**

Open the notebook and make these changes:

1. Any cell that references `prob_flag` column from `run_backtest()` output — remove or replace with a note that the column no longer exists in the backtest output (it's an edge_finder output only).

2. The Sharpe heatmap cell that displayed a 2D grid of `(ev_threshold, min_prob_edge)` — replace with a simple bar chart or table showing Sharpe by `ev_threshold` only (1D). The `sweep_thresholds()` result no longer has a `min_prob_edge` column.

3. Any cell calling `sweep_thresholds()` with `prob_edge_low`, `prob_edge_high`, or `prob_edge_step` arguments — remove those keyword arguments.

4. Update the "Optimal thresholds" summary cell to reflect `EV_THRESHOLD=0.20` and remove the `MIN_PROB_EDGE` line.

- [ ] **Step 3: Final commit**

```bash
git add notebooks/02_backtest.ipynb
git commit -m "chore: update backtest notebook — remove min_prob_edge, update sweep to 1D"
```

- [ ] **Step 4: Update CLAUDE.md**

In `CLAUDE.md`, update the config constants table and the threshold sweep entry. Key changes:
- `EV_THRESHOLD` comment: change `0.50 (Sharpe-optimal...)` → `0.20`
- Remove `MIN_PROB_EDGE` line
- Add `HIGH_CONFIDENCE_EV # 0.40` and `HIGH_CONFIDENCE_PROB_EDGE # 0.15`
- Update the threshold sweep bullet to note it's now 1D

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md to reflect threshold rebalance"
```

---

## Task 7: Update README.md

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Read the current README**

```bash
cat /Users/jaydengould/Documents/projects/mlb-edge-finder/README.md
```

- [ ] **Step 2: Update threshold-related sections**

Find and update every mention of the old thresholds. Key areas to update:

1. Any mention of `EV_THRESHOLD=0.50` → `EV_THRESHOLD=0.20`
2. Any mention of `MIN_PROB_EDGE=0.30` or `MIN_PROB_EDGE` → remove the constant, explain that the EV threshold implicitly handles this for normal MLB lines
3. Any mention of `prob_flag` → replace with `high_confidence`. Change framing from "warning flag for suspicious model probabilities" to "positive badge on the strongest edges (EV > 0.40 and model prob gap > 15pp over market)"
4. Any mention of the Sharpe heatmap or 2D threshold sweep → update to describe the 1D sweep over `ev_threshold` only
5. The edge output schema table or column list: `prob_flag (bool)` → `high_confidence (bool)`
6. The config constants table if present: remove `MIN_PROB_EDGE`, add `HIGH_CONFIDENCE_EV` and `HIGH_CONFIDENCE_PROB_EDGE`

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: update README for threshold rebalance and high_confidence badge"
```
