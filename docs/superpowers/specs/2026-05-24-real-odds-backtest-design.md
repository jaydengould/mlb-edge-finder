# Threshold Sweep & Market-Edge Filter — Design Spec

**Date:** 2026-05-24
**Status:** Approved

## Problem

The existing backtest compares model predictions against synthetic −110/−110 market odds, producing inflated results (60.3% win rate, +15.1% ROI). `EV_THRESHOLD = 0.05` (5%) is calibrated against that synthetic baseline, so with 15 real morning games it flags 10–13 bets/day. Real-world results (May 21–23, 5 days) show 34% win rate and −32% ROI.

The Odds API historical endpoint requires a paid plan, so replacing the synthetic backtest with real historical lines is off the table. Instead we address the root cause directly: the EV filter alone is too permissive, and doesn't require that the model genuinely disagrees with the real bookmaker line by a meaningful margin.

## Goal

1. Add a `MIN_PROB_EDGE` filter: only flag a bet when `model_prob − market_implied_prob > MIN_PROB_EDGE`. This measures real disagreement between model and market using the bookmaker odds already present in the daily pipeline.
2. Sweep `EV_THRESHOLD` (5%–50%) and `MIN_PROB_EDGE` (0%–30%) over the synthetic backtest to find the Sharpe-optimal combination.
3. Apply the winning `(EV_THRESHOLD, MIN_PROB_EDGE)` pair to `config.py` and commit.
4. Validate the chosen thresholds against the 5 real-world days already collected (May 19–24).

## Out of Scope

- Fetching real historical odds (requires paid API plan).
- Changes to model training or feature engineering.
- UI or dashboard changes.

---

## Architecture

### New helper: `market_implied_prob(american_odds) -> float`

Added to `edge_finder.py` alongside `compute_ev` and `compute_kelly`. Converts a bookmaker's American odds to a raw implied probability (vig included — no vig-stripping needed for a relative comparison):

```python
# Negative odds (favourite): -110 → 110/210 = 52.4%
implied = abs(odds) / (abs(odds) + 100)

# Positive odds (underdog): +130 → 100/230 = 43.5%
implied = 100 / (odds + 100)
```

### Modified: `edge_finder.find_edges()`

Add a `min_prob_edge: float = config.MIN_PROB_EDGE` parameter. After the existing EV and `MIN_AMERICAN_ODDS` filters, apply:

```
model_prob − market_implied_prob(american_odds) > min_prob_edge
```

Logged at INFO level: `"Applied prob-edge filter: {n_before} → {n_after} edges"`.

### Modified: `config.py`

Add `MIN_PROB_EDGE: float = <sweep_result>` alongside `EV_THRESHOLD`. Both updated to the Sharpe-optimal values found by the sweep.

### Modified: `backtest.py`

**New: `sweep_thresholds(clf, training_df, ...) -> pd.DataFrame`**

- Iterates a grid of `(ev_threshold, min_prob_edge)` pairs:
  - `ev_threshold`: 0.05 → 0.50 in 0.05 steps (10 values)
  - `min_prob_edge`: 0.00 → 0.30 in 0.05 steps (7 values)
  - 70 combinations total — fast on the existing synthetic path.
- At each combination, calls `run_backtest()` with the given thresholds.
- Computes `n_bets`, `win_rate`, `roi_pct`, `sharpe_ratio`, `avg_bets_per_day` via `compute_summary()`.
- Logs progress every 10 combinations: `"Sweep: {i}/70 complete"`.
- Excludes combinations with 0 bets (Sharpe = NaN).
- Returns DataFrame sorted by `sharpe_ratio` descending.
- Logs the winner: `"Optimal thresholds: EV={ev:.0%}, MIN_PROB_EDGE={edge:.0%} (Sharpe={sharpe:.3f}, {n} bets, {avg:.1f}/day)"`.

`run_backtest()` gains `ev_threshold` and `min_prob_edge` parameters (both default to `config` values) so the sweep can drive them without mutating global state.

### Modified: `notebooks/02_backtest.ipynb`

New section appended after the existing synthetic backtest cells:

1. Run `sweep_thresholds()` — display results table sorted by Sharpe.
2. Plot: heatmap of Sharpe ratio across `(ev_threshold, min_prob_edge)` grid.
3. Print the chosen `(EV_THRESHOLD, MIN_PROB_EDGE)` pair.
4. Re-run `run_backtest()` at the optimal pair — print `compute_summary()` and plot cumulative P&L.
5. **Real-world validation section:** load May 19–24 edge outputs + fetch actual results via `statsapi`, compute real P&L at both old and new thresholds, display side-by-side.

---

## Data Flow

```
existing synthetic backtest (run_backtest)
    ↓
sweep_thresholds(ev_range, prob_edge_range)
    → 70 (ev_threshold, min_prob_edge) combinations
    → Sharpe-sorted DataFrame
    → best_ev, best_min_prob_edge
    ↓
config.EV_THRESHOLD = best_ev
config.MIN_PROB_EDGE = best_min_prob_edge   (new constant, committed)
    ↓
find_edges() uses both filters going forward
```

---

## Error Handling & Logging

| Stage | Level | Message |
|---|---|---|
| `market_implied_prob` invalid odds (0) | WARNING | `"market_implied_prob received odds=0, returning 0.5"` |
| `find_edges` prob-edge filter applied | INFO | `"Prob-edge filter ({min_prob_edge:.0%}): {n_before} → {n_after} edges"` |
| Sweep combination with 0 bets | DEBUG | `"Skipping EV={ev:.0%}, edge={e:.0%} — no bets at this threshold"` |
| Sweep progress | INFO | `"Threshold sweep: {i}/70 complete"` (every 10) |
| Sweep winner | INFO | `"Optimal: EV={ev:.0%} MIN_PROB_EDGE={e:.0%} Sharpe={s:.3f} ({n} bets, {avg:.1f}/day)"` |
| All combinations produce 0 bets | RuntimeError | `"Threshold sweep produced no valid combinations"` |

---

## Testing

- Unit: `market_implied_prob` — favourite, underdog, zero-odds edge case.
- Unit: `find_edges` with `min_prob_edge > 0` — assert rows are correctly filtered vs a mock features DataFrame.
- Unit: `sweep_thresholds` with a small synthetic DataFrame — assert the known-best combination wins.
- Unit: `run_backtest` with explicit `ev_threshold` / `min_prob_edge` args — assert results differ from defaults when args differ.
- Regression: all existing 158 tests remain green (no existing signatures broken).
