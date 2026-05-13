# compute_kelly() Design Spec

**Date:** 2026-05-12  
**Status:** Approved

## Goal

Add `compute_kelly(prob, american_odds) -> float` to `edge_finder.py` alongside `compute_ev`. Integrate it into `find_edges()` so the output DataFrame gains a `kelly_fraction` column giving half-Kelly bet sizing for each flagged edge.

## Function: `compute_kelly`

**Signature:** `compute_kelly(prob: float, american_odds: int) -> float`

**Formula:**

1. Derive net payout `b` — same logic as `compute_ev`:
   - Negative odds: `b = 100 / abs(american_odds)`
   - Positive odds: `b = american_odds / 100`
2. Compute EV: `ev = prob * b - (1 - prob)`
3. Full Kelly fraction: `f_full = ev / b`
4. Half Kelly: `kelly_fraction = f_full / 2`
5. Clamp to `[0.0, 1.0]` — negative EV → 0.0, pathological high values → 1.0

**Rationale for half-Kelly:** Halves variance while preserving most of the growth rate advantage. Standard risk-reduction practice.

**Rationale for clamping:** Negative EV bets have `kelly_fraction < 0`, which means "don't bet". Returning 0.0 is the correct signal. Values above 1.0 can't occur when EV thresholds are sane but the clamp guards against edge cases.

## Integration: `find_edges` update

Both the home and away passes call `compute_kelly(model_prob, american_odds)` for each flagged row and store the result in a `kelly_fraction` column. `output_cols` gains `"kelly_fraction"`. The CSV written to disk includes the column.

**Updated output schema:**

| Column | Description |
|---|---|
| `game_id` | Game identifier |
| `home_team` | Home team name |
| `away_team` | Away team name |
| `bet_side` | `"home"` or `"away"` |
| `american_odds` | Best American moneyline for the bet side |
| `model_prob` | Model-predicted win probability |
| `ev` | Expected value per unit wagered |
| `kelly_fraction` | Half-Kelly fraction of bankroll to wager (0.0–1.0) |

## No Changes Needed

- `config.py` — half-Kelly (÷2) is baked in; no new constant needed
- `compute_ev` — unchanged per the CLAUDE.md Kelly seam note
- All other modules — `find_edges` is the only consumer

## Tests

New tests in `tests/test_edge_finder.py`:

| Test | What it verifies |
|---|---|
| `test_compute_kelly_signature` | Function exists with `prob` and `american_odds` params |
| `test_compute_kelly_zero_ev_returns_zero` | prob=0.60, -150 → EV=0.0 → kelly=0.0 |
| `test_compute_kelly_positive_ev_underdog` | prob=0.55, +110 → specific positive value |
| `test_compute_kelly_negative_ev_returns_zero` | prob=0.40, -150 → EV<0 → kelly=0.0 |
| `test_compute_kelly_clamped_at_one` | Pathological input clamped to 1.0 |
| `test_find_edges_includes_kelly_fraction` | edges output has `kelly_fraction` column, value > 0 |
| Update `test_find_edges_returns_home_edge` | Assert `"kelly_fraction"` in `result.columns` |
| Update `test_find_edges_both_sides` | Assert `"kelly_fraction"` in `result.columns` |
| Update `test_find_edges_empty_when_no_edges` | Assert `"kelly_fraction"` in empty DataFrame columns |
| Update `test_find_edges_filters_min_odds` | Assert `"kelly_fraction"` in empty DataFrame columns |
