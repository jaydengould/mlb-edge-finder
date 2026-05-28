# Design: Threshold Rebalance + High-Confidence Badge

**Date:** 2026-05-28
**Status:** Approved

## Problem

The current edge-finding thresholds (`EV_THRESHOLD=0.50`, `MIN_PROB_EDGE=0.30`) were derived from a Sharpe-optimal sweep against a **synthetic** flat -110/-110 market. Against real bookmaker lines — which price each team asymmetrically — `MIN_PROB_EDGE=0.30` almost never fires (model would need ~84%+ confidence on a -110 favorite). The result: zero edges reported on most days, an empty dashboard, and a portfolio project that looks broken.

Additionally, the existing `prob_flag` column was a warning label (`model_prob > 0.80`) that fires most often on the same bets that look strongest — creating contradictory signals. The model is now calibrated with isotonic regression, which was specifically added to address overconfidence; the `prob_flag` is redundant and should be retired.

## Goals

- Produce ~3–4 edges per day on average (some days 1–2, some days 5–7, depending on the slate)
- Filter out genuinely weak edges (EV < 0.20 is noise at a 57%-accuracy model)
- Positively label the strongest edges as "High Confidence" — no negative labeling for the rest
- Keep the output schema stable (column rename only, no structural changes)

## Out of Scope

- Changing the model, training data, or calibration
- A real-odds backtest (tracked separately as Future Work #1)
- Guaranteed daily output — zero edges on a dead market day is honest and acceptable

---

## Design

### 1. Config (`config.py`)

Four changes:

| Constant | Old | New | Reason |
|---|---|---|---|
| `EV_THRESHOLD` | `0.50` | `0.20` | Filters noise but fires regularly |
| `MIN_PROB_EDGE` | `0.30` | *(removed)* | Redundant with EV threshold — clearing EV > 0.20 on a normal MLB line already implies an 8–13pp prob gap. The concept moves to the badge logic only. |
| `HIGH_CONFIDENCE_EV` | *(new)* | `0.40` | EV bar for positive badge |
| `HIGH_CONFIDENCE_PROB_EDGE` | *(new)* | `0.15` | Prob-gap bar for positive badge |

`HIGH_CONFIDENCE_EV` and `HIGH_CONFIDENCE_PROB_EDGE` are defined in `config.py` alongside the existing thresholds. They are never used as filters — they only determine whether an edge that already passed the permissive threshold earns the badge.

`MIN_PROB_EDGE` is deleted from `config.py` and all call sites. `find_edges()` loses its `min_prob_edge` parameter entirely.

### 2. `edge_finder.py`

One logic change: replace `prob_flag` with `high_confidence`.

**Old:**
```python
edges["prob_flag"] = model_prob[mask] > 0.80
```

**New:**
```python
market_implied = np.array([market_implied_prob(int(o)) for o in odds])
edges["high_confidence"] = (
    (ev_series > config.HIGH_CONFIDENCE_EV) &
    ((model_prob_array - market_implied) > config.HIGH_CONFIDENCE_PROB_EDGE)
)
```

Both conditions must be true simultaneously:
- EV is genuinely strong (> 0.40)
- Model has a meaningful gap over the market (> 15pp) — not just high EV on a bet the market already prices similarly

Output schema is unchanged except the column rename (`prob_flag` → `high_confidence`). Same dtype (bool), same position in the DataFrame.

The `market_implied` array is already computed during the home/away mask passes — no new API calls or loops.

### 3. Dashboard (`generate_site.py`)

One display change in the edge history table:

- Rows where `high_confidence=True`: render the `bet_side` cell with a `★` prefix (e.g. `★ home`) or a small inline badge
- Rows where `high_confidence=False`: render exactly as today — no label, no asterisk, no negative framing

No structural changes to the HTML template. The badge is purely cosmetic — a CSS class on the table cell or a prepended character in the f-string template.

### 4. `backtest.py` + `02_backtest.ipynb`

`run_backtest()` currently computes `prob_flag` directly using the old logic. Update to use `high_confidence` instead, consistent with the new column name and logic. No change to the backtest's P&L calculation — `high_confidence` is metadata, not a filter.

Update the notebook output column references to match.

### 5. Tests

- Update any test that asserts `prob_flag` column exists or checks its values → assert `high_confidence` instead
- Add one test: edge with EV=0.45 and prob_gap=0.20 → `high_confidence=True`
- Add one test: edge with EV=0.25 and prob_gap=0.08 → `high_confidence=False`
- Verify new config constants are present and in valid range

---

## What Does Not Change

- `compute_ev()` — untouched
- `compute_kelly()` — untouched
- `market_implied_prob()` — untouched
- Model, training data, pipeline orchestration
- Output CSV schema (same columns, `prob_flag` renamed to `high_confidence`)
- `MIN_AMERICAN_ODDS = -300` — still filters extreme favorites
- `market_implied_prob()` is still called inside `find_edges()` — it's needed for the `high_confidence` badge computation, just no longer used as a filter gate

---

## Expected Outcome

Based on pre-change output files (2026-05-19 through 2026-05-24), lowering to `EV_THRESHOLD=0.20` with `MIN_PROB_EDGE=0.05` should produce approximately 3–8 edges on full-slate days and 1–3 on lighter days. The `high_confidence` badge will fire on the subset with EV > 0.40 and a 15pp+ prob gap — roughly the top third of edges on a typical day.
