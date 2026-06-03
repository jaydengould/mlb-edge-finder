# ROI Reframe + Market-Efficiency Sensitivity — Design Spec

**Date:** 2026-06-02
**Status:** Approved

## Problem

The dashboard and README present the temporal-holdout backtest's **+18.3% ROI / 62% win rate** as the project's headline result. These numbers are an artifact of the synthetic **−110/−110 (50/50) market** the backtest runs against, not a real betting edge:

- The model only fires a bet when EV > 0.20 against a flat −110 line, which requires `model_prob ≳ 0.66`. Every bet is therefore on a strong favorite.
- A real sportsbook *prices the favorite* — it moves the line toward the strong team. The synthetic market stays a naive 50/50 coin, so the model profits purely from the market "not knowing" who the favorite is.
- The model's actual signal is weak: ROC-AUC 0.555, Brier 0.248 (≈ base rate). The +18% is not predictive skill.
- A technical reviewer will see the contradiction (near-random ROC-AUC, strongly positive ROI) immediately. As presented, the headline is the project's weakest point.

A real-odds backtest is the gold-standard fix but requires a paid Odds API plan, which is out of scope for a portfolio project.

## Goal

Reframe the project's evaluation story to be honest and self-critical — which, for a portfolio, is a *strength* — without paid data. Specifically:

1. Replace the misleading P&L curve with a **market-efficiency sensitivity analysis** that visualizes the real reason the edge isn't real: it evaporates as the market gets informed.
2. Lead the dashboard and README with legitimate signals (methodology, ROC-AUC, calibration) and clearly tag the betting ROI as illustrative.
3. Add an honest **Limitations & What I'd Do Next** section.

The product code (`edge_finder`, `pipeline`, daily workflow) is unchanged — only the *evaluation and presentation* layer changes.

## Why a market-efficiency sweep (not a vig sweep)

A vig sweep was considered and **rejected**: the production backtest already runs at the real ~4.76% MLB vig and still shows +18% ROI, so sweeping vig upward would show the edge "surviving" past the real-market level — the opposite of the honest story. Vig is not what kills the edge; *the market pricing the favorite* is.

The honest knob is **how much the market knows**. We interpolate each game's synthetic market probability between two endpoints:

- **α = 0 (naive market):** 0.5 for every game — ignores the matchup. Reproduces today's illustrative +18% ROI.
- **α = 1 (sharp market):** the market's implied probability equals the model's own calibrated prediction → the model has zero informational edge and only pays the vig → ROI ≈ −vig.

Per-game market-implied home probability at sweep point α:

```
market_prob_home(α, game) = 0.5 * (1 - α) + model_prob_home(game) * α
```

Sweeping α from 0 → 1 and plotting realized ROI shows the edge collapsing as the market becomes informed. The **break-even α** (where ROI crosses zero) is the headline: "the market only has to be ~X% as informed as my own model before the edge vanishes."

**Stated caveat (in README + dashboard caption):** using the model's own prediction as the "sharp market" is a proxy — a real market could be sharper or duller. It is a principled way to demonstrate edge fragility to market efficiency without real odds, not a claim of equivalence to live lines.

---

## Architecture

Three layers change; the boundary between them is the `temporal_eval_2025.json` artifact.

```
backtest.sweep_market_efficiency()  ──►  temporal_eval.run()  ──►  temporal_eval_2025.json  ──►  generate_site.generate()
   (new analysis function)              (calls sweep, writes JSON)      (data contract)            (reads JSON, renders dashboard)
```

### 1. `backtest.py` — new analysis function

**Refactor first (DRY):** the per-bet EV/Kelly loop currently lives inside `simulate_bets()`, which generates a single flat odds pair via `simulate_market_odds()` and applies it to all games. Extract the loop into a helper that accepts **per-game** American odds:

```python
def _run_bet_loop(
    clf, X_test, y_test, meta_df,
    home_odds: pd.Series,   # per-game home American odds, indexed like X_test
    away_odds: pd.Series,   # per-game away American odds, indexed like X_test
    unit: float = 100.0,
    ev_threshold: float | None = None,
) -> pd.DataFrame:
    """Existing simulate_bets per-bet loop, but with per-game odds inputs."""
```

`simulate_bets()` keeps its current signature and behavior — it builds flat per-game odds Series from `simulate_market_odds(home_market_prob, vig)` and delegates to `_run_bet_loop`. This preserves all existing `simulate_bets` tests and callers (`run_backtest`, `temporal_eval`).

**New function:**

```python
def sweep_market_efficiency(
    clf, X_test, y_test, meta_df,
    alpha_grid=None,        # default: np.round(np.arange(0.0, 1.0001, 0.05), 4)
    vig: float = 0.0476,
    ev_threshold: float | None = None,
) -> pd.DataFrame:
    """Sweep market efficiency α from 0 (naive 50/50) to 1 (market = model).

    For each α:
      1. home_prob = clf.predict_proba(X_test)[:, 1]
      2. market_home = 0.5*(1-α) + home_prob*α    (per game)
      3. apply vig: split additively, convert each side to American odds
      4. run _run_bet_loop with those per-game odds
      5. summarize via compute_summary
    Returns a DataFrame with columns:
        alpha, roi_pct, n_bets, win_rate
    sorted by alpha ascending. Rows with 0 bets get roi_pct=0.0, win_rate=0.0.
    """
```

Vig application per game (mirrors `simulate_market_odds`'s additive split):
```
home_implied = market_home + vig/2
away_implied = (1 - market_home) + vig/2
home_american = _to_american(home_implied)   # reuse simulate_market_odds's inner logic
away_american = _to_american(away_implied)
```
`_to_american` is currently a closure inside `simulate_market_odds`; promote it to a module-level helper `_prob_to_american(p: float) -> float` so both functions use it.

**Break-even computation** (lives in `temporal_eval`, not `backtest`, since it reads the sweep result):

```python
def _break_even_alpha(sweep_df: pd.DataFrame) -> float | None:
    """First α where roi_pct crosses from >= 0 to < 0, linearly interpolated.
    Returns None if ROI never goes negative across the grid."""
```

### 2. `temporal_eval.py` — extend `run()` and the JSON

After the existing production-vig `simulate_bets` + `compute_summary` block, add:

```python
sweep_df = sweep_market_efficiency(cal_clf, X_test, y_test, meta_df)
break_even = _break_even_alpha(sweep_df)
```

**JSON changes:**
- **Add** `"market_efficiency_sweep"`: list of `{"alpha": float, "roi_pct": float, "n_bets": int}` (win_rate omitted from the artifact to keep it lean; available in the DataFrame if needed).
- **Add** `"break_even_alpha"`: float (rounded 4dp) or `null`.
- **Remove** `"pnl_series"` — no longer charted.
- All existing scalar keys (`accuracy`, `roc_auc`, `log_loss`, `brier_score`, `n_bets`, `win_rate`, `roi_pct`, `sharpe_ratio`, `total_pnl`, `avg_ev`, `max_drawdown`, `holdout_season`, `train_seasons`, `n_train`, `n_test`) stay unchanged.

`import numpy as np` added to `temporal_eval.py` for the alpha grid default if not constructed in `backtest`. (Grid default lives in `backtest.sweep_market_efficiency`, so `temporal_eval` does not need numpy unless `_break_even_alpha` uses it — it can use plain Python.)

The CLI `__main__` block prints `break_even_alpha` in its summary table.

### 3. `generate_site.py` — dashboard

**Data loading:** `_load_temporal_eval` unchanged (still globs `temporal_eval_*.json`).

**Hero chart** — rename `_render_pnl_html` → `_render_efficiency_html(te_data)`:
- Card title: "Edge vs Market Efficiency"
- Canvas id: `efficiency-chart`
- Returns `""` when `te_data` is None or has no `market_efficiency_sweep`.
- A short caption `<div>` under the canvas: "Synthetic-market stress test — ROI as the market becomes as informed as the model."

**Stats card** — `_render_stats_html(te_data)` reordered and relabeled:
- **ROC-AUC** (first — the legitimate positive signal), neutral styling.
- **Break-even efficiency** — `break_even_alpha` shown as `α ≈ 0.NN` (or "none in range" if null).
- **Accuracy**, **Holdout games** — neutral.
- **Backtest ROI** — kept, green if positive, but label becomes **"ROI (naive market)"** with the existing subtitle carrying the "illustrative" caveat.
- **Win Rate** — kept, label **"Win Rate (naive market)"**.
- Subtitle: `Trained 2019–2024 · 2025 holdout · synthetic-market stress test`.

**JS:** replace the second IIFE (P&L line chart reading `TE.pnl_series`) with an efficiency line chart reading `TE.market_efficiency_sweep`:
- `labels`: `alpha` values formatted as numbers (e.g. `0`, `0.25`, `0.5`, `0.75`, `1`).
- `data`: `roi_pct`.
- A horizontal zero baseline (Chart.js: a second dataset of zeros, thin grey, no points, OR `y` grid emphasises 0 — use a flat zero-line dataset for clarity).
- Tooltip: `α = {alpha}: ROI {roi}%`.
- Guard: `if(!el||!TE||!TE.market_efficiency_sweep||TE.market_efficiency_sweep.length===0)return;`
- The history-chart IIFE (first) is unchanged.

Both existing IIFEs already have balanced braces after the prior bug fix — preserve that; do not reintroduce extra `}}` pairs.

### 4. `README.md`

- **Dashboard section:** replace "the model's validated backtest performance" framing with the honest methodology framing — temporal holdout + calibration + market-efficiency stress test. State that ROI shown is vs a naive synthetic market and is illustrative.
- **Model section:** keep the production + temporal tables; relabel the temporal table's ROI/Win-rate rows as "vs naive synthetic market (illustrative)"; add a row or sentence for `break_even_alpha` with a plain-English reading.
- **Backtest section:** add a paragraph describing the market-efficiency sweep and what break-even α means; keep the existing synthetic-odds caveat.
- **New section "Limitations & What I'd Do Next"** (after Backtest, before Running Tests):
  - Synthetic odds, not real lines (paid-API gap).
  - Market-as-model proxy caveat for the sweep.
  - Calibration drift under temporal shift (model predicts ~66% on its bets, realizes ~62%).
  - Modest signal: ROC-AUC 0.555 — weak but positive; the project demonstrates rigorous methodology, not a profitable system.
  - Next: real-odds backtest, time-matched team stats, additional features (rest, travel, park, weather).

### 5. Housekeeping (in-scope)

- `model.calibrate()` docstring: replace the stale "Uses sklearn's CalibratedClassifierCV with cv='prefit'" sentence with an accurate description referencing `FrozenEstimator`.
- `CLAUDE.md`: add a bullet describing `sweep_market_efficiency` + the reframe; fix the stale "`_load_training_csv()` … (largest file by size)" line (now filename-range parsing) and the "212 total passing" line (now 220, will become higher after new tests).

---

## Data Contract — `temporal_eval_2025.json` (after change)

```json
{
  "holdout_season": 2025,
  "train_seasons": [2019, 2021, 2022, 2023, 2024],
  "n_train": 12606,
  "n_test": 2444,
  "accuracy": 0.5471,
  "roc_auc": 0.5552,
  "log_loss": 0.69,
  "brier_score": 0.2482,
  "n_bets": 205,
  "win_rate": 0.6195,
  "roi_pct": 18.27,
  "sharpe_ratio": 0.1966,
  "total_pnl": 3745.45,
  "avg_ev": 0.3236,
  "max_drawdown": 618.18,
  "break_even_alpha": 0.1,
  "market_efficiency_sweep": [
    {"alpha": 0.0, "roi_pct": 18.27, "n_bets": 205},
    {"alpha": 0.05, "roi_pct": 11.4, "n_bets": 198},
    {"alpha": 1.0, "roi_pct": -4.76, "n_bets": 12}
  ]
}
```

(`break_even_alpha` and sweep values above are illustrative — filled by the actual run.)

---

## Files Changed

| File | Change |
|---|---|
| `src/mlb_edge_finder/backtest.py` | Promote `_prob_to_american`; extract `_run_bet_loop`; add `sweep_market_efficiency`; `simulate_bets` delegates to `_run_bet_loop` (behavior unchanged) |
| `src/mlb_edge_finder/temporal_eval.py` | Call sweep; add `_break_even_alpha`; write `market_efficiency_sweep` + `break_even_alpha`; drop `pnl_series`; print break-even in CLI |
| `src/mlb_edge_finder/generate_site.py` | `_render_pnl_html` → `_render_efficiency_html`; reorder/relabel stats card; swap P&L JS chart for efficiency chart |
| `src/mlb_edge_finder/model.py` | Fix stale `calibrate()` docstring |
| `models/temporal_eval_2025.json` | Regenerated by running `temporal_eval` |
| `docs/index.html` | Regenerated by running `generate_site` |
| `README.md` | Reframe Dashboard/Model/Backtest sections; add Limitations section |
| `CLAUDE.md` | Document sweep + reframe; fix two stale lines |
| `tests/test_backtest.py` | Tests for `sweep_market_efficiency` and `_run_bet_loop` |
| `tests/test_temporal_eval.py` | Assert new JSON keys; drop `pnl_series` assertions |
| `tests/test_generate_site.py` | Assert efficiency chart renders, ROC-AUC in stats, no P&L chart |

`backtest.export_pnl_json` and `data/backtest_pnl.json` are left as-is (already unused by the dashboard; out of scope).

---

## Tests

**`tests/test_backtest.py` (new):**
1. `test_run_bet_loop_per_game_odds` — `_run_bet_loop` selects bets using per-game odds (a game with generous odds is bet, one with stingy odds is not).
2. `test_simulate_bets_unchanged_after_refactor` — existing `simulate_bets` output is identical pre/post refactor for a fixed input (regression guard).
3. `test_sweep_returns_expected_columns` — output has `alpha, roi_pct, n_bets, win_rate`, one row per grid point.
4. `test_sweep_roi_decreases_with_efficiency` — `roi_pct` at α=0 ≥ `roi_pct` at α=1.
5. `test_sweep_alpha_one_roi_near_negative_vig` — at α=1, ROI ≤ 0 (model bets into its own number, pays vig).
6. `test_sweep_handles_no_bets` — a clf that never clears the EV threshold yields all-zero `n_bets` without error.

**`tests/test_temporal_eval.py` (update):**
7. `test_run_json_has_market_efficiency_sweep` — output has `market_efficiency_sweep` (non-empty list of `{alpha, roi_pct, n_bets}`) and `break_even_alpha` key.
8. Remove/replace `pnl_series` assertions.
9. `test_break_even_alpha_interpolates` — `_break_even_alpha` returns the interpolated crossing for a hand-built sweep; returns `None` when ROI stays non-negative.

**`tests/test_generate_site.py` (update):**
10. `test_generate_renders_efficiency_chart` — output HTML contains `efficiency-chart` canvas and references `market_efficiency_sweep`.
11. `test_generate_no_pnl_chart` — output HTML no longer contains `pnl-chart` / `pnl_series`.
12. `test_stats_card_leads_with_roc_auc` — ROC-AUC stat present; ROI labeled as naive/illustrative.
13. Graceful degradation: missing `market_efficiency_sweep` → no efficiency card, no crash.

---

## Running After Implementation

```bash
python -m mlb_edge_finder.temporal_eval --force      # regenerate JSON with sweep
python -m mlb_edge_finder.generate_site              # regenerate dashboard
pytest tests/ -v
```

Commit the regenerated `models/temporal_eval_2025.json` and `docs/index.html` as static artifacts.
