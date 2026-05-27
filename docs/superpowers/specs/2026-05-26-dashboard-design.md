# Dashboard Design Spec
**Date:** 2026-05-26  
**Status:** Approved

## Overview

A GitHub Pages static site that makes the MLB Edge Finder portfolio-demonstrable in 30 seconds without cloning. Generated daily by the existing GitHub Actions workflow. Primary audience: portfolio reviewers. Secondary: daily personal use.

---

## Goals

- Show today's recommended edges in a clean, readable table
- Demonstrate model credibility via backtest stats (win rate, ROI, Sharpe, ROC-AUC)
- Show the system is actively running via an edge history chart (bets flagged per day)
- Be visually distinctive and professional as a portfolio piece

## Non-Goals

- Live P&L tracking against real outcomes (requires paid Odds API historical plan)
- Mobile-first design (portfolio viewers are predominantly on desktop)
- Any server-side logic — purely static HTML

---

## Architecture

### New module: `src/mlb_edge_finder/generate_site.py`

Standalone script responsible for all HTML generation. Called by the daily workflow after the edges CSV is promoted to `outputs/`. Writes a single `docs/index.html`.

**Inputs:**
- `outputs/edges_YYYY-MM-DD.csv` — all historical edge files (reads all, sorts by date)
- `models/metrics_*.json` — latest metrics file (sorted by filename date, most recent wins)
- `data/backtest_pnl.json` — pre-committed cumulative P&L array from the held-out backtest

**Output:**
- `docs/index.html` — complete self-contained HTML page (Chart.js via CDN, no local assets)

### GitHub Pages configuration

- Served from `docs/` directory on `main` branch
- Enabled via repository Settings → Pages → Source: `docs/` on `main`
- URL: `https://<username>.github.io/<repo>/`

### Workflow integration

New step added to `.github/workflows/daily.yml` after "Promote edges file to outputs/":

```yaml
- name: Generate dashboard
  run: python -m mlb_edge_finder.generate_site
```

`docs/index.html` is added to the existing commit step alongside the edges CSV.

### New data artifact: `data/backtest_pnl.json`

Committed once from the backtest notebook. Contains the cumulative unit P&L array from the held-out 20% test split. Format:

```json
{
  "cumulative_pnl": [0.0, 0.91, 1.83, ...],
  "summary": {
    "n_bets": 2370,
    "win_rate": 0.603,
    "roi_pct": 15.1,
    "sharpe": 0.735,
    "max_drawdown": 26.73
  }
}
```

Generated once by running a new helper `backtest.export_pnl_json()` from the notebook, then committed. Updated only when the model is retrained from scratch (rare). The feedback loop retrains periodically but does not regenerate this file — it's a stable portfolio artifact.

---

## Page Layout: Two-Column

```
┌─────────────────────────────────────────────────────────────┐
│  HEADER: ⚾ MLB Edge Finder  |  subtitle + date badge        │
├──────────────────────────────────────┬──────────────────────┤
│  LEFT COLUMN (main)                  │  RIGHT SIDEBAR        │
│                                      │                       │
│  Today's Edges                       │  Last updated: date   │
│  ┌────────────────────────────────┐  │                       │
│  │ Matchup | Side | Odds | EV |.. │  │  Backtest Stats       │
│  │ ...                            │  │  Win Rate   60.3%     │
│  └────────────────────────────────┘  │  ROI        +15.1%    │
│                                      │  Sharpe     0.74      │
│  Edge History (last 30 days)         │  ROC-AUC    0.601     │
│  ▂▃▅▂▆▄▇▃▅▆▂▄▃▅▇▆▂▃▄▅             │  Test games 3,168     │
│                                      │                       │
│                                      │  Backtest P&L         │
│                                      │  ╱──────────╱         │
└──────────────────────────────────────┴──────────────────────┘
```

---

## Page Components

### Header
- Title: "⚾ MLB Edge Finder"
- Subtitle: "XGBoost model identifying positive expected-value MLB moneyline bets"
- Date badge: "Updated {date}" in Giants orange

### Today's Edges Table (left, top)

Columns (in order): Matchup, Side, Odds, Model Prob, EV, Kelly, Flag

| Column | Source | Display |
|---|---|---|
| Matchup | `home_team` vs `away_team` | "SD Padres vs PHI Phillies" |
| Side | `bet_side` | "home" / "away" badge |
| Odds | `american_odds` | "+161" / "-109" |
| Model Prob | `model_prob` | "91.6%" |
| EV | `ev` | "+75.6%" in orange |
| Kelly | `kelly_fraction` | "41.2%" |
| Flag | `prob_flag` | ⚠ badge when True (signals manual review) |

Empty state (header-only CSV): centered message "No edges found today — model found no +EV opportunities meeting the current thresholds."

### Edge History Chart (left, bottom)

- Bar chart, last 30 days (or all available if < 30)
- X-axis: date labels (abbreviated)
- Y-axis: number of edges found (integer)
- Zero-edge days shown as empty bars (height 0) to preserve the timeline
- Chart.js `bar` type, Giants orange bars on dark background

### Backtest Stats Block (right sidebar, top)

Pulled from the most recent `models/metrics_*.json`:
- Win Rate (from backtest results — hardcoded from `data/backtest_pnl.json` or metrics)
- Backtest ROI
- Sharpe Ratio
- ROC-AUC
- Test sample count

Note: `metrics_*.json` currently stores `accuracy`, `roc_auc`, `log_loss`, `brier_score`, `n_test_samples`. Win rate, ROI, and Sharpe come from `data/backtest_pnl.json`.

### Backtest P&L Chart (right sidebar, bottom)

- Line chart of cumulative unit P&L over the held-out test set
- Data sourced from `data/backtest_pnl.json`
- Chart.js `line` type, Giants orange line, no fill
- X-axis: test game index (no date labels — too many points)
- Y-axis: cumulative units won/lost

---

## Color Scheme: SF Giants

A personal touch — the SF Giants are the developer's favorite team.

| Role | Color |
|---|---|
| Background | `#1a1713` |
| Surface (cards, nav) | `#27251F` (Giants black) |
| Border | `#3d3930` |
| Accent / highlight | `#FD5A1E` (Giants orange) |
| Body text | `#d6cfc4` |
| Muted text | `#8a8070` |
| Positive values | `#4ade80` (green, color-blind safe) |

---

## Data Files

### `data/backtest_pnl.json` (new, committed once)

```json
{
  "cumulative_pnl": [0.0, 0.91, 1.83, ...],
  "summary": {
    "n_bets": 2370,
    "win_rate": 0.603,
    "roi_pct": 15.1,
    "sharpe": 0.735,
    "max_drawdown": 26.73
  }
}
```

Generated by new `backtest.export_pnl_json(backtest_df, summary, path)` helper. Run once from `notebooks/02_backtest.ipynb`, committed alongside the spec.

### `docs/index.html` (generated daily)

Complete self-contained HTML. All data embedded as inline JSON `<script>` blocks. No external data fetches at runtime — works offline and as a saved file.

---

## Error Handling

| Scenario | Behavior |
|---|---|
| No edges CSV for today | Shows "No edges found today" message; history chart still renders |
| `data/backtest_pnl.json` missing | Sidebar P&L chart section is omitted with a note; site still generates |
| `models/metrics_*.json` missing | Backtest stats block omitted; site still generates |
| `outputs/` is empty | History chart renders with no bars; table shows empty state |

`generate_site.py` never raises — it degrades gracefully and always writes a valid `docs/index.html`.

---

## README Addition

New "Dashboard" section in `README.md`:

> **Live Dashboard:** [https://\<username\>.github.io/mlb-edge-finder/](url)
>
> Updated daily by GitHub Actions at 9:30 AM EDT. Shows today's recommended edges, a 30-day edge history, and the model's validated backtest performance.
>
> *The color scheme uses the SF Giants' official black and orange — a small personal touch from a lifelong Giants fan.*

---

## Testing

- Unit test for `generate_site.generate(outputs_dir, metrics_path, pnl_path, out_path)` — mock file inputs, assert output file exists and contains expected strings (team names, EV values, chart data)
- Test empty-state: no edges CSV for today → output contains "No edges found"
- Test missing backtest file → output is still valid HTML, P&L section omitted
- No browser/rendering tests — pure string output assertions

---

## Files Changed

| File | Change |
|---|---|
| `src/mlb_edge_finder/generate_site.py` | New module |
| `src/mlb_edge_finder/backtest.py` | Add `export_pnl_json()` helper |
| `data/backtest_pnl.json` | New artifact, committed once |
| `docs/index.html` | Generated output, committed daily |
| `.github/workflows/daily.yml` | Add "Generate dashboard" step |
| `README.md` | Add "Dashboard" section with live URL |
| `CLAUDE.md` | Update current phase notes |
