# Time-Matched Pitcher Snapshots — Design Spec

**Date:** 2026-05-26  
**Status:** Approved

## Problem

The model produces extreme win probabilities (>80%) on edges that the market prices near 50/50. Root cause: pitcher stats in training use a single September 28 end-of-season snapshot (150+ IP per pitcher), but inference runs in mid-May with ~50 IP per pitcher. The model cannot distinguish "6.04 ERA in 50 innings" from "6.04 ERA in 200 innings" and treats early-season noise as a strong signal.

A secondary issue: pitchers with very few innings (< 30 IP) produce meaningless ERA/WHIP values (e.g. 0.00 ERA in 4 IP) that flow straight through to the model.

## Goal

Fix the training/inference distribution mismatch on pitcher stats so that the model learns from time-matched data — the stats a bettor would actually have available on a given game date. Accept fewer edges in exchange for higher-confidence, more credible predictions.

## What Changes

Three components change. Everything else — inference pipeline, edge finder, feedback loop, daily workflow — is untouched at the interface level.

### 1. `pitcher_ingestion.py` — `fetch_pitcher_snapshot`

New function alongside the existing `fetch_pitcher_stats` and `fetch_probable_starters`:

```python
def fetch_pitcher_snapshot(season: int, snapshot_date: date, force: bool = False) -> pd.DataFrame:
```

- Calls the same statsapi endpoint as `fetch_pitcher_stats`
- **IP floor:** pitchers with `ip < MIN_PITCHER_IP` (30 IP) are excluded from the output entirely — they arrive as NaN in downstream joins, identical to "no starter matched"
- Writes to `data/raw/pitcher_snapshot_YYYY-MM-DD.csv` (distinct from the ephemeral `pitcher_stats_YYYY-MM-DD.csv`)
- Cache-first: returns cached file if it exists and `force=False`

`MIN_PITCHER_IP = 30` added to `config.py`.

### 2. `training_data.py` — Multi-snapshot join in `_build_season`

Replace the single `fetch_pitcher_stats(date(season, 9, 28))` call with a time-matched snapshot join.

**Snapshot dates per season:**

| Snapshot | Covers games from |
|---|---|
| April 30 | May 1 – May 31 |
| June 1 | June 2 – July 30 |
| July 31 | August 1 – September 27 |
| September 28 | September 29+ |

**Join logic:**
- For each game row, find the latest snapshot date strictly before `game_date`
- Join pitcher stats from that snapshot
- Games before April 30 (early April) get `NaN` pitcher stats — correct, since no reliable data exists yet
- September 28 falls back to `fetch_pitcher_stats(date(season, 9, 28))` if no `pitcher_snapshot_YYYY-09-28.csv` exists, preserving current behavior for seasons predating this change

**Implementation:** load all available snapshots for the season at the start of `_build_season`, build a lookup dict `{snapshot_date: df}`, then for each game date select the correct snapshot before the pitcher join.

### 3. `.github/workflows/snapshot.yml` — New workflow

Three cron triggers plus `workflow_dispatch` for manual backfilling:

```yaml
on:
  schedule:
    - cron: '30 14 30 4 *'   # April 30, 2:30 PM UTC (10:30 AM EDT)
    - cron: '30 14 1 6 *'    # June 1
    - cron: '30 14 31 7 *'   # July 31
  workflow_dispatch:
```

Steps:
1. Checkout repo
2. Install dependencies
3. Run `fetch_pitcher_snapshot(season=current_year, snapshot_date=today)`
4. Commit `data/raw/pitcher_snapshot_YYYY-MM-DD.csv` as `github-actions[bot]`

The `workflow_dispatch` trigger allows immediate backfilling of past snapshot dates (e.g. April 30 and June 1 if this ships in late May).

**`.gitignore` change:**
```
!data/raw/pitcher_snapshot_*.csv
```
Added alongside the existing `!data/raw/historical_*.csv` exception.

## What Does NOT Change

- `fetch_pitcher_stats(game_date)` — inference still uses this (daily cache); gains the 30 IP filter so low-IP pitchers are excluded from output, making NaN the natural result of the left join in `features.py`. No interface change.
- `fetch_probable_starters(game_date)` — untouched
- `features.py` — inference feature engineering untouched
- `edge_finder.py` — untouched
- `pipeline.run()` — untouched
- `feedback.py` / feedback loop — untouched; retraining benefits automatically once snapshots exist
- `daily.yml` workflow — untouched

## Data Flow (training path, after this change)

```
historical_YYYY.csv
    │
    ├── end-of-season team stats (fetch_stats, Sept 28) ──────────────────┐
    │                                                                       │
    ├── rolling stats (compute_rolling_stats) ──────────────────────────── ┤
    │                                                                       ├─► _build_season()
    └── time-matched pitcher snapshots                                      │
        pitcher_snapshot_YYYY-04-30.csv  ─┐                                │
        pitcher_snapshot_YYYY-06-01.csv  ─┤─ join by game_date ────────────┘
        pitcher_snapshot_YYYY-07-31.csv  ─┘
        pitcher_snapshot_YYYY-09-28.csv (or fallback to fetch_pitcher_stats)
```

## IP Threshold

`MIN_PITCHER_IP = 30` in `config.py`. Applied inside `fetch_pitcher_snapshot` at write time — pitchers below threshold are excluded from the snapshot file. Effect: early-season pitchers with few innings appear as NaN in training joins, consistent with how ~49% of training rows already have NaN pitcher stats.

The same threshold is applied at inference time inside `fetch_pitcher_stats` — if a probable starter has < 30 IP today, their stats are NaN-filled rather than passed to the model.

## Testing

Three new test areas (all existing 183 tests continue to pass):

1. **`fetch_pitcher_snapshot` unit test** — assert pitchers < 30 IP excluded, file named `pitcher_snapshot_*.csv`
2. **Snapshot join logic test** — synthetic game dates + mock snapshot files, assert correct snapshot selected per game date and NaN for pre-April-30 games
3. **`_build_season` integration test** — mock snapshot files, assert mid-season game rows get mid-season pitcher stats (not end-of-season)

## Retrain

After this change ships and snapshot files are backfilled for all historical seasons, trigger a manual retrain (`build_training_set(force=True)` + `model.train()` + `model.save_model()`). The existing feedback loop will use this retrained model going forward.

## Success Criteria

- No edges with `model_prob > 0.80` on a typical mid-season day
- Total daily edges drops (expected: 0–2 on most days vs current 3–4)
- Edges that do surface have `model_prob` in a credible range (0.55–0.75) consistent with a near-coinflip sport
