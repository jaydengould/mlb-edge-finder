# Phase 7 — Starting Pitcher Features Design

**Date:** 2026-05-02  
**Status:** Approved

## Overview

Add per-game starting pitcher stats to both the training dataset and the daily inference feature set. The model currently uses team-level batting and pitching aggregates plus rolling team form stats. Knowing today's specific starter (rather than the team average) is the single strongest remaining predictor of game outcome not yet in the model.

## Decisions

- **Granularity:** Season-to-date aggregate stats per pitcher (not rolling per-start). Simpler, consistent with how team stats work. Rolling starts deferred to a future phase.
- **Data source:** statsapi only — no FanGraphs dependency. FanGraphs has been unreliable throughout the project. FanGraphs is retained in `stats_ingestion.py` as-is (already written and tested); it is simply not added to the new pitcher module.
- **Join key:** Pitcher ID (integer from statsapi), not pitcher name string. Avoids name normalization issues entirely.
- **xFIP:** Dropped — requires league-average HR/FB rate, only available from FanGraphs.

## New Module: `pitcher_ingestion.py`

Three public functions, all following existing conventions:

### `fetch_pitcher_stats(game_date, force=False)`

Fetches season-to-date stats for all pitchers on all 30 team rosters via statsapi. Season year derived from `game_date`. Implementation: fetch each team's 40-man roster (30 calls), collect all pitcher IDs, then batch-fetch season pitching stats. Writes to `data/raw/pitcher_stats_YYYY-MM-DD.csv`. Cache-first unless `force=True`. Raises `RuntimeError` if statsapi fails.

Building the full season snapshot (not just starters) means both training and inference simply look up from the same cached CSV — no coupling between this module and historical data.

**Output columns:** `pitcher_id` (int), `pitcher_name`, `era`, `whip`, `k_per_9`, `bb_per_9`, `ip`, `fip_computed`

`fip_computed` uses the same formula already in `stats_ingestion.py`: `(13*HR + 3*BB - 2*K) / IP + 3.15`.

### `load_cached_pitcher_stats(game_date)`

Loads `data/raw/pitcher_stats_YYYY-MM-DD.csv`. Raises `FileNotFoundError` if absent.

### `fetch_probable_starters(game_date)`

Calls `statsapi.schedule(start_date, end_date)` for the given date. Extracts `home_probable_pitcher_id` / `away_probable_pitcher_id`. Maps team names → abbreviations via `HISTORICAL_NAME_TO_ABBR` (statsapi uses full team names matching that mapping, not the Odds API format). Returns a DataFrame with columns: `home_abbr`, `away_abbr`, `home_starter_id`, `away_starter_id`. Not cached — cheap single-date call, starters can change day-of.

## Changes to `historical_ingestion.py`

Add `home_starter_id` and `away_starter_id` to `_KEEP_COLS`. statsapi already returns these in the schedule response — they are currently discarded. Games where statsapi did not record a starter receive `NaN`.

This is a breaking change to the historical CSV schema. Existing cached historical CSVs must be re-fetched (`force=True`) to pick up the new columns.

## Training Join Flow (`training_data.py`)

In `_build_season()`, after the existing team stats and rolling stats joins:

1. Call `fetch_pitcher_stats(date(season, 9, 28))` — same end-of-season snapshot pattern as `fetch_stats()`, fully cached.
2. Double-join on `pitcher_id`:
   - `home_starter_id` → `home_era`, `home_whip`, `home_k_per_9`, `home_bb_per_9`, `home_ip`, `home_fip_computed`
   - `away_starter_id` → same columns with `away_` prefix

Rows where starter ID is NaN → NaN pitcher columns → XGBoost handles natively. No rows dropped.

`model.py` — add `home_starter_id`, `away_starter_id`, `home_pitcher_name`, and `away_pitcher_name` to `NON_FEATURE_COLS` (metadata, not features). The pitcher name columns land in the DataFrame after the double-join and must be explicitly excluded.

## Inference Join Flow (`features.py`)

`build_features(game_date)` gains two steps after the existing rolling stats join:

1. Call `fetch_probable_starters(game_date)` — merge onto main DataFrame on `home_abbr` / `away_abbr` to add `home_starter_id`, `away_starter_id`.
2. Load `load_cached_pitcher_stats(game_date)` — raises `RuntimeError` with "run fetch_pitcher_stats() first" if absent. Double-join on `pitcher_id` with `home_`/`away_` prefixes.

## `pipeline.run()` Changes

Add `fetch_pitcher_stats(game_date)` as a pre-step before `build_features()`, so the pitcher stats cache is warm. Mirrors how `fetch_odds()` and `fetch_stats()` are called today.

## Error Handling and NaN Degradation

| Scenario | Behaviour |
|---|---|
| No probable starter recorded | `home_starter_id` / `away_starter_id` is NaN → NaN pitcher columns → XGBoost handles natively |
| Pitcher not in stats CSV (call-up, no season stats) | Join miss → NaN pitcher columns → debug log counting misses |
| `load_cached_pitcher_stats` before cache exists | `RuntimeError`: "run fetch_pitcher_stats() first" |
| statsapi fails in `fetch_pitcher_stats` | `RuntimeError` propagates |
| `fetch_probable_starters` fails | `RuntimeError` propagates |

No new error handling patterns — every case maps to an existing convention.

## File Changes Summary

| File | Change |
|---|---|
| `src/mlb_edge_finder/pitcher_ingestion.py` | New module |
| `src/mlb_edge_finder/historical_ingestion.py` | Add `home_starter_id`, `away_starter_id` to `_KEEP_COLS` |
| `src/mlb_edge_finder/training_data.py` | Add pitcher stats fetch + double-join in `_build_season()` |
| `src/mlb_edge_finder/features.py` | Add `fetch_probable_starters` + pitcher stats double-join |
| `src/mlb_edge_finder/pipeline.py` | Add `fetch_pitcher_stats` pre-step |
| `src/mlb_edge_finder/model.py` | Add `home_starter_id`, `away_starter_id` to `NON_FEATURE_COLS` |
| `tests/test_pitcher_ingestion.py` | New test file |
| `notebooks/01_exploration.ipynb` | Update Section 4b and add Phase 7 section |

## Deferred

- Rolling last-N-starts per pitcher (Phase 9 candidate)
- xFIP (requires FanGraphs)
- FanGraphs pitcher stats fallback
