# Phase 4b Design: `training_data.py`

**Date:** 2026-04-28  
**Status:** Approved

## Overview

Implements `training_data.build_training_set(seasons)` — joins end-of-season team stats to historical game results to produce a labeled training set for the XGBoost model. One snapshot of stats per season (late September), joined to every game in that season. No date-accurate rolling stats; this is a known simplification.

## Module-Level Constants

Two dicts owned entirely by `training_data.py`:

```python
HISTORICAL_NAME_TO_ABBR: dict[str, str]
```
Maps statsapi full team names to **current** abbreviations (always uses current franchise identity, regardless of historical name). Example: `"Oakland Athletics" → "ATH"` (not `"OAK"`). Covers all 30 franchises.

```python
_LEGACY_ABBR_NORMALIZE: dict[str, str]
```
Maps FanGraphs abbreviations that changed over time to the current abbreviation. Applied to the stats DataFrame before joining. Example: `{"OAK": "ATH"}`. Ensures both sides of the join use consistent, current identifiers — critical since the model will see current abbreviations at inference time.

## Public API

```python
def build_training_set(seasons: list[int], force: bool = False) -> pd.DataFrame
def load_training_set(seasons: list[int]) -> pd.DataFrame
```

- Cache-first: `build_training_set` skips rebuild if the output file exists and `force=False`.
- Raises `RuntimeError` (not `FileNotFoundError`) if historical or stats data is missing for any season, with a message directing the user to run `fetch_historical()` or `fetch_stats()` first. Consistent with `features.py` error handling.
- Output path: `data/processed/training_{min(seasons)}-{max(seasons)}.csv` (e.g. `training_2023-2025.csv`).
- `load_training_set` raises `FileNotFoundError` if the cache file does not exist.

## Data Flow (per season)

For each season in `seasons`:

1. Load `historical_YYYY.csv` via `load_cached_historical(season)` — catch its `FileNotFoundError` and re-raise as `RuntimeError` with a "run fetch_historical(season) first" message.
2. Call `fetch_stats(date(season, 9, 28))` for end-of-season stats snapshot — cache-first via existing machinery, no changes to `fetch_stats`.
3. Apply `_LEGACY_ABBR_NORMALIZE` to the stats `team_abbr` column.
4. Map `home_name`/`away_name` in the historical DataFrame to abbreviations via `HISTORICAL_NAME_TO_ABBR`. Log a warning and drop rows with unmatched names.
5. Drop `data_source` from stats — not a model feature.
6. Join stats twice using `home_`/`away_` prefixes (same double-join pattern as `features.py`).
7. Tag each row with a `season` column (the season year integer).
8. Retain: `game_date`, `season`, `home_name`, `away_name`, `home_abbr`, `away_abbr`, `home_win`, and all prefixed stat columns.

Concatenate all per-season DataFrames, write to the output CSV, return.

## Stat Columns

Whatever columns exist in the stats DataFrame flow through automatically — no hardcoded column list. This means FanGraphs-specific columns (`w_oba`, `bat_wrc_plus`, `fip`) appear in the training set when present, and are absent otherwise. Downstream code (model.py) must guard with `col in df.columns`. This is consistent with the existing pattern and makes it trivial to add new stat columns (pitcher features, park factors, etc.) without changes to this module.

## Error Handling

| Condition | Behavior |
|---|---|
| Historical CSV missing for a season | `RuntimeError`: "run fetch_historical(season) first" |
| Stats CSV missing for a season | `RuntimeError`: "run fetch_stats(date(season, 9, 28)) first" |
| Unmatched team name in historical data | Log warning, drop row (same as `features.py`) |
| Training set CSV missing on load | `FileNotFoundError` |

## Extensibility Notes

- `HISTORICAL_NAME_TO_ABBR` and `_LEGACY_ABBR_NORMALIZE` are module-level dicts — easy to extend when franchises move or rename.
- `seasons` is always a parameter — trivial to add seasons or change the training window.
- `home_name`/`away_name` are kept in the output for future joins (e.g. stadium/park factor lookups by full name).
- No stat columns are hardcoded — new data sources integrate without changes to this module.

## Files Affected

| File | Change |
|---|---|
| `src/mlb_edge_finder/training_data.py` | Create (new module) |
| `data/processed/training_YYYY-YYYY.csv` | Created at runtime (gitignored) |
