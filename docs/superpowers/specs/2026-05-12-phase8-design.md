# Phase 8: Expand Training Seasons

**Date:** 2026-05-12  
**Status:** Approved

## Goal

Expand the training dataset from 3 seasons (2023–2025) to 6 seasons (2019, 2021–2025), skipping 2020 (60-game anomaly). This roughly triples available training data (~490 → ~810 games) with no structural changes to the pipeline.

## Approach

Option A — minimal constant change. No new modules, no new abstractions.

## Files Changed

### `src/mlb_edge_finder/historical_ingestion.py`

Change `_HISTORICAL_SEASONS`:

```python
# Before
_HISTORICAL_SEASONS = [2023, 2024, 2025]

# After
_HISTORICAL_SEASONS = [2019, 2021, 2022, 2023, 2024, 2025]
```

`fetch_all_historical()` iterates this constant — no other changes needed in this file.

### `notebooks/01_exploration.ipynb`

Update the training section:

```python
# Before
seasons = [2023, 2024, 2025]

# After
seasons = [2019, 2021, 2022, 2023, 2024, 2025]
```

Update the comment above to reflect the expanded range.

### `tests/test_historical_ingestion.py`

`test_fetch_all_historical_concatenates` mocks one game per season. Update assertion:

```python
# Before
assert len(df) == 3   # 3 seasons × 1 game

# After
assert len(df) == 6   # 6 seasons × 1 game
```

## No Changes Needed

| Concern | Why it's already handled |
|---|---|
| "Cleveland Indians" name (2019–2021) | `HISTORICAL_NAME_TO_ABBR` already maps it → `CLE` |
| "Florida Marlins" name | Already mapped → `MIA` |
| "Oakland Athletics" name | Already mapped → `ATH` |
| FanGraphs `OAK` abbreviation in stats | `_LEGACY_ABBR_NORMALIZE` already remaps `OAK → ATH` |
| 2021 late season start (Apr 1) | `_SEASON_START = "03-20"` window still captures all games |
| 2022 late season start (Apr 7, lockout) | Same — no March games, window still correct |
| `_build_season()`, rolling stats, model, edge finder | Unchanged — all season-agnostic |

## Data Flow at Runtime

1. `fetch_all_historical(force=True)` — calls `statsapi.schedule` for each of the 5 new/existing seasons; writes `historical_YYYY.csv` for each
2. `build_training_set([2019, 2021, 2022, 2023, 2024, 2025], force=True)` — joins end-of-season stats + rolling stats + pitcher stats for all 6 seasons; writes `training_2019-2025.csv`
3. `model.train(training_df)` — trains on the expanded set; re-save with today's date

## Testing

- One test updated (`test_fetch_all_historical_concatenates`: 3 → 6)
- All other tests remain valid — mock-based suite is season-agnostic
- No new tests needed; the expanded constant exercises the same code paths
