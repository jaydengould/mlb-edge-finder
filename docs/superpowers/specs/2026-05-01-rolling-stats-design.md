# Phase 6: Rolling Window Team Stats Design

**Date:** 2026-05-01
**Scope:** New `rolling_stats.py` module + additive changes to `training_data.py` and `features.py`

---

## Problem

`training_data._build_season()` joins a single end-of-season stat snapshot (September 28) to every game in the season — meaning April games get September stats. This is temporally incorrect and a known simplification. Rolling stats replace this with form-based features derived from actual game results.

---

## Decisions

| Question | Decision |
|---|---|
| Data source for rolling stats | Existing `historical_YYYY.csv` game results (no new API or paid source) |
| Rolling features | runs_scored, runs_allowed, win_pct, run_diff — all per-game averages |
| Window size | 15 games |
| Season boundary | Season-only lookback; no cross-season fill |
| Small windows (< 15 games available) | `min_periods=1` — use whatever games are available |
| First game of season (no prior games) | NaN — left as-is; XGBoost handles NaN natively |
| Relationship to existing FanGraphs/MLB API stats | Additive — rolling stats are 8 new columns, existing stats unchanged |
| FanGraphs dependency | None — rolling stats computed entirely from cached historical CSVs |
| `pipeline.run()` changes | None — `build_features()` is self-contained |

---

## New Module: `rolling_stats.py`

`src/mlb_edge_finder/rolling_stats.py`

### Private helper

**`_reshape_to_team_games(historical_df)`**

Reshapes from one-row-per-game to one-row-per-team-game. Each game becomes two rows:
- Home team: `runs_scored=home_score`, `runs_allowed=away_score`, `win=home_win`
- Away team: `runs_scored=away_score`, `runs_allowed=home_score`, `win=1-home_win`

Maps `home_name`/`away_name` → `team_abbr` via `HISTORICAL_NAME_TO_ABBR` (imported from `training_data`). Drops unmapped teams with a warning log. Returns DataFrame sorted by `(team_abbr, game_date)`.

Output columns: `team_abbr`, `game_date`, `runs_scored`, `runs_allowed`, `win`

### Public functions

**`compute_rolling_stats(historical_df, window=15) → pd.DataFrame`**

For training. Calls `_reshape_to_team_games`, then groups by `team_abbr` and computes rolling mean with `min_periods=1` for each metric. Applies `shift(1)` so each game's rolling stats reflect only games played before it. First game of the season per team has NaN rolling stats.

Returns DataFrame with columns: `team_abbr`, `game_date`, `rolling_runs_scored`, `rolling_runs_allowed`, `rolling_win_pct`, `rolling_run_diff`

**`latest_rolling_stats(historical_df, window=15) → pd.DataFrame`**

For inference. Same computation as `compute_rolling_stats` but without `shift(1)` — all completed games are included. Takes the last row per team via `groupby("team_abbr").last().reset_index()`. Returns one row per team with their current form stats.

Same output columns as above, one row per `team_abbr`.

---

## Changes to `training_data.py`

`_build_season()` gets two new steps after the existing FanGraphs/MLB API stats join:

1. Call `compute_rolling_stats(hist, window=15)` using the already-loaded `hist` DataFrame.
2. Double-join rolling stats with `home_`/`away_` prefixes:
   - Rename `team_abbr → home_abbr`, prefix cols with `home_`; merge on `["home_abbr", "game_date"]`
   - Rename `team_abbr → away_abbr`, prefix cols with `away_`; merge on `["away_abbr", "game_date"]`
   - Both merges use `how="left"` to preserve rows with NaN rolling stats

Output gains 8 new columns: `home_rolling_runs_scored`, `home_rolling_runs_allowed`, `home_rolling_win_pct`, `home_rolling_run_diff`, plus `away_` equivalents.

`build_training_set()` and `load_training_set()` signatures unchanged. Callers must pass `force=True` to rebuild the training CSV with rolling columns.

---

## Changes to `features.py`

`build_features(game_date)` gets one new block inserted after loading cached odds and stats:

1. Call `fetch_historical(game_date.year)` — cache-first; fetches from statsapi if not cached.
2. Call `latest_rolling_stats(hist_df, window=15)` → one row per `team_abbr`.
3. Double-join rolling stats on `team_abbr` only (no `game_date` — today's games haven't happened):
   - Rename `team_abbr → home_abbr`, prefix cols with `home_`; merge on `"home_abbr"`
   - Rename `team_abbr → away_abbr`, prefix cols with `away_`; merge on `"away_abbr"`

If a team has no completed games (e.g. opening day), rolling stats are NaN — not an error.

`load_features()` and `pipeline.run()` are unchanged.

---

## Error Handling

| Scenario | Behaviour |
|---|---|
| `fetch_historical()` statsapi failure | `RuntimeError` propagates — no silent fallback |
| Unmapped team name in `_reshape_to_team_games()` | Warning log, row dropped |
| NaN rolling stats (first game of season) | Left as NaN — XGBoost handles natively |
| Team with no completed games (opening day) | NaN rolling stats — not an error |

---

## Tests

### New: `tests/test_rolling_stats.py`

| Test | Scenario | Assert |
|---|---|---|
| `test_compute_rolling_stats_shift` | 2 teams, 3 games each | Game 1 of season has NaN rolling stats; game 3 reflects only games 1-2 |
| `test_compute_rolling_stats_window` | 1 team, 20 games | After game 16, rolling average uses only last 15 games |
| `test_compute_rolling_stats_min_periods` | 1 team, 3 games | No error raised; game 2 uses 1-game average |
| `test_latest_rolling_stats_one_row_per_team` | 30 teams, multiple games | Exactly 30 rows, no duplicate team_abbr |
| `test_latest_rolling_stats_includes_last_game` | 1 team, 3 games | Latest stats include game 3's result (no shift) |

### Additions to existing test files

| File | Test | Assert |
|---|---|---|
| `tests/test_training_data.py` | `test_build_training_set_includes_rolling_cols` | `home_rolling_runs_scored` and `away_rolling_run_diff` in output columns |
| `tests/test_features.py` | `test_build_features_includes_rolling_cols` | `home_rolling_runs_scored` and `away_rolling_run_diff` in output columns |
