# Stats Ingestion Design

**Date:** 2026-04-22
**Module:** `src/mlb_edge_finder/stats_ingestion.py`

## Overview

Implement `fetch_stats()` and `load_cached_stats()` in `stats_ingestion.py`. Pulls season-to-date team batting and pitching stats from FanGraphs via pybaseball, merges them into one row per team, and writes to `data/raw/stats_YYYY-MM-DD.csv`. Includes `ODDS_NAME_TO_ABBR` dict for the downstream features join.

---

## Public API

```python
ODDS_NAME_TO_ABBR: dict[str, str]

def fetch_stats(game_date: date, force: bool = False) -> pd.DataFrame: ...
def load_cached_stats(game_date: date) -> pd.DataFrame: ...
```

- `fetch_stats` is cache-first: returns cached CSV unless `force=True`
- Season year is derived from `game_date.year`
- Cache file is `data/raw/stats_{game_date}.csv`
- `load_cached_stats` raises `FileNotFoundError` if file is absent

---

## Private Helper

```python
def _build_stats_df(season: int) -> pd.DataFrame: ...
```

Steps:
1. Call `team_batting(season, season)` and `team_pitching(season, season)`
2. Raise `RuntimeError` if either returns an empty DataFrame
3. Check all expected source columns are present; raise `RuntimeError("team_batting missing columns: {missing}")` / `RuntimeError("team_pitching missing columns: {missing}")` if any are absent
4. Select columns:
   - Batting: `Team`, `AVG`, `OBP`, `SLG`, `OPS`, `R`, `wOBA`, `wRC+`
   - Pitching: `Team`, `ERA`, `WHIP`, `FIP`, `K/9`, `BB/9`
5. Rename `Team` → `team_abbr` in **both** DataFrames before the merge
6. Rename stats to snake_case:
   - `AVG` → `bat_avg`, `OBP` → `obp`, `SLG` → `slg`, `OPS` → `ops`, `R` → `runs`, `wOBA` → `w_oba`, `wRC+` → `bat_wrc_plus`
   - `ERA` → `era`, `WHIP` → `whip`, `FIP` → `fip`, `K/9` → `k_per_9`, `BB/9` → `bb_per_9`
7. Merge both frames on `team_abbr` → 13 columns total (`team_abbr` + 12 stats)

---

## ODDS_NAME_TO_ABBR

Module-level dict mapping all 30 Odds API full team names to FanGraphs abbreviations:

```python
ODDS_NAME_TO_ABBR: dict[str, str] = {
    "Arizona Diamondbacks": "ARI",
    "Atlanta Braves": "ATL",
    "Baltimore Orioles": "BAL",
    "Boston Red Sox": "BOS",
    "Chicago Cubs": "CHC",
    "Chicago White Sox": "CWS",
    "Cincinnati Reds": "CIN",
    "Cleveland Guardians": "CLE",
    "Colorado Rockies": "COL",
    "Detroit Tigers": "DET",
    "Houston Astros": "HOU",
    "Kansas City Royals": "KC",
    "Los Angeles Angels": "LAA",
    "Los Angeles Dodgers": "LAD",
    "Miami Marlins": "MIA",
    "Milwaukee Brewers": "MIL",
    "Minnesota Twins": "MIN",
    "New York Mets": "NYM",
    "New York Yankees": "NYY",
    "Athletics": "ATH",         # relocated to Sacramento 2025; verify against live API
    "Philadelphia Phillies": "PHI",
    "Pittsburgh Pirates": "PIT",
    "San Diego Padres": "SD",
    "San Francisco Giants": "SF",
    "Seattle Mariners": "SEA",
    "St. Louis Cardinals": "STL",
    "Tampa Bay Rays": "TB",
    "Texas Rangers": "TEX",
    "Toronto Blue Jays": "TOR",
    "Washington Nationals": "WSH",
}
```

`features.py` imports this dict to translate `home_team`/`away_team` in the odds DataFrame to abbreviations before merging on `team_abbr`.

---

## Error Handling

| Condition | Behavior |
|---|---|
| pybaseball returns empty DataFrame | `RuntimeError("team_batting returned no data for season {season}")` |
| Expected source columns missing | `RuntimeError("team_batting missing columns: {missing}")` |
| Cache file absent in `load_cached_stats` | `FileNotFoundError` |
| No silent fallbacks | If pybaseball fails, raise — never return empty DataFrame |

---

## Output Schema

One row per team. Columns:

| Column | Source | Rename from |
|---|---|---|
| `team_abbr` | both | `Team` |
| `bat_avg` | batting | `AVG` |
| `obp` | batting | `OBP` |
| `slg` | batting | `SLG` |
| `ops` | batting | `OPS` |
| `runs` | batting | `R` |
| `w_oba` | batting | `wOBA` |
| `bat_wrc_plus` | batting | `wRC+` |
| `era` | pitching | `ERA` |
| `whip` | pitching | `WHIP` |
| `fip` | pitching | `FIP` |
| `k_per_9` | pitching | `K/9` |
| `bb_per_9` | pitching | `BB/9` |

---

## Test Updates (`test_stats_ingestion.py`)

- Update `test_fetch_stats_signature`: check for `game_date` and `force` (drop `start_date`/`end_date`)
- `test_load_cached_stats_signature`: no change needed
- Add `test_odds_name_to_abbr_exists`: assert `ODDS_NAME_TO_ABBR` is a `dict` with 30 entries
- No external API calls — smoke tests only, consistent with existing suite

---

## Data Flow

```
game_date.year
      │
      ▼
_build_stats_df(season)
  ├── team_batting(season, season)  → select + validate + rename
  ├── team_pitching(season, season) → select + validate + rename
  └── merge on team_abbr
      │
      ▼
data/raw/stats_YYYY-MM-DD.csv  (one row per team, 13 columns)
```
