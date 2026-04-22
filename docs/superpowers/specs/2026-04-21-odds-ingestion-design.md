# Odds Ingestion Module — Design Spec

**Date:** 2026-04-21  
**Module:** `src/mlb_edge_finder/odds_ingestion.py`  
**Status:** Approved

---

## Overview

Implements `fetch_odds()` and `load_cached_odds()` in `odds_ingestion.py`. Fetches MLB moneyline odds from The Odds API v4, persists a dated CSV to `data/raw/`, and returns a cleaned long-format DataFrame.

---

## Architecture

Flat module, two public functions and one private helper:

- `fetch_odds(game_date, force=False)` — public entry point
- `load_cached_odds(game_date)` — public cache reader
- `_parse_response(games, game_date)` — private parser (testable in isolation)

No class wrappers. Follows existing flat module style.

---

## API Call

**Endpoint:** `GET https://api.the-odds-api.com/v4/sports/{SPORT}/odds`

**Params:**
- `apiKey` — from `config.ODDS_API_KEY`
- `regions` — from `config.REGION` (default: `"us"`)
- `markets` — from `config.MARKET` (default: `"h2h"`)
- `dateFormat=iso`
- `oddsFormat=american`

Returns all upcoming and live games. We filter to games where `commence_time` date matches `game_date`.

---

## Caching

`fetch_odds(game_date, force=False)`:

1. Compute cache path: `config.DATA_RAW_DIR / f"odds_{game_date}.csv"`
2. If file exists and `force=False` → call `load_cached_odds(game_date)` and return
3. Otherwise → call API, parse, write CSV, return DataFrame

`load_cached_odds(game_date)`:
- Reads the dated CSV with `pd.read_csv()`
- Raises `FileNotFoundError` if the file does not exist

---

## Response Parsing (`_parse_response`)

The Odds API returns a list of game objects, each with nested `bookmakers → markets → outcomes`. The parser flattens this to one row per bookmaker per game.

**Input:** raw list of game dicts from the API, plus `game_date` for filtering  
**Output:** DataFrame with columns:

| Column | Type | Source |
|---|---|---|
| `game_id` | str | `game["id"]` |
| `home_team` | str | `game["home_team"]` |
| `away_team` | str | `game["away_team"]` |
| `home_odds_american` | int | outcome matching `home_team` |
| `away_odds_american` | int | outcome matching `away_team` |
| `bookmaker` | str | `bookmaker["key"]` |
| `commence_time` | str (ISO) | `game["commence_time"]` |

**Filtering:** Only include games where `commence_time[:10] == str(game_date)`.

**Outcome matching:** For each bookmaker's h2h market, find the outcome whose `name` matches `game["home_team"]` → `home_odds_american`; the other outcome → `away_odds_american`.

Games with no bookmakers or no h2h market are skipped with a debug log.

---

## Error Handling

| Condition | Behavior |
|---|---|
| `ODDS_API_KEY` is empty string | Log error + raise `RuntimeError` before making the request |
| API returns non-200 status | Log `f"Odds API returned {status}: {text}"` + raise `RuntimeError` |
| No games found for `game_date` | Return empty DataFrame (not an error) |
| No edges after filtering | Return empty DataFrame (logged as warning by caller) |
| Cache file missing in `load_cached_odds` | Raise `FileNotFoundError` |

---

## Logging

Uses `logger = logging.getLogger(__name__)` (already in scaffold). No handler configuration — only `config.setup_logging()` does that.

Key log points:
- `DEBUG` — cache hit / skip
- `INFO` — successful fetch, number of rows written
- `WARNING` — zero games matched `game_date`
- `ERROR` — API failure (before raising)

---

## Testing Notes

The existing smoke tests verify signatures only. The private `_parse_response()` helper can be unit tested with a fixture dict (no API call needed) in a future test expansion.
