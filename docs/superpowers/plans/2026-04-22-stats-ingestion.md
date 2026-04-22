# Stats Ingestion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `fetch_stats()`, `load_cached_stats()`, and `_build_stats_df()` in `stats_ingestion.py`, plus `ODDS_NAME_TO_ABBR`, so the pipeline can produce a per-team stats CSV ready for the features join.

**Architecture:** `_build_stats_df(season)` calls pybaseball's `team_batting` and `team_pitching`, validates columns, renames to snake_case, and merges on `team_abbr`. `fetch_stats` wraps it with cache-first logic (mirrors `odds_ingestion.fetch_odds`). `ODDS_NAME_TO_ABBR` is a module-level dict all 30 teams, used by `features.py` to normalize odds team names before the join.

**Tech Stack:** Python 3.10+, pybaseball 2.2.7, pandas, pytest

---

## Files

| Action | Path | Responsibility |
|---|---|---|
| Modify | `src/mlb_edge_finder/stats_ingestion.py` | Full implementation |
| Modify | `tests/test_stats_ingestion.py` | Updated + new smoke tests |

---

### Task 1: Update smoke tests to match new signatures

**Files:**
- Modify: `tests/test_stats_ingestion.py`

- [ ] **Step 1: Replace the test file contents**

```python
"""Smoke tests: stats_ingestion exposes expected public API."""
import inspect


def test_fetch_stats_signature():
    """fetch_stats should accept game_date and force."""
    from mlb_edge_finder import stats_ingestion
    assert callable(stats_ingestion.fetch_stats)
    sig = inspect.signature(stats_ingestion.fetch_stats)
    assert "game_date" in sig.parameters
    assert "force" in sig.parameters


def test_load_cached_stats_signature():
    """load_cached_stats should accept game_date."""
    from mlb_edge_finder import stats_ingestion
    assert callable(stats_ingestion.load_cached_stats)
    sig = inspect.signature(stats_ingestion.load_cached_stats)
    assert "game_date" in sig.parameters


def test_odds_name_to_abbr_exists():
    """ODDS_NAME_TO_ABBR should be a dict with 30 entries."""
    from mlb_edge_finder import stats_ingestion
    assert isinstance(stats_ingestion.ODDS_NAME_TO_ABBR, dict)
    assert len(stats_ingestion.ODDS_NAME_TO_ABBR) == 30
```

- [ ] **Step 2: Run tests to verify they fail (stub not yet updated)**

```bash
pytest tests/test_stats_ingestion.py -v
```

Expected: `test_fetch_stats_signature` FAILS (still checks `start_date`/`end_date`), `test_odds_name_to_abbr_exists` FAILS (`ODDS_NAME_TO_ABBR` not defined).

- [ ] **Step 3: Commit the updated tests**

```bash
git add tests/test_stats_ingestion.py
git commit -m "test: update stats_ingestion smoke tests for new signatures"
```

---

### Task 2: Add `ODDS_NAME_TO_ABBR` and stub the new signatures

**Files:**
- Modify: `src/mlb_edge_finder/stats_ingestion.py`

- [ ] **Step 1: Replace the entire file with the new skeleton**

```python
"""Fetch and cache team and pitcher stats via pybaseball."""
import logging
from datetime import date

import pandas as pd
from pybaseball import team_batting, team_pitching

from mlb_edge_finder import config

logger = logging.getLogger(__name__)

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
    "Athletics": "ATH",
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


def _build_stats_df(season: int) -> pd.DataFrame:
    raise NotImplementedError


def fetch_stats(game_date: date, force: bool = False) -> pd.DataFrame:
    raise NotImplementedError


def load_cached_stats(game_date: date) -> pd.DataFrame:
    raise NotImplementedError
```

- [ ] **Step 2: Run smoke tests — all three should pass now**

```bash
pytest tests/test_stats_ingestion.py -v
```

Expected: all 3 PASS.

- [ ] **Step 3: Commit**

```bash
git add src/mlb_edge_finder/stats_ingestion.py
git commit -m "feat: add ODDS_NAME_TO_ABBR and updated function stubs"
```

---

### Task 3: Implement `load_cached_stats`

**Files:**
- Modify: `src/mlb_edge_finder/stats_ingestion.py`

- [ ] **Step 1: Replace the `load_cached_stats` stub**

```python
def load_cached_stats(game_date: date) -> pd.DataFrame:
    """Load previously fetched stats from DATA_RAW_DIR/stats_YYYY-MM-DD.csv.

    Args:
        game_date: The date whose cached CSV to load.

    Returns:
        DataFrame with the same schema as fetch_stats().

    Raises:
        FileNotFoundError: If no cached file exists for the given date.
    """
    cache_path = config.DATA_RAW_DIR / f"stats_{game_date}.csv"
    if not cache_path.exists():
        raise FileNotFoundError(f"No cached stats for {game_date}: {cache_path}")
    return pd.read_csv(cache_path)
```

- [ ] **Step 2: Run the full test suite to confirm no regressions**

```bash
pytest tests/ -v
```

Expected: all 17+ tests PASS.

- [ ] **Step 3: Commit**

```bash
git add src/mlb_edge_finder/stats_ingestion.py
git commit -m "feat: implement load_cached_stats"
```

---

### Task 4: Implement `_build_stats_df`

**Files:**
- Modify: `src/mlb_edge_finder/stats_ingestion.py`

- [ ] **Step 1: Replace the `_build_stats_df` stub**

```python
_BATTING_COLS = ["Team", "AVG", "OBP", "SLG", "OPS", "R", "wOBA", "wRC+"]
_PITCHING_COLS = ["Team", "ERA", "WHIP", "FIP", "K/9", "BB/9"]

_BATTING_RENAME = {
    "Team": "team_abbr",
    "AVG": "bat_avg",
    "OBP": "obp",
    "SLG": "slg",
    "OPS": "ops",
    "R": "runs",
    "wOBA": "w_oba",
    "wRC+": "bat_wrc_plus",
}

_PITCHING_RENAME = {
    "Team": "team_abbr",
    "ERA": "era",
    "WHIP": "whip",
    "FIP": "fip",
    "K/9": "k_per_9",
    "BB/9": "bb_per_9",
}


def _build_stats_df(season: int) -> pd.DataFrame:
    bat = team_batting(season, season, qual=0)
    if bat.empty:
        raise RuntimeError(f"team_batting returned no data for season {season}")
    missing_bat = [c for c in _BATTING_COLS if c not in bat.columns]
    if missing_bat:
        raise RuntimeError(f"team_batting missing columns: {missing_bat}")

    pit = team_pitching(season, season, qual=0)
    if pit.empty:
        raise RuntimeError(f"team_pitching returned no data for season {season}")
    missing_pit = [c for c in _PITCHING_COLS if c not in pit.columns]
    if missing_pit:
        raise RuntimeError(f"team_pitching missing columns: {missing_pit}")

    bat = bat[_BATTING_COLS].rename(columns=_BATTING_RENAME)
    pit = pit[_PITCHING_COLS].rename(columns=_PITCHING_RENAME)

    df = bat.merge(pit, on="team_abbr", how="inner")
    logger.debug("Built stats DataFrame: %d teams, %d columns", len(df), len(df.columns))
    return df
```

Place the `_BATTING_COLS`, `_PITCHING_COLS`, `_BATTING_RENAME`, `_PITCHING_RENAME` module-level constants immediately before `_build_stats_df` in the file.

- [ ] **Step 2: Run the full test suite**

```bash
pytest tests/ -v
```

Expected: all tests PASS (smoke tests don't call `_build_stats_df` directly).

- [ ] **Step 3: Commit**

```bash
git add src/mlb_edge_finder/stats_ingestion.py
git commit -m "feat: implement _build_stats_df with column validation"
```

---

### Task 5: Implement `fetch_stats`

**Files:**
- Modify: `src/mlb_edge_finder/stats_ingestion.py`

- [ ] **Step 1: Replace the `fetch_stats` stub**

```python
def fetch_stats(game_date: date, force: bool = False) -> pd.DataFrame:
    """Fetch season-to-date team batting and pitching stats for a given date.

    Calls pybaseball.team_batting() and pybaseball.team_pitching() for the
    season year derived from game_date. Writes the result to
    DATA_RAW_DIR/stats_YYYY-MM-DD.csv. If a cached file already exists and
    force=False, returns the cached data without calling pybaseball.

    Args:
        game_date: The date for which to fetch stats. Season year is
            game_date.year.
        force: If True, re-fetch from pybaseball even if a cache file exists.

    Returns:
        DataFrame with columns: team_abbr, bat_avg, obp, slg, ops, runs,
        w_oba, bat_wrc_plus, era, whip, fip, k_per_9, bb_per_9.
        One row per team.

    Raises:
        RuntimeError: If pybaseball returns no data or expected columns are
            missing.
    """
    cache_path = config.DATA_RAW_DIR / f"stats_{game_date}.csv"
    if cache_path.exists() and not force:
        logger.debug("Cache hit for %s, loading from disk", game_date)
        return load_cached_stats(game_date)

    df = _build_stats_df(game_date.year)

    config.DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path, index=False)
    logger.info("Wrote %d rows to %s", len(df), cache_path)

    return df
```

- [ ] **Step 2: Run the full test suite**

```bash
pytest tests/ -v
```

Expected: all tests PASS.

- [ ] **Step 3: Commit**

```bash
git add src/mlb_edge_finder/stats_ingestion.py
git commit -m "feat: implement fetch_stats with cache-first pattern"
```

---

### Task 6: Final verification

- [ ] **Step 1: Run the complete test suite one final time**

```bash
pytest tests/ -v
```

Expected: all tests PASS, no warnings about missing signatures.

- [ ] **Step 2: Verify module imports cleanly**

```bash
python3 -c "
from mlb_edge_finder import stats_ingestion
import inspect
print('ODDS_NAME_TO_ABBR entries:', len(stats_ingestion.ODDS_NAME_TO_ABBR))
print('fetch_stats params:', list(inspect.signature(stats_ingestion.fetch_stats).parameters))
print('load_cached_stats params:', list(inspect.signature(stats_ingestion.load_cached_stats).parameters))
print('OK')
"
```

Expected output:
```
ODDS_NAME_TO_ABBR entries: 30
fetch_stats params: ['game_date', 'force']
load_cached_stats params: ['game_date']
OK
```

- [ ] **Step 3: Commit final state if any files are uncommitted**

```bash
git status
# If clean, nothing to do. Otherwise:
git add src/mlb_edge_finder/stats_ingestion.py tests/test_stats_ingestion.py
git commit -m "feat: stats_ingestion complete — fetch_stats, load_cached_stats, ODDS_NAME_TO_ABBR"
```
