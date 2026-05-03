# Starting Pitcher Features Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add individual starting pitcher season stats to both the training dataset and daily inference feature set, using statsapi as the sole data source.

**Architecture:** A new `pitcher_ingestion.py` module owns all pitcher data fetching. `historical_ingestion.py` is enriched to persist probable pitcher names from the existing statsapi schedule response. Both `training_data.py` and `features.py` gain a double-join on pitcher name using a `home_sp_*`/`away_sp_*` column prefix to avoid colliding with existing team-level pitching stats.

**Tech Stack:** Python 3.10+, statsapi, pandas, pytest, unittest.mock

---

## File Map

| File | Change |
|---|---|
| `src/mlb_edge_finder/pitcher_ingestion.py` | **CREATE** — `fetch_pitcher_stats`, `load_cached_pitcher_stats`, `fetch_probable_starters` |
| `tests/test_pitcher_ingestion.py` | **CREATE** — tests for all three public functions |
| `src/mlb_edge_finder/historical_ingestion.py` | **MODIFY** — add `home_starter_name`/`away_starter_name` columns |
| `tests/test_historical_ingestion.py` | **MODIFY** — update column assertions, add starter name tests |
| `src/mlb_edge_finder/model.py` | **MODIFY** — extend `NON_FEATURE_COLS` |
| `src/mlb_edge_finder/training_data.py` | **MODIFY** — add pitcher stats fetch + double-join |
| `tests/test_training_data.py` | **MODIFY** — add pitcher join coverage |
| `src/mlb_edge_finder/features.py` | **MODIFY** — add probable starters + pitcher stats double-join |
| `tests/test_features.py` | **MODIFY** — add pitcher join coverage |
| `src/mlb_edge_finder/pipeline.py` | **MODIFY** — add `fetch_pitcher_stats` pre-step |
| `notebooks/01_exploration.ipynb` | **MODIFY** — update Phase 4b, add Phase 7 section |
| `CLAUDE.md` | **MODIFY** — mark Phase 7 complete, update roadmap |

---

## Task 1: `pitcher_ingestion.py` — `fetch_pitcher_stats` + `load_cached_pitcher_stats`

**Files:**
- Create: `tests/test_pitcher_ingestion.py`
- Create: `src/mlb_edge_finder/pitcher_ingestion.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_pitcher_ingestion.py
"""Tests for pitcher_ingestion module."""
import inspect
from datetime import date
from unittest.mock import patch

import pandas as pd
import pytest


def _make_stats_response(era="3.50", whip="1.20", k9="9.0", bb9="3.0",
                          ip="150.0", hr=15, bb=45, k=135):
    return {
        "stats": [{
            "splits": [{
                "player": {"id": 592789, "fullName": "Gerrit Cole"},
                "stat": {
                    "era": era, "whip": whip,
                    "strikeoutsPer9Inn": k9, "walksPer9Inn": bb9,
                    "inningsPitched": ip,
                    "homeRuns": hr, "baseOnBalls": bb, "strikeOuts": k,
                },
            }]
        }]
    }


def test_fetch_pitcher_stats_signature():
    from mlb_edge_finder import pitcher_ingestion
    assert callable(pitcher_ingestion.fetch_pitcher_stats)
    sig = inspect.signature(pitcher_ingestion.fetch_pitcher_stats)
    assert "game_date" in sig.parameters
    assert "force" in sig.parameters


def test_load_cached_pitcher_stats_signature():
    from mlb_edge_finder import pitcher_ingestion
    assert callable(pitcher_ingestion.load_cached_pitcher_stats)
    sig = inspect.signature(pitcher_ingestion.load_cached_pitcher_stats)
    assert "game_date" in sig.parameters


def test_load_cached_pitcher_stats_raises_when_missing(tmp_path):
    from mlb_edge_finder import pitcher_ingestion
    with patch("mlb_edge_finder.pitcher_ingestion.config.DATA_RAW_DIR", tmp_path):
        with pytest.raises(FileNotFoundError):
            pitcher_ingestion.load_cached_pitcher_stats(date(2025, 9, 28))


def test_fetch_pitcher_stats_returns_expected_columns(tmp_path):
    from mlb_edge_finder import pitcher_ingestion
    with patch("mlb_edge_finder.pitcher_ingestion.statsapi.get",
               return_value=_make_stats_response()), \
         patch("mlb_edge_finder.pitcher_ingestion.config.DATA_RAW_DIR", tmp_path):
        df = pitcher_ingestion.fetch_pitcher_stats(date(2025, 9, 28), force=True)
    assert set(df.columns) == {
        "pitcher_id", "pitcher_name", "era", "whip",
        "k_per_9", "bb_per_9", "ip", "fip_computed",
    }


def test_fetch_pitcher_stats_computes_fip(tmp_path):
    from mlb_edge_finder import pitcher_ingestion
    # fip = (13*15 + 3*45 - 2*135) / 150.0 + 3.15
    # = (195 + 135 - 270) / 150 + 3.15 = 60/150 + 3.15 = 0.4 + 3.15 = 3.55
    with patch("mlb_edge_finder.pitcher_ingestion.statsapi.get",
               return_value=_make_stats_response(ip="150.0", hr=15, bb=45, k=135)), \
         patch("mlb_edge_finder.pitcher_ingestion.config.DATA_RAW_DIR", tmp_path):
        df = pitcher_ingestion.fetch_pitcher_stats(date(2025, 9, 28), force=True)
    assert abs(df.iloc[0]["fip_computed"] - 3.55) < 0.01


def test_fetch_pitcher_stats_skips_zero_ip(tmp_path):
    from mlb_edge_finder import pitcher_ingestion
    response = {
        "stats": [{
            "splits": [
                {"player": {"id": 1, "fullName": "A"}, "stat": {"inningsPitched": "0", "era": "0", "whip": "0", "strikeoutsPer9Inn": "0", "walksPer9Inn": "0", "homeRuns": 0, "baseOnBalls": 0, "strikeOuts": 0}},
                {"player": {"id": 2, "fullName": "B"}, "stat": {"inningsPitched": "50.1", "era": "3.50", "whip": "1.20", "strikeoutsPer9Inn": "9.0", "walksPer9Inn": "3.0", "homeRuns": 5, "baseOnBalls": 15, "strikeOuts": 50}},
            ]
        }]
    }
    with patch("mlb_edge_finder.pitcher_ingestion.statsapi.get", return_value=response), \
         patch("mlb_edge_finder.pitcher_ingestion.config.DATA_RAW_DIR", tmp_path):
        df = pitcher_ingestion.fetch_pitcher_stats(date(2025, 9, 28), force=True)
    assert len(df) == 1
    assert df.iloc[0]["pitcher_name"] == "B"


def test_fetch_pitcher_stats_cache_first(tmp_path):
    from mlb_edge_finder import pitcher_ingestion
    cached = pd.DataFrame([{
        "pitcher_id": 1, "pitcher_name": "Cached Cole",
        "era": 3.0, "whip": 1.1, "k_per_9": 9.0, "bb_per_9": 2.0,
        "ip": 100.0, "fip_computed": 3.2,
    }])
    cache_path = tmp_path / "pitcher_stats_2025-09-28.csv"
    cached.to_csv(cache_path, index=False)
    with patch("mlb_edge_finder.pitcher_ingestion.statsapi.get") as mock_get, \
         patch("mlb_edge_finder.pitcher_ingestion.config.DATA_RAW_DIR", tmp_path):
        df = pitcher_ingestion.fetch_pitcher_stats(date(2025, 9, 28))
    mock_get.assert_not_called()
    assert df.iloc[0]["pitcher_name"] == "Cached Cole"


def test_fetch_pitcher_stats_raises_on_api_failure(tmp_path):
    from mlb_edge_finder import pitcher_ingestion
    with patch("mlb_edge_finder.pitcher_ingestion.statsapi.get",
               side_effect=Exception("timeout")), \
         patch("mlb_edge_finder.pitcher_ingestion.config.DATA_RAW_DIR", tmp_path):
        with pytest.raises(RuntimeError, match="statsapi failed"):
            pitcher_ingestion.fetch_pitcher_stats(date(2025, 9, 28), force=True)


def test_fetch_pitcher_stats_raises_on_empty_response(tmp_path):
    from mlb_edge_finder import pitcher_ingestion
    with patch("mlb_edge_finder.pitcher_ingestion.statsapi.get",
               return_value={"stats": [{"splits": []}]}), \
         patch("mlb_edge_finder.pitcher_ingestion.config.DATA_RAW_DIR", tmp_path):
        with pytest.raises(RuntimeError, match="no pitcher stats"):
            pitcher_ingestion.fetch_pitcher_stats(date(2025, 9, 28), force=True)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/jaydengould/Documents/projects/mlb-edge-finder
pytest tests/test_pitcher_ingestion.py -v
```

Expected: `ModuleNotFoundError` or `ImportError` — module doesn't exist yet.

- [ ] **Step 3: Create the module with `fetch_pitcher_stats` + `load_cached_pitcher_stats`**

```python
# src/mlb_edge_finder/pitcher_ingestion.py
"""Fetch and cache individual pitcher season stats and probable starting pitchers."""
import logging
from datetime import date

import pandas as pd
import statsapi

from mlb_edge_finder import config

logger = logging.getLogger(__name__)

_FIP_CONSTANT: float = 3.15


def fetch_pitcher_stats(game_date: date, force: bool = False) -> pd.DataFrame:
    """Fetch season-to-date pitching stats for all pitchers via the MLB Stats API.

    Queries the stats endpoint with playerPool=All to get every pitcher's
    season stats in one call. Writes to DATA_RAW_DIR/pitcher_stats_YYYY-MM-DD.csv.
    Cache-first unless force=True.

    Args:
        game_date: Date whose season year to use for the stats fetch.
        force: If True, re-fetch even if a cache file exists.

    Returns:
        DataFrame with columns: pitcher_id, pitcher_name, era, whip,
        k_per_9, bb_per_9, ip, fip_computed. One row per pitcher with IP > 0.

    Raises:
        RuntimeError: If the statsapi call fails or returns no splits.
    """
    cache_path = config.DATA_RAW_DIR / f"pitcher_stats_{game_date}.csv"
    if cache_path.exists() and not force:
        logger.debug("Cache hit for pitcher_stats %s, loading from disk", game_date)
        return load_cached_pitcher_stats(game_date)

    season = game_date.year
    try:
        data = statsapi.get("stats", {
            "stats": "season",
            "group": "pitching",
            "sportId": 1,
            "season": season,
            "playerPool": "All",
            "limit": 5000,
        })
    except Exception as exc:
        raise RuntimeError(
            f"statsapi failed fetching pitcher stats for {season}: {exc}"
        ) from exc

    splits = data.get("stats", [{}])[0].get("splits", [])
    if not splits:
        raise RuntimeError(f"statsapi returned no pitcher stats for season {season}")

    rows = []
    for s in splits:
        player = s.get("player", {})
        st = s.get("stat", {})
        ip_str = st.get("inningsPitched", "0") or "0"
        ip = float(ip_str)
        if ip == 0:
            continue
        hr = int(st.get("homeRuns", 0) or 0)
        bb = int(st.get("baseOnBalls", 0) or 0)
        k_out = int(st.get("strikeOuts", 0) or 0)
        fip = (13 * hr + 3 * bb - 2 * k_out) / ip + _FIP_CONSTANT
        rows.append({
            "pitcher_id": player.get("id"),
            "pitcher_name": player.get("fullName"),
            "era": float(st.get("era", 0) or 0),
            "whip": float(st.get("whip", 0) or 0),
            "k_per_9": float(st.get("strikeoutsPer9Inn", 0) or 0),
            "bb_per_9": float(st.get("walksPer9Inn", 0) or 0),
            "ip": ip,
            "fip_computed": fip,
        })

    df = pd.DataFrame(rows)
    config.DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path, index=False)
    logger.info("Wrote %d pitchers to %s", len(df), cache_path)
    return df


def load_cached_pitcher_stats(game_date: date) -> pd.DataFrame:
    """Load previously fetched pitcher stats from DATA_RAW_DIR/pitcher_stats_YYYY-MM-DD.csv.

    Args:
        game_date: The date whose cached CSV to load.

    Returns:
        DataFrame with the same schema as fetch_pitcher_stats().

    Raises:
        FileNotFoundError: If no cached file exists for the given date.
    """
    cache_path = config.DATA_RAW_DIR / f"pitcher_stats_{game_date}.csv"
    if not cache_path.exists():
        raise FileNotFoundError(
            f"No cached pitcher stats for {game_date}: {cache_path}"
        )
    return pd.read_csv(cache_path)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_pitcher_ingestion.py -v
```

Expected: all 8 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/test_pitcher_ingestion.py src/mlb_edge_finder/pitcher_ingestion.py
git commit -m "feat: add pitcher_ingestion — fetch_pitcher_stats + load_cached_pitcher_stats"
```

---

## Task 2: `pitcher_ingestion.py` — `fetch_probable_starters`

**Files:**
- Modify: `tests/test_pitcher_ingestion.py`
- Modify: `src/mlb_edge_finder/pitcher_ingestion.py`

- [ ] **Step 1: Add failing tests**

Append to `tests/test_pitcher_ingestion.py`:

```python
def _make_schedule(home_name="New York Yankees", away_name="Boston Red Sox",
                   home_pitcher="Gerrit Cole", away_pitcher="Brayan Bello",
                   game_type="R"):
    return [{
        "game_id": 12345,
        "game_type": game_type,
        "home_name": home_name,
        "away_name": away_name,
        "home_probable_pitcher": home_pitcher,
        "away_probable_pitcher": away_pitcher,
    }]


def test_fetch_probable_starters_signature():
    from mlb_edge_finder import pitcher_ingestion
    assert callable(pitcher_ingestion.fetch_probable_starters)
    sig = inspect.signature(pitcher_ingestion.fetch_probable_starters)
    assert "game_date" in sig.parameters


def test_fetch_probable_starters_returns_expected_columns():
    from mlb_edge_finder import pitcher_ingestion
    with patch("mlb_edge_finder.pitcher_ingestion.statsapi.schedule",
               return_value=_make_schedule()):
        df = pitcher_ingestion.fetch_probable_starters(date(2025, 4, 22))
    assert set(df.columns) == {"home_abbr", "away_abbr", "home_starter_name", "away_starter_name"}


def test_fetch_probable_starters_maps_team_names():
    from mlb_edge_finder import pitcher_ingestion
    with patch("mlb_edge_finder.pitcher_ingestion.statsapi.schedule",
               return_value=_make_schedule()):
        df = pitcher_ingestion.fetch_probable_starters(date(2025, 4, 22))
    assert len(df) == 1
    assert df.iloc[0]["home_abbr"] == "NYY"
    assert df.iloc[0]["away_abbr"] == "BOS"
    assert df.iloc[0]["home_starter_name"] == "Gerrit Cole"
    assert df.iloc[0]["away_starter_name"] == "Brayan Bello"


def test_fetch_probable_starters_none_when_empty_string():
    from mlb_edge_finder import pitcher_ingestion
    with patch("mlb_edge_finder.pitcher_ingestion.statsapi.schedule",
               return_value=_make_schedule(home_pitcher="", away_pitcher="")):
        df = pitcher_ingestion.fetch_probable_starters(date(2025, 4, 22))
    assert pd.isna(df.iloc[0]["home_starter_name"])
    assert pd.isna(df.iloc[0]["away_starter_name"])


def test_fetch_probable_starters_skips_non_regular_season():
    from mlb_edge_finder import pitcher_ingestion
    with patch("mlb_edge_finder.pitcher_ingestion.statsapi.schedule",
               return_value=_make_schedule(game_type="S")):
        df = pitcher_ingestion.fetch_probable_starters(date(2025, 4, 22))
    assert len(df) == 0


def test_fetch_probable_starters_returns_empty_on_no_games():
    from mlb_edge_finder import pitcher_ingestion
    with patch("mlb_edge_finder.pitcher_ingestion.statsapi.schedule", return_value=[]):
        df = pitcher_ingestion.fetch_probable_starters(date(2025, 4, 22))
    assert len(df) == 0
    assert list(df.columns) == ["home_abbr", "away_abbr", "home_starter_name", "away_starter_name"]


def test_fetch_probable_starters_raises_on_api_failure():
    from mlb_edge_finder import pitcher_ingestion
    with patch("mlb_edge_finder.pitcher_ingestion.statsapi.schedule",
               side_effect=Exception("timeout")):
        with pytest.raises(RuntimeError, match="statsapi.schedule failed"):
            pitcher_ingestion.fetch_probable_starters(date(2025, 4, 22))
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_pitcher_ingestion.py::test_fetch_probable_starters_signature -v
```

Expected: `AttributeError` — `fetch_probable_starters` not defined yet.

- [ ] **Step 3: Add `fetch_probable_starters` to `pitcher_ingestion.py`**

Add to the top of the module (after existing imports):

```python
from mlb_edge_finder.rolling_stats import HISTORICAL_NAME_TO_ABBR
```

Append to `pitcher_ingestion.py`:

```python
def fetch_probable_starters(game_date: date) -> pd.DataFrame:
    """Fetch today's probable starting pitchers for all regular season games.

    Calls statsapi.schedule for the given date and extracts home/away
    probable pitcher names. Maps team names to abbreviations via
    HISTORICAL_NAME_TO_ABBR. Not cached — starters can change day-of.

    Args:
        game_date: Date to fetch probable starters for.

    Returns:
        DataFrame with columns: home_abbr, away_abbr,
        home_starter_name, away_starter_name. One row per game.
        Empty string probable pitchers are returned as NaN.
        Returns empty DataFrame (with correct columns) if no games found.

    Raises:
        RuntimeError: If the statsapi.schedule call fails.
    """
    try:
        games = statsapi.schedule(
            start_date=str(game_date),
            end_date=str(game_date),
            sportId=1,
        )
    except Exception as exc:
        raise RuntimeError(
            f"statsapi.schedule failed for {game_date}: {exc}"
        ) from exc

    rows = []
    for g in games:
        if g.get("game_type") != "R":
            continue
        home_abbr = HISTORICAL_NAME_TO_ABBR.get(g.get("home_name", ""))
        away_abbr = HISTORICAL_NAME_TO_ABBR.get(g.get("away_name", ""))
        if home_abbr is None or away_abbr is None:
            logger.warning(
                "fetch_probable_starters: unmapped team in game %s", g.get("game_id")
            )
            continue
        home_starter = g.get("home_probable_pitcher") or None
        away_starter = g.get("away_probable_pitcher") or None
        rows.append({
            "home_abbr": home_abbr,
            "away_abbr": away_abbr,
            "home_starter_name": home_starter,
            "away_starter_name": away_starter,
        })

    return pd.DataFrame(
        rows,
        columns=["home_abbr", "away_abbr", "home_starter_name", "away_starter_name"],
    )
```

- [ ] **Step 4: Run all pitcher ingestion tests**

```bash
pytest tests/test_pitcher_ingestion.py -v
```

Expected: all 15 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/test_pitcher_ingestion.py src/mlb_edge_finder/pitcher_ingestion.py
git commit -m "feat: add fetch_probable_starters to pitcher_ingestion"
```

---

## Task 3: Enrich `historical_ingestion.py` with starter names

**Files:**
- Modify: `src/mlb_edge_finder/historical_ingestion.py`
- Modify: `tests/test_historical_ingestion.py`

- [ ] **Step 1: Update the failing test first**

In `tests/test_historical_ingestion.py`, update `test_fetch_historical_filters_and_derives_home_win` and `test_fetch_all_historical_concatenates` to expect the new columns:

```python
def test_fetch_historical_filters_and_derives_home_win(tmp_path):
    games = [
        _make_game("Yankees", "Red Sox", 5, 3),
        _make_game("Cubs", "Cardinals", 1, 4),
        _make_game("Dodgers", "Giants", 2, 2, status="Postponed"),
        _make_game("Mets", "Phillies", 3, 1, game_type="S"),
    ]
    with patch("mlb_edge_finder.historical_ingestion.statsapi.schedule", return_value=games), \
         patch("mlb_edge_finder.historical_ingestion.config.DATA_RAW_DIR", tmp_path):
        df = historical_ingestion_module().fetch_historical(2024, force=True)

    assert len(df) == 2
    expected_cols = {
        "game_date", "home_name", "away_name", "home_score", "away_score",
        "home_win", "home_starter_name", "away_starter_name",
    }
    assert expected_cols == set(df.columns)
    assert df.loc[0, "home_win"] == 1
    assert df.loc[1, "home_win"] == 0


def test_fetch_historical_starter_names_are_none_when_empty(tmp_path):
    games = [_make_game("Yankees", "Red Sox", 5, 3)]
    with patch("mlb_edge_finder.historical_ingestion.statsapi.schedule", return_value=games), \
         patch("mlb_edge_finder.historical_ingestion.config.DATA_RAW_DIR", tmp_path):
        df = historical_ingestion_module().fetch_historical(2024, force=True)
    # _make_game sets home_probable_pitcher="" — should become NaN
    assert pd.isna(df.iloc[0]["home_starter_name"])
    assert pd.isna(df.iloc[0]["away_starter_name"])


def test_fetch_all_historical_concatenates(tmp_path):
    from mlb_edge_finder import historical_ingestion
    one_game = [_make_game("Yankees", "Red Sox", 5, 3)]
    with patch("mlb_edge_finder.historical_ingestion.statsapi.schedule", return_value=one_game), \
         patch("mlb_edge_finder.historical_ingestion.config.DATA_RAW_DIR", tmp_path):
        df = historical_ingestion.fetch_all_historical(force=True)
    assert len(df) == 3
    assert "home_starter_name" in df.columns
    assert "away_starter_name" in df.columns
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_historical_ingestion.py -v
```

Expected: `test_fetch_historical_filters_and_derives_home_win` and `test_fetch_all_historical_concatenates` FAIL (wrong columns), `test_fetch_historical_starter_names_are_none_when_empty` FAIL (column absent).

- [ ] **Step 3: Update `historical_ingestion.py`**

Replace the `_KEEP_COLS` constant and the relevant part of `fetch_historical`:

```python
# Replace _KEEP_COLS (line 14):
_KEEP_COLS = [
    "game_date", "home_name", "away_name", "home_score", "away_score", "home_win",
    "home_probable_pitcher", "away_probable_pitcher",
]
```

In `fetch_historical`, replace the final block that currently reads:
```python
df["home_win"] = (df["home_score"] > df["away_score"]).astype(int)
df = df[_KEEP_COLS].reset_index(drop=True)
```

with:
```python
df["home_win"] = (df["home_score"] > df["away_score"]).astype(int)
df = df.reindex(columns=_KEEP_COLS)
df = df.rename(columns={
    "home_probable_pitcher": "home_starter_name",
    "away_probable_pitcher": "away_starter_name",
})
for col in ("home_starter_name", "away_starter_name"):
    df[col] = df[col].replace("", None)
df = df.reset_index(drop=True)
```

- [ ] **Step 4: Run all historical ingestion tests**

```bash
pytest tests/test_historical_ingestion.py -v
```

Expected: all tests PASS.

- [ ] **Step 5: Run full test suite to check for regressions**

```bash
pytest tests/ -v
```

Expected: all tests PASS. (Training data and rolling stats tests still pass since `_make_hist()` helpers don't include starter columns — the join will just produce NaN pitcher stats, which is correct.)

- [ ] **Step 6: Commit**

```bash
git add src/mlb_edge_finder/historical_ingestion.py tests/test_historical_ingestion.py
git commit -m "feat: enrich historical_ingestion with home_starter_name and away_starter_name"
```

---

## Task 4: Update `model.py` NON_FEATURE_COLS

**Files:**
- Modify: `src/mlb_edge_finder/model.py:18-22`
- Modify: `tests/test_model.py`

- [ ] **Step 1: Add a failing test**

Open `tests/test_model.py` and add:

```python
def test_non_feature_cols_excludes_pitcher_metadata():
    from mlb_edge_finder.model import NON_FEATURE_COLS
    for col in ("home_starter_name", "away_starter_name",
                "home_pitcher_id", "away_pitcher_id"):
        assert col in NON_FEATURE_COLS, f"{col} missing from NON_FEATURE_COLS"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_model.py::test_non_feature_cols_excludes_pitcher_metadata -v
```

Expected: FAIL — columns not in list yet.

- [ ] **Step 3: Update `NON_FEATURE_COLS` in `model.py`**

Replace lines 18–22 in `src/mlb_edge_finder/model.py`:

```python
NON_FEATURE_COLS = [
    "game_date", "home_name", "away_name",
    "home_score", "away_score", "home_abbr", "away_abbr",
    "season", TARGET_COL,
    "home_starter_name", "away_starter_name",
    "home_pitcher_id", "away_pitcher_id",
]
```

- [ ] **Step 4: Run model tests**

```bash
pytest tests/test_model.py -v
```

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/mlb_edge_finder/model.py tests/test_model.py
git commit -m "feat: add pitcher metadata columns to NON_FEATURE_COLS"
```

---

## Task 5: Update `training_data.py` with pitcher stats join

**Files:**
- Modify: `src/mlb_edge_finder/training_data.py`
- Modify: `tests/test_training_data.py`

- [ ] **Step 1: Add failing tests**

Append to `tests/test_training_data.py`:

```python
def _make_pitcher_stats():
    return pd.DataFrame([
        {
            "pitcher_id": 1, "pitcher_name": "Cole Pitcher",
            "era": 3.50, "whip": 1.10, "k_per_9": 10.0, "bb_per_9": 2.5,
            "ip": 150.0, "fip_computed": 3.20,
        },
        {
            "pitcher_id": 2, "pitcher_name": "Bello Pitcher",
            "era": 4.00, "whip": 1.25, "k_per_9": 8.5, "bb_per_9": 3.0,
            "ip": 120.0, "fip_computed": 3.80,
        },
    ])


def _make_hist_with_starters(home="New York Yankees", away="Boston Red Sox",
                              home_starter="Cole Pitcher", away_starter="Bello Pitcher"):
    return pd.DataFrame([{
        "game_date": "2024-04-01",
        "home_name": home,
        "away_name": away,
        "home_score": 5,
        "away_score": 3,
        "home_win": 1,
        "home_starter_name": home_starter,
        "away_starter_name": away_starter,
    }])


def test_build_training_set_includes_pitcher_sp_cols(tmp_path):
    from mlb_edge_finder import training_data
    with patch("mlb_edge_finder.training_data.load_cached_historical",
               return_value=_make_hist_with_starters()), \
         patch("mlb_edge_finder.training_data.fetch_stats", return_value=_make_stats()), \
         patch("mlb_edge_finder.training_data.fetch_pitcher_stats",
               return_value=_make_pitcher_stats()), \
         patch("mlb_edge_finder.training_data.config.DATA_PROCESSED_DIR", tmp_path):
        df = training_data.build_training_set([2024], force=True)
    for col in ("home_sp_era", "away_sp_era", "home_sp_fip_computed", "away_sp_fip_computed",
                "home_sp_k_per_9", "away_sp_k_per_9"):
        assert col in df.columns, f"Missing column: {col}"


def test_build_training_set_pitcher_join_values_correct(tmp_path):
    from mlb_edge_finder import training_data
    with patch("mlb_edge_finder.training_data.load_cached_historical",
               return_value=_make_hist_with_starters()), \
         patch("mlb_edge_finder.training_data.fetch_stats", return_value=_make_stats()), \
         patch("mlb_edge_finder.training_data.fetch_pitcher_stats",
               return_value=_make_pitcher_stats()), \
         patch("mlb_edge_finder.training_data.config.DATA_PROCESSED_DIR", tmp_path):
        df = training_data.build_training_set([2024], force=True)
    assert abs(df.iloc[0]["home_sp_era"] - 3.50) < 0.01
    assert abs(df.iloc[0]["away_sp_era"] - 4.00) < 0.01


def test_build_training_set_pitcher_nan_when_starter_absent(tmp_path):
    from mlb_edge_finder import training_data
    hist = _make_hist_with_starters(home_starter=None, away_starter=None)
    with patch("mlb_edge_finder.training_data.load_cached_historical", return_value=hist), \
         patch("mlb_edge_finder.training_data.fetch_stats", return_value=_make_stats()), \
         patch("mlb_edge_finder.training_data.fetch_pitcher_stats",
               return_value=_make_pitcher_stats()), \
         patch("mlb_edge_finder.training_data.config.DATA_PROCESSED_DIR", tmp_path):
        df = training_data.build_training_set([2024], force=True)
    assert pd.isna(df.iloc[0]["home_sp_era"])
    assert pd.isna(df.iloc[0]["away_sp_era"])


def test_build_training_set_keeps_starter_name_columns(tmp_path):
    from mlb_edge_finder import training_data
    with patch("mlb_edge_finder.training_data.load_cached_historical",
               return_value=_make_hist_with_starters()), \
         patch("mlb_edge_finder.training_data.fetch_stats", return_value=_make_stats()), \
         patch("mlb_edge_finder.training_data.fetch_pitcher_stats",
               return_value=_make_pitcher_stats()), \
         patch("mlb_edge_finder.training_data.config.DATA_PROCESSED_DIR", tmp_path):
        df = training_data.build_training_set([2024], force=True)
    assert "home_starter_name" in df.columns
    assert "away_starter_name" in df.columns
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_training_data.py::test_build_training_set_includes_pitcher_sp_cols -v
```

Expected: FAIL — `fetch_pitcher_stats` not imported in `training_data`.

- [ ] **Step 3: Update `training_data.py`**

Add import at the top of `training_data.py` (after existing imports):

```python
from mlb_edge_finder.pitcher_ingestion import fetch_pitcher_stats
```

In `_build_season()`, after the existing rolling stats merge block (after line ~82, the second `df = df.merge(away_rolling...)`), add:

```python
    # Join starting pitcher season stats with home_sp_*/away_sp_* prefix
    pitcher_stats = fetch_pitcher_stats(date(season, _SNAPSHOT_MONTH, _SNAPSHOT_DAY))
    sp_cols = [c for c in pitcher_stats.columns if c not in ("pitcher_name", "pitcher_id")]
    home_pitcher = pitcher_stats.rename(columns={
        "pitcher_name": "home_starter_name",
        "pitcher_id": "home_pitcher_id",
        **{c: f"home_sp_{c}" for c in sp_cols},
    })
    away_pitcher = pitcher_stats.rename(columns={
        "pitcher_name": "away_starter_name",
        "pitcher_id": "away_pitcher_id",
        **{c: f"away_sp_{c}" for c in sp_cols},
    })
    home_pitcher_cols = ["home_starter_name", "home_pitcher_id"] + [f"home_sp_{c}" for c in sp_cols]
    away_pitcher_cols = ["away_starter_name", "away_pitcher_id"] + [f"away_sp_{c}" for c in sp_cols]
    df = df.merge(home_pitcher[home_pitcher_cols], on="home_starter_name", how="left")
    df = df.merge(away_pitcher[away_pitcher_cols], on="away_starter_name", how="left")
    logger.debug(
        "Season %d: pitcher join — %d/%d home starters matched, %d/%d away starters matched",
        season,
        df["home_pitcher_id"].notna().sum(), len(df),
        df["away_pitcher_id"].notna().sum(), len(df),
    )
```

**Important — existing tests need `fetch_pitcher_stats` mocked:** Every existing test that calls `build_training_set` without a cache hit will now fail because `_build_season()` calls `fetch_pitcher_stats`. Add `patch("mlb_edge_finder.training_data.fetch_pitcher_stats", return_value=_make_pitcher_stats())` to these eight existing tests:

- `test_build_training_set_joins_home_and_away_stats`
- `test_build_training_set_includes_season_column`
- `test_build_training_set_preserves_home_win`
- `test_build_training_set_keeps_name_and_abbr_columns`
- `test_build_training_set_raises_runtime_error_when_historical_missing`
- `test_build_training_set_drops_unmapped_teams`
- `test_build_training_set_applies_legacy_abbr_normalization`
- `test_build_training_set_includes_rolling_cols`
- `test_build_training_set_multi_season_concatenates`

`test_build_training_set_cache_first` uses `force=False` with a pre-populated cache, so `_build_season()` is never called — no change needed there.

Example updated signature (apply this pattern to all eight above):
```python
def test_build_training_set_joins_home_and_away_stats(tmp_path):
    from mlb_edge_finder import training_data
    with patch("mlb_edge_finder.training_data.load_cached_historical", return_value=_make_hist()), \
         patch("mlb_edge_finder.training_data.fetch_stats", return_value=_make_stats()), \
         patch("mlb_edge_finder.training_data.fetch_pitcher_stats",
               return_value=_make_pitcher_stats()), \
         patch("mlb_edge_finder.training_data.config.DATA_PROCESSED_DIR", tmp_path):
        df = training_data.build_training_set([2024])
    # ... assertions unchanged ...
```

The existing `_make_hist()` helper doesn't have `home_starter_name`/`away_starter_name` columns — the left-join produces NaN pitcher columns, which is correct. Existing assertions remain unchanged.

- [ ] **Step 4: Run all training data tests**

```bash
pytest tests/test_training_data.py -v
```

Expected: all tests PASS (new tests + existing tests).

- [ ] **Step 5: Commit**

```bash
git add src/mlb_edge_finder/training_data.py tests/test_training_data.py
git commit -m "feat: add pitcher stats double-join to training_data._build_season()"
```

---

## Task 6: Update `features.py` with probable starters + pitcher stats join

**Files:**
- Modify: `src/mlb_edge_finder/features.py`
- Modify: `tests/test_features.py`

- [ ] **Step 1: Add failing tests**

Append to `tests/test_features.py`:

```python
def _make_probable_starters():
    return pd.DataFrame([{
        "home_abbr": "NYY",
        "away_abbr": "BOS",
        "home_starter_name": "Gerrit Cole",
        "away_starter_name": "Brayan Bello",
    }])


def _make_pitcher_stats():
    return pd.DataFrame([
        {
            "pitcher_id": 1, "pitcher_name": "Gerrit Cole",
            "era": 3.20, "whip": 1.05, "k_per_9": 10.5, "bb_per_9": 2.2,
            "ip": 180.0, "fip_computed": 3.00,
        },
        {
            "pitcher_id": 2, "pitcher_name": "Brayan Bello",
            "era": 4.10, "whip": 1.30, "k_per_9": 8.0, "bb_per_9": 3.1,
            "ip": 140.0, "fip_computed": 3.90,
        },
    ])


def test_build_features_includes_pitcher_sp_cols():
    from mlb_edge_finder import features
    with patch("mlb_edge_finder.features.load_cached_odds", return_value=_make_odds()), \
         patch("mlb_edge_finder.features.load_cached_stats", return_value=_make_stats()), \
         patch("mlb_edge_finder.features.fetch_historical", return_value=_make_hist()), \
         patch("mlb_edge_finder.features.fetch_probable_starters",
               return_value=_make_probable_starters()), \
         patch("mlb_edge_finder.features.load_cached_pitcher_stats",
               return_value=_make_pitcher_stats()), \
         patch("mlb_edge_finder.features.config.DATA_PROCESSED_DIR") as mock_dir:
        mock_dir.__truediv__ = lambda self, other: __import__("pathlib").Path("/tmp") / other
        df = features.build_features(date(2025, 4, 22))
    for col in ("home_sp_era", "away_sp_era", "home_sp_fip_computed", "away_sp_fip_computed"):
        assert col in df.columns, f"Missing column: {col}"


def test_build_features_pitcher_join_values_correct():
    from mlb_edge_finder import features
    with patch("mlb_edge_finder.features.load_cached_odds", return_value=_make_odds()), \
         patch("mlb_edge_finder.features.load_cached_stats", return_value=_make_stats()), \
         patch("mlb_edge_finder.features.fetch_historical", return_value=_make_hist()), \
         patch("mlb_edge_finder.features.fetch_probable_starters",
               return_value=_make_probable_starters()), \
         patch("mlb_edge_finder.features.load_cached_pitcher_stats",
               return_value=_make_pitcher_stats()), \
         patch("mlb_edge_finder.features.config.DATA_PROCESSED_DIR") as mock_dir:
        mock_dir.__truediv__ = lambda self, other: __import__("pathlib").Path("/tmp") / other
        df = features.build_features(date(2025, 4, 22))
    assert abs(df.iloc[0]["home_sp_era"] - 3.20) < 0.01
    assert abs(df.iloc[0]["away_sp_era"] - 4.10) < 0.01


def test_build_features_pitcher_nan_when_no_probable_starter():
    from mlb_edge_finder import features
    no_starters = pd.DataFrame([{
        "home_abbr": "NYY", "away_abbr": "BOS",
        "home_starter_name": None, "away_starter_name": None,
    }])
    with patch("mlb_edge_finder.features.load_cached_odds", return_value=_make_odds()), \
         patch("mlb_edge_finder.features.load_cached_stats", return_value=_make_stats()), \
         patch("mlb_edge_finder.features.fetch_historical", return_value=_make_hist()), \
         patch("mlb_edge_finder.features.fetch_probable_starters", return_value=no_starters), \
         patch("mlb_edge_finder.features.load_cached_pitcher_stats",
               return_value=_make_pitcher_stats()), \
         patch("mlb_edge_finder.features.config.DATA_PROCESSED_DIR") as mock_dir:
        mock_dir.__truediv__ = lambda self, other: __import__("pathlib").Path("/tmp") / other
        df = features.build_features(date(2025, 4, 22))
    assert pd.isna(df.iloc[0]["home_sp_era"])
    assert pd.isna(df.iloc[0]["away_sp_era"])


def test_build_features_raises_on_missing_pitcher_stats():
    from mlb_edge_finder import features
    with patch("mlb_edge_finder.features.load_cached_odds", return_value=_make_odds()), \
         patch("mlb_edge_finder.features.load_cached_stats", return_value=_make_stats()), \
         patch("mlb_edge_finder.features.fetch_historical", return_value=_make_hist()), \
         patch("mlb_edge_finder.features.fetch_probable_starters",
               return_value=_make_probable_starters()), \
         patch("mlb_edge_finder.features.load_cached_pitcher_stats",
               side_effect=FileNotFoundError("no file")), \
         patch("mlb_edge_finder.features.config.DATA_PROCESSED_DIR") as mock_dir:
        mock_dir.__truediv__ = lambda self, other: __import__("pathlib").Path("/tmp") / other
        with pytest.raises(RuntimeError, match="fetch_pitcher_stats"):
            features.build_features(date(2025, 4, 22))
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_features.py::test_build_features_includes_pitcher_sp_cols -v
```

Expected: FAIL — `fetch_probable_starters` not imported.

- [ ] **Step 3: Update `features.py`**

Add imports at the top (after existing imports):

```python
from mlb_edge_finder.pitcher_ingestion import fetch_probable_starters, load_cached_pitcher_stats
```

In `build_features()`, after the existing rolling stats merge block (after the second `df = df.merge(away_rolling...)`), add:

```python
    # Join probable starting pitcher names onto the game rows
    probable_df = fetch_probable_starters(game_date)
    df = df.merge(probable_df, on=["home_abbr", "away_abbr"], how="left")

    # Load pitcher stats and double-join with home_sp_*/away_sp_* prefix
    try:
        pitcher_stats = load_cached_pitcher_stats(game_date)
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"No cached pitcher stats for {game_date} — run fetch_pitcher_stats() first"
        ) from exc

    sp_cols = [c for c in pitcher_stats.columns if c not in ("pitcher_name", "pitcher_id")]
    home_pitcher = pitcher_stats.rename(columns={
        "pitcher_name": "home_starter_name",
        "pitcher_id": "home_pitcher_id",
        **{c: f"home_sp_{c}" for c in sp_cols},
    })
    away_pitcher = pitcher_stats.rename(columns={
        "pitcher_name": "away_starter_name",
        "pitcher_id": "away_pitcher_id",
        **{c: f"away_sp_{c}" for c in sp_cols},
    })
    home_pitcher_cols = ["home_starter_name", "home_pitcher_id"] + [f"home_sp_{c}" for c in sp_cols]
    away_pitcher_cols = ["away_starter_name", "away_pitcher_id"] + [f"away_sp_{c}" for c in sp_cols]
    df = df.merge(home_pitcher[home_pitcher_cols], on="home_starter_name", how="left")
    df = df.merge(away_pitcher[away_pitcher_cols], on="away_starter_name", how="left")
    logger.debug(
        "Built features: pitcher join — %d/%d home starters matched",
        df["home_pitcher_id"].notna().sum(), len(df),
    )
```

**Note:** The existing tests in `test_features.py` call `features.build_features` without mocking `fetch_probable_starters` or `load_cached_pitcher_stats`. Update those tests to add the new mocks. Specifically, update `test_build_features_joins_home_and_away`, `test_build_features_includes_rolling_cols`, and `test_build_features_raises_on_missing_odds` to patch the new dependencies:

```python
def test_build_features_joins_home_and_away():
    from mlb_edge_finder import features
    with patch("mlb_edge_finder.features.load_cached_odds", return_value=_make_odds()), \
         patch("mlb_edge_finder.features.load_cached_stats", return_value=_make_stats()), \
         patch("mlb_edge_finder.features.fetch_historical", return_value=_make_hist()), \
         patch("mlb_edge_finder.features.fetch_probable_starters",
               return_value=_make_probable_starters()), \
         patch("mlb_edge_finder.features.load_cached_pitcher_stats",
               return_value=_make_pitcher_stats()), \
         patch("mlb_edge_finder.features.config.DATA_PROCESSED_DIR") as mock_dir:
        mock_dir.__truediv__ = lambda self, other: __import__("pathlib").Path("/tmp") / other
        df = features.build_features(date(2025, 4, 22))
    assert len(df) == 1
    assert "home_bat_avg" in df.columns
    assert "away_bat_avg" in df.columns
    assert "home_era" in df.columns
    assert "away_era" in df.columns
    assert "data_source" not in df.columns


def test_build_features_includes_rolling_cols():
    from mlb_edge_finder import features
    with patch("mlb_edge_finder.features.load_cached_odds", return_value=_make_odds()), \
         patch("mlb_edge_finder.features.load_cached_stats", return_value=_make_stats()), \
         patch("mlb_edge_finder.features.fetch_historical", return_value=_make_hist()), \
         patch("mlb_edge_finder.features.fetch_probable_starters",
               return_value=_make_probable_starters()), \
         patch("mlb_edge_finder.features.load_cached_pitcher_stats",
               return_value=_make_pitcher_stats()), \
         patch("mlb_edge_finder.features.config.DATA_PROCESSED_DIR") as mock_dir:
        mock_dir.__truediv__ = lambda self, other: __import__("pathlib").Path("/tmp") / other
        df = features.build_features(date(2025, 4, 22))
    for col in ("home_rolling_runs_scored", "away_rolling_runs_scored",
                "home_rolling_run_diff", "away_rolling_run_diff"):
        assert col in df.columns, f"Missing column: {col}"
    assert not df["home_rolling_runs_scored"].isna().any()
```

- [ ] **Step 4: Run all features tests**

```bash
pytest tests/test_features.py -v
```

Expected: all tests PASS.

- [ ] **Step 5: Run full suite**

```bash
pytest tests/ -v
```

Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
git add src/mlb_edge_finder/features.py tests/test_features.py
git commit -m "feat: add probable starters and pitcher stats join to features.build_features()"
```

---

## Task 7: Update `pipeline.py` with `fetch_pitcher_stats` pre-step

**Files:**
- Modify: `src/mlb_edge_finder/pipeline.py`
- Modify: `tests/test_pipeline.py`

- [ ] **Step 1: Add a failing test**

Open `tests/test_pipeline.py` and add:

```python
def test_pipeline_calls_fetch_pitcher_stats_before_build_features():
    from mlb_edge_finder import pipeline
    from datetime import date
    import pandas as pd
    from unittest.mock import patch, MagicMock, call

    mock_clf = MagicMock()
    mock_clf.feature_names_in_ = []
    mock_edges = pd.DataFrame()

    with patch("mlb_edge_finder.pipeline.odds_ingestion.fetch_odds"), \
         patch("mlb_edge_finder.pipeline.stats_ingestion.fetch_stats"), \
         patch("mlb_edge_finder.pipeline.pitcher_ingestion.fetch_pitcher_stats") as mock_pitcher, \
         patch("mlb_edge_finder.pipeline.features.build_features", return_value=pd.DataFrame()), \
         patch("mlb_edge_finder.pipeline.model.load_model", return_value=mock_clf), \
         patch("mlb_edge_finder.pipeline.edge_finder.find_edges", return_value=mock_edges), \
         patch("mlb_edge_finder.pipeline.config.MODELS_DIR") as mock_models_dir:
        mock_models_dir.glob.return_value = [
            __import__("pathlib").Path("models/xgb_2025-01-01.pkl")
        ]
        pipeline.run(date(2025, 4, 22))

    mock_pitcher.assert_called_once_with(date(2025, 4, 22))
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_pipeline.py::test_pipeline_calls_fetch_pitcher_stats_before_build_features -v
```

Expected: FAIL — `pitcher_ingestion` not imported in pipeline.

- [ ] **Step 3: Update `pipeline.py`**

Replace the import line:
```python
from mlb_edge_finder import config, edge_finder, features, model, odds_ingestion, stats_ingestion
```
with:
```python
from mlb_edge_finder import (
    config, edge_finder, features, model,
    odds_ingestion, pitcher_ingestion, stats_ingestion,
)
```

Add the `fetch_pitcher_stats` call after `fetch_stats` in `run()`:

```python
    odds_ingestion.fetch_odds(game_date)
    stats_ingestion.fetch_stats(game_date)
    pitcher_ingestion.fetch_pitcher_stats(game_date)
    features_df = features.build_features(game_date)
```

- [ ] **Step 4: Run all pipeline tests**

```bash
pytest tests/test_pipeline.py -v
```

Expected: all tests PASS.

- [ ] **Step 5: Run full test suite**

```bash
pytest tests/ -v
```

Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
git add src/mlb_edge_finder/pipeline.py tests/test_pipeline.py
git commit -m "feat: add fetch_pitcher_stats pre-step to pipeline.run()"
```

---

## Task 8: Update notebook + CLAUDE.md

**Files:**
- Modify: `notebooks/01_exploration.ipynb`
- Modify: `CLAUDE.md`

- [ ] **Step 1: Update `CLAUDE.md`**

In the **Current Phase** section, update to reflect Phase 7 complete. Add to the bullet list:

```
- **7 complete:** `pitcher_ingestion.py` — `fetch_pitcher_stats(game_date)`, `load_cached_pitcher_stats(game_date)`, `fetch_probable_starters(game_date)`. statsapi-only (no FanGraphs). Cache at `data/raw/pitcher_stats_YYYY-MM-DD.csv`. `historical_ingestion.fetch_historical()` enriched with `home_starter_name`/`away_starter_name` columns. `training_data._build_season()` and `features.build_features()` both join pitcher stats with `home_sp_*`/`away_sp_*` prefix. `model.NON_FEATURE_COLS` updated. `pipeline.run()` calls `fetch_pitcher_stats` before `build_features`.
```

Update the **Roadmap** section to check off Phase 7 and update the **Module Responsibilities** table to include `pitcher_ingestion.py`.

- [ ] **Step 2: Update `notebooks/01_exploration.ipynb`**

Add a **Section 7** cell after the existing Phase 6 section. The cell should demonstrate the new pitcher ingestion functions:

```python
# Section 7 — Starting Pitcher Features
# Requires: historical data cached (Section 4a), stats cached (Section 4b)

from mlb_edge_finder.pitcher_ingestion import (
    fetch_pitcher_stats,
    load_cached_pitcher_stats,
    fetch_probable_starters,
)
from datetime import date

# Fetch season pitcher stats (cached after first run)
game_date = date(2025, 9, 28)
pitcher_df = fetch_pitcher_stats(game_date)
print(f"Fetched {len(pitcher_df)} pitchers")
pitcher_df.head(10)
```

Add a second cell:
```python
# Fetch today's probable starters (live call, not cached)
today = date.today()
starters_df = fetch_probable_starters(today)
print(f"Probable starters for {today}: {len(starters_df)} games")
starters_df
```

Update the **Section 4b** and any cells that call `build_training_set` to add `force=True` since the training cache will need to be rebuilt to include pitcher columns:

```python
# Rebuild training set to include pitcher stats (force=True required after Phase 7)
training_df = build_training_set([2023, 2024, 2025], force=True)
print(training_df.shape)
training_df[["home_starter_name", "home_sp_era", "away_sp_era"]].head()
```

- [ ] **Step 3: Run full test suite one last time**

```bash
pytest tests/ -v
```

Expected: all tests PASS. Record final test count.

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md notebooks/01_exploration.ipynb
git commit -m "docs: mark Phase 7 complete, update notebook for pitcher features"
```
