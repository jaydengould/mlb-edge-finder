# Phase 6: Rolling Window Team Stats Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a new `rolling_stats.py` module that computes per-team rolling averages (runs scored, runs allowed, win %, run diff) from existing cached game results, and wire those rolling features into both the training pipeline and the inference pipeline.

**Architecture:** `rolling_stats.py` owns `HISTORICAL_NAME_TO_ABBR` (moved from `training_data.py`) and exposes two functions: `compute_rolling_stats()` (shift-1, for training) and `latest_rolling_stats()` (no shift, for inference). `training_data._build_season()` adds a rolling stats join after the existing stats join. `features.build_features()` fetches the current season's completed games and joins latest rolling stats by team abbreviation. No changes to `pipeline.run()`.

**Tech Stack:** pandas (rolling, groupby, transform), pytest, unittest.mock

---

## File Map

| File | Action | What changes |
|---|---|---|
| `src/mlb_edge_finder/rolling_stats.py` | Create | New module: `HISTORICAL_NAME_TO_ABBR`, `compute_rolling_stats()`, `latest_rolling_stats()` |
| `src/mlb_edge_finder/training_data.py` | Modify | Import `HISTORICAL_NAME_TO_ABBR` + `compute_rolling_stats` from rolling_stats; add rolling join in `_build_season()` |
| `src/mlb_edge_finder/features.py` | Modify | Import `fetch_historical` + `latest_rolling_stats`; add rolling join in `build_features()` |
| `tests/test_rolling_stats.py` | Create | 5 tests for the new module |
| `tests/test_training_data.py` | Modify | Add 1 test for rolling columns in training set output |
| `tests/test_features.py` | Modify | Update existing join test to mock `fetch_historical`; add 1 test for rolling columns in features output |
| `CLAUDE.md` | Modify | Mark Phase 6 complete, document new module |

---

## Task 1: `rolling_stats.py` — tests then implementation

**Files:**
- Create: `tests/test_rolling_stats.py`
- Create: `src/mlb_edge_finder/rolling_stats.py`

- [ ] **Step 1: Write the 5 failing tests**

Create `tests/test_rolling_stats.py`:

```python
"""Tests for rolling_stats module."""
import pandas as pd
import pytest


def _make_hist(games: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(games)


def test_compute_rolling_stats_shift():
    """First game of season has NaN rolling stats; game 3 reflects only games 1-2."""
    from mlb_edge_finder.rolling_stats import compute_rolling_stats

    hist = _make_hist([
        {"game_date": "2024-04-01", "home_name": "New York Yankees",
         "away_name": "Boston Red Sox", "home_score": 5, "away_score": 3, "home_win": 1},
        {"game_date": "2024-04-03", "home_name": "Boston Red Sox",
         "away_name": "New York Yankees", "home_score": 2, "away_score": 4, "home_win": 0},
        {"game_date": "2024-04-05", "home_name": "New York Yankees",
         "away_name": "Boston Red Sox", "home_score": 7, "away_score": 2, "home_win": 1},
    ])
    result = compute_rolling_stats(hist, window=15)
    nyy = result[result["team_abbr"] == "NYY"].sort_values("game_date").reset_index(drop=True)

    # First game has no prior games → NaN
    assert pd.isna(nyy.iloc[0]["rolling_runs_scored"])

    # Third game (Apr 5) reflects Apr 1 (scored 5) and Apr 3 (scored 4 as away) → avg 4.5
    assert abs(nyy.iloc[2]["rolling_runs_scored"] - 4.5) < 1e-6


def test_compute_rolling_stats_window():
    """Window of 15 limits rolling average to last 15 prior games."""
    from mlb_edge_finder.rolling_stats import compute_rolling_stats

    # 17 games for NYY (always home, away_score=0), runs scored = game number
    games = [
        {"game_date": f"2024-04-{i+1:02d}", "home_name": "New York Yankees",
         "away_name": "Boston Red Sox", "home_score": i + 1, "away_score": 0, "home_win": 1}
        for i in range(17)
    ]
    hist = _make_hist(games)
    result = compute_rolling_stats(hist, window=15)
    nyy = result[result["team_abbr"] == "NYY"].sort_values("game_date").reset_index(drop=True)

    # Game 17 (index 16): shift(1) means rolling = avg of games 1-16 capped at window=15
    # = avg of games 2-16 = (2+3+...+16)/15 = 9.0
    assert abs(nyy.iloc[16]["rolling_runs_scored"] - 9.0) < 1e-6


def test_compute_rolling_stats_min_periods():
    """min_periods=1 allows rolling with fewer than window games without error."""
    from mlb_edge_finder.rolling_stats import compute_rolling_stats

    hist = _make_hist([
        {"game_date": "2024-04-01", "home_name": "New York Yankees",
         "away_name": "Boston Red Sox", "home_score": 5, "away_score": 3, "home_win": 1},
        {"game_date": "2024-04-03", "home_name": "New York Yankees",
         "away_name": "Boston Red Sox", "home_score": 3, "away_score": 2, "home_win": 1},
        {"game_date": "2024-04-05", "home_name": "New York Yankees",
         "away_name": "Boston Red Sox", "home_score": 4, "away_score": 1, "home_win": 1},
    ])
    # window=15 but only 3 games — should not raise
    result = compute_rolling_stats(hist, window=15)
    nyy = result[result["team_abbr"] == "NYY"].sort_values("game_date").reset_index(drop=True)

    # Game 2 (index 1): shift(1) → rolling = avg of game 1 only = 5.0
    assert abs(nyy.iloc[1]["rolling_runs_scored"] - 5.0) < 1e-6


def test_latest_rolling_stats_one_row_per_team():
    """Returns exactly one row per team with no duplicate team_abbr."""
    from mlb_edge_finder.rolling_stats import latest_rolling_stats

    hist = _make_hist([
        {"game_date": "2024-04-01", "home_name": "New York Yankees",
         "away_name": "Boston Red Sox", "home_score": 5, "away_score": 3, "home_win": 1},
        {"game_date": "2024-04-03", "home_name": "New York Yankees",
         "away_name": "Boston Red Sox", "home_score": 3, "away_score": 2, "home_win": 1},
    ])
    result = latest_rolling_stats(hist, window=15)

    assert len(result) == 2
    assert result["team_abbr"].nunique() == 2
    assert set(result["team_abbr"]) == {"NYY", "BOS"}


def test_latest_rolling_stats_includes_last_game():
    """Latest stats include the most recent completed game (no shift applied)."""
    from mlb_edge_finder.rolling_stats import latest_rolling_stats

    hist = _make_hist([
        {"game_date": "2024-04-01", "home_name": "New York Yankees",
         "away_name": "Boston Red Sox", "home_score": 2, "away_score": 1, "home_win": 1},
        {"game_date": "2024-04-03", "home_name": "New York Yankees",
         "away_name": "Boston Red Sox", "home_score": 8, "away_score": 1, "home_win": 1},
    ])
    result = latest_rolling_stats(hist, window=15)
    nyy = result[result["team_abbr"] == "NYY"].iloc[0]

    # No shift: avg of both games = (2 + 8) / 2 = 5.0
    assert abs(nyy["rolling_runs_scored"] - 5.0) < 1e-6
```

- [ ] **Step 2: Run the tests to verify they all fail**

```bash
pytest tests/test_rolling_stats.py -v
```

Expected: 5 FAIL with `ModuleNotFoundError: No module named 'mlb_edge_finder.rolling_stats'`

- [ ] **Step 3: Create `rolling_stats.py`**

Create `src/mlb_edge_finder/rolling_stats.py`:

```python
"""Compute rolling per-team stats from historical game results."""
import logging

import pandas as pd

logger = logging.getLogger(__name__)

# statsapi full team names → current franchise abbreviations.
# Moved here from training_data so both training and inference paths
# share the same mapping without circular imports.
HISTORICAL_NAME_TO_ABBR: dict[str, str] = {
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
    "Oakland Athletics": "ATH",
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
    # Legacy names for pre-rename seasons
    "Cleveland Indians": "CLE",
    "Florida Marlins": "MIA",
    "Tampa Bay Devil Rays": "TB",
    "Montreal Expos": "WSH",
}

_ROLLING_COLS = [
    "rolling_runs_scored",
    "rolling_runs_allowed",
    "rolling_win_pct",
    "rolling_run_diff",
]


def _reshape_to_team_games(historical_df: pd.DataFrame) -> pd.DataFrame:
    """Reshape from one-row-per-game to one-row-per-team-game.

    Each game becomes two rows — one for the home team and one for the away team.
    Maps full statsapi team names to abbreviations via HISTORICAL_NAME_TO_ABBR.
    Drops rows with unmapped team names and logs a warning.
    """
    home = pd.DataFrame({
        "team_abbr": historical_df["home_name"].map(HISTORICAL_NAME_TO_ABBR),
        "game_date": historical_df["game_date"].values,
        "runs_scored": historical_df["home_score"].astype(float).values,
        "runs_allowed": historical_df["away_score"].astype(float).values,
        "win": historical_df["home_win"].astype(float).values,
    })
    away = pd.DataFrame({
        "team_abbr": historical_df["away_name"].map(HISTORICAL_NAME_TO_ABBR),
        "game_date": historical_df["game_date"].values,
        "runs_scored": historical_df["away_score"].astype(float).values,
        "runs_allowed": historical_df["home_score"].astype(float).values,
        "win": (1 - historical_df["home_win"]).astype(float).values,
    })
    long_df = pd.concat([home, away], ignore_index=True)
    n_unmapped = long_df["team_abbr"].isna().sum()
    if n_unmapped:
        logger.warning("Rolling stats: dropped %d rows with unmapped team names", n_unmapped)
    long_df = long_df.dropna(subset=["team_abbr"])
    return long_df.sort_values(["team_abbr", "game_date"]).reset_index(drop=True)


def _roll(long_df: pd.DataFrame, window: int, shift: bool) -> pd.DataFrame:
    """Apply rolling aggregation per team, optionally shifting by 1."""
    df = long_df.copy()
    df["run_diff"] = df["runs_scored"] - df["runs_allowed"]
    result = df[["team_abbr", "game_date"]].copy()
    for raw_col, out_col in [
        ("runs_scored", "rolling_runs_scored"),
        ("runs_allowed", "rolling_runs_allowed"),
        ("win", "rolling_win_pct"),
        ("run_diff", "rolling_run_diff"),
    ]:
        result[out_col] = (
            df.groupby("team_abbr")[raw_col]
            .transform(lambda x: x.rolling(window, min_periods=1).mean())
        )
    if shift:
        for col in _ROLLING_COLS:
            result[col] = result.groupby("team_abbr")[col].transform(lambda x: x.shift(1))
    return result


def compute_rolling_stats(historical_df: pd.DataFrame, window: int = 15) -> pd.DataFrame:
    """Compute per-game pregame rolling stats for training.

    Rolling stats for each game reflect up to `window` completed games BEFORE
    that game date. The current game is excluded via shift(1). First game of
    the season per team has NaN rolling stats — XGBoost handles NaN natively.

    Args:
        historical_df: Output of fetch_historical() or load_cached_historical().
            Columns: game_date, home_name, away_name, home_score, away_score, home_win.
        window: Number of prior games to average over. Default 15.

    Returns:
        DataFrame with columns: team_abbr, game_date, rolling_runs_scored,
        rolling_runs_allowed, rolling_win_pct, rolling_run_diff.
    """
    long_df = _reshape_to_team_games(historical_df)
    return _roll(long_df, window, shift=True)


def latest_rolling_stats(historical_df: pd.DataFrame, window: int = 15) -> pd.DataFrame:
    """Compute current rolling stats per team for inference.

    All completed games are included (no shift). Returns one row per team
    reflecting their current form going into today's games.

    Args:
        historical_df: Output of fetch_historical() for the current season.
            Columns: game_date, home_name, away_name, home_score, away_score, home_win.
        window: Number of prior games to average over. Default 15.

    Returns:
        DataFrame with one row per team_abbr. Columns: team_abbr,
        rolling_runs_scored, rolling_runs_allowed, rolling_win_pct, rolling_run_diff.
    """
    long_df = _reshape_to_team_games(historical_df)
    rolled = _roll(long_df, window, shift=False)
    return rolled.groupby("team_abbr").last().reset_index()
```

- [ ] **Step 4: Run the 5 tests to verify they pass**

```bash
pytest tests/test_rolling_stats.py -v
```

Expected: 5 PASS.

- [ ] **Step 5: Run full suite to check for regressions**

```bash
pytest tests/ -v
```

Expected: all 82 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add tests/test_rolling_stats.py src/mlb_edge_finder/rolling_stats.py
git commit -m "feat: add rolling_stats module with compute and latest functions"
```

---

## Task 2: Update `training_data.py`

**Files:**
- Modify: `tests/test_training_data.py`
- Modify: `src/mlb_edge_finder/training_data.py`

- [ ] **Step 1: Add the failing test for rolling columns**

Append to `tests/test_training_data.py`:

```python
def test_build_training_set_includes_rolling_cols(tmp_path):
    """build_training_set output includes home_ and away_ rolling stat columns."""
    from mlb_edge_finder import training_data
    with patch("mlb_edge_finder.training_data.load_cached_historical", return_value=_make_hist()), \
         patch("mlb_edge_finder.training_data.fetch_stats", return_value=_make_stats()), \
         patch("mlb_edge_finder.training_data.config.DATA_PROCESSED_DIR", tmp_path):
        df = training_data.build_training_set([2024], force=True)
    for col in ("home_rolling_runs_scored", "away_rolling_runs_scored",
                "home_rolling_run_diff", "away_rolling_run_diff"):
        assert col in df.columns, f"Missing column: {col}"
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
pytest tests/test_training_data.py::test_build_training_set_includes_rolling_cols -v
```

Expected: FAIL — `AssertionError: Missing column: home_rolling_runs_scored`

- [ ] **Step 3: Update `training_data.py`**

Replace the entire contents of `src/mlb_edge_finder/training_data.py` with:

```python
"""Build and cache a labeled training dataset for XGBoost model training."""
import logging
from datetime import date

import pandas as pd

from mlb_edge_finder import config
from mlb_edge_finder.historical_ingestion import load_cached_historical
from mlb_edge_finder.rolling_stats import HISTORICAL_NAME_TO_ABBR, compute_rolling_stats
from mlb_edge_finder.stats_ingestion import fetch_stats

logger = logging.getLogger(__name__)

# FanGraphs abbreviations that changed between seasons → current abbreviation.
# Applied to the stats DataFrame before joining so both join sides use current identifiers.
_LEGACY_ABBR_NORMALIZE: dict[str, str] = {
    "OAK": "ATH",   # Oakland → Sacramento Athletics
}

_SNAPSHOT_MONTH = 9
_SNAPSHOT_DAY = 28


def _build_season(season: int) -> pd.DataFrame:
    """Load historical games, end-of-season stats, and rolling stats for one season."""
    try:
        hist = load_cached_historical(season)
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"No cached historical data for {season} — run fetch_historical({season}) first"
        ) from exc

    stats = fetch_stats(date(season, _SNAPSHOT_MONTH, _SNAPSHOT_DAY))

    # Normalize FanGraphs abbreviations that changed between seasons
    stats = stats.copy()
    stats["team_abbr"] = stats["team_abbr"].replace(_LEGACY_ABBR_NORMALIZE)

    # Map full team names → current abbreviations
    hist = hist.copy()
    hist["home_abbr"] = hist["home_name"].map(HISTORICAL_NAME_TO_ABBR)
    hist["away_abbr"] = hist["away_name"].map(HISTORICAL_NAME_TO_ABBR)

    unmapped = pd.concat([
        hist.loc[hist["home_abbr"].isna(), "home_name"],
        hist.loc[hist["away_abbr"].isna(), "away_name"],
    ]).unique()
    if len(unmapped):
        logger.warning("Season %d: unmapped team names dropped: %s", season, list(unmapped))
    hist = hist.dropna(subset=["home_abbr", "away_abbr"])

    # Drop data_source — not a model feature
    stats = stats.drop(columns=["data_source"], errors="ignore")

    # Double-join end-of-season stats with home_/away_ prefixes
    stat_cols = [c for c in stats.columns if c != "team_abbr"]
    home_stats = stats.rename(columns={"team_abbr": "home_abbr"} | {c: f"home_{c}" for c in stat_cols})
    away_stats = stats.rename(columns={"team_abbr": "away_abbr"} | {c: f"away_{c}" for c in stat_cols})

    df = hist.merge(home_stats, on="home_abbr", how="inner")
    df = df.merge(away_stats, on="away_abbr", how="inner")
    df["season"] = season

    # Add rolling stats — computed from game results, no new API calls.
    # shift(1) ensures each game's rolling stats use only prior games.
    # First game of season per team has NaN rolling stats (XGBoost handles NaN natively).
    rolling = compute_rolling_stats(hist)
    rolling_cols = [c for c in rolling.columns if c not in ("team_abbr", "game_date")]
    home_rolling = rolling.rename(
        columns={"team_abbr": "home_abbr"} | {c: f"home_{c}" for c in rolling_cols}
    )
    away_rolling = rolling.rename(
        columns={"team_abbr": "away_abbr"} | {c: f"away_{c}" for c in rolling_cols}
    )
    df = df.merge(
        home_rolling[["home_abbr", "game_date"] + [f"home_{c}" for c in rolling_cols]],
        on=["home_abbr", "game_date"], how="left",
    )
    df = df.merge(
        away_rolling[["away_abbr", "game_date"] + [f"away_{c}" for c in rolling_cols]],
        on=["away_abbr", "game_date"], how="left",
    )

    logger.debug("Season %d: %d games, %d columns", season, len(df), len(df.columns))
    return df


def build_training_set(seasons: list[int], force: bool = False) -> pd.DataFrame:
    """Build and cache a labeled training set by joining historical games with stats.

    For each season, loads historical game results and fetches end-of-season stats
    (September 28 snapshot), normalizes abbreviations, joins stats twice with home_/away_
    prefixes, adds rolling stats, and tags rows with a season column. Concatenates all seasons.

    Args:
        seasons: List of season years to include (e.g. [2023, 2024, 2025]).
        force: If True, rebuild even if a cache file exists.

    Returns:
        DataFrame with one row per game. Columns: game_date, season, home_name, away_name,
        home_abbr, away_abbr, home_win, plus home_<stat>/away_<stat> for every stat column,
        and home_rolling_*/away_rolling_* for rolling stats.

    Raises:
        RuntimeError: If historical data is missing for any season.
    """
    out_path = config.DATA_PROCESSED_DIR / f"training_{min(seasons)}-{max(seasons)}.csv"
    if out_path.exists() and not force:
        logger.debug("Cache hit, loading from %s", out_path)
        return load_training_set(seasons)

    frames = [_build_season(s) for s in seasons]
    df = pd.concat(frames, ignore_index=True)

    config.DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    logger.info("Wrote %d rows (%d seasons) to %s", len(df), len(seasons), out_path)
    return df


def load_training_set(seasons: list[int]) -> pd.DataFrame:
    """Load a previously built training set from DATA_PROCESSED_DIR.

    Args:
        seasons: The seasons list whose training CSV to load (determines filename).

    Returns:
        DataFrame with the same schema as build_training_set().

    Raises:
        FileNotFoundError: If no training set file exists for the given seasons.
    """
    out_path = config.DATA_PROCESSED_DIR / f"training_{min(seasons)}-{max(seasons)}.csv"
    if not out_path.exists():
        raise FileNotFoundError(f"No cached training set for seasons {seasons}: {out_path}")
    return pd.read_csv(out_path)
```

- [ ] **Step 4: Run the training_data tests**

```bash
pytest tests/test_training_data.py -v
```

Expected: all 13 tests PASS (12 existing + 1 new).

- [ ] **Step 5: Run full suite**

```bash
pytest tests/ -v
```

Expected: all 87 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add tests/test_training_data.py src/mlb_edge_finder/training_data.py
git commit -m "feat: add rolling stats to training_data._build_season()"
```

---

## Task 3: Update `features.py`

**Files:**
- Modify: `tests/test_features.py`
- Modify: `src/mlb_edge_finder/features.py`

- [ ] **Step 1: Update the existing join test and add the rolling columns test**

Replace the entire contents of `tests/test_features.py` with:

```python
"""Smoke tests: features exposes expected public API."""
import inspect
from datetime import date
from unittest.mock import patch

import pandas as pd
import pytest


def _make_odds():
    return pd.DataFrame([{
        "game_id": "abc",
        "home_team": "New York Yankees",
        "away_team": "Boston Red Sox",
        "home_odds_american": -150,
        "away_odds_american": 130,
        "commence_time": "2025-04-22T18:05:00Z",
    }])


def _make_stats():
    return pd.DataFrame([
        {"team_abbr": "NYY", "bat_avg": ".260", "obp": ".330", "slg": ".420",
         "ops": ".750", "runs_per_game": 4.8, "era": "3.80", "whip": "1.20",
         "k_per_9": "9.0", "bb_per_9": "3.0", "data_source": "mlb_api"},
        {"team_abbr": "BOS", "bat_avg": ".255", "obp": ".320", "slg": ".410",
         "ops": ".730", "runs_per_game": 4.5, "era": "4.10", "whip": "1.30",
         "k_per_9": "8.5", "bb_per_9": "3.2", "data_source": "mlb_api"},
    ])


def _make_hist():
    return pd.DataFrame([{
        "game_date": "2025-04-20",
        "home_name": "New York Yankees",
        "away_name": "Boston Red Sox",
        "home_score": 5,
        "away_score": 3,
        "home_win": 1,
    }])


def test_build_features_signature():
    """build_features should accept game_date."""
    from mlb_edge_finder import features
    assert callable(features.build_features)
    sig = inspect.signature(features.build_features)
    assert "game_date" in sig.parameters


def test_load_features_signature():
    """load_features should accept game_date."""
    from mlb_edge_finder import features
    assert callable(features.load_features)
    sig = inspect.signature(features.load_features)
    assert "game_date" in sig.parameters


def test_build_features_joins_home_and_away():
    """build_features should produce home_ and away_ prefixed stat columns."""
    from mlb_edge_finder import features

    with patch("mlb_edge_finder.features.load_cached_odds", return_value=_make_odds()), \
         patch("mlb_edge_finder.features.load_cached_stats", return_value=_make_stats()), \
         patch("mlb_edge_finder.features.fetch_historical", return_value=_make_hist()), \
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
    """build_features output includes home_ and away_ rolling stat columns."""
    from mlb_edge_finder import features

    with patch("mlb_edge_finder.features.load_cached_odds", return_value=_make_odds()), \
         patch("mlb_edge_finder.features.load_cached_stats", return_value=_make_stats()), \
         patch("mlb_edge_finder.features.fetch_historical", return_value=_make_hist()), \
         patch("mlb_edge_finder.features.config.DATA_PROCESSED_DIR") as mock_dir:
        mock_dir.__truediv__ = lambda self, other: __import__("pathlib").Path("/tmp") / other
        df = features.build_features(date(2025, 4, 22))

    for col in ("home_rolling_runs_scored", "away_rolling_runs_scored",
                "home_rolling_run_diff", "away_rolling_run_diff"):
        assert col in df.columns, f"Missing column: {col}"

    # latest_rolling_stats has no shift, so values should be non-NaN
    assert not df["home_rolling_runs_scored"].isna().any()


def test_build_features_raises_on_missing_odds(tmp_path):
    """build_features raises RuntimeError when the odds cache is absent."""
    from mlb_edge_finder import features
    with patch("mlb_edge_finder.features.config.DATA_RAW_DIR", tmp_path), \
         patch("mlb_edge_finder.features.config.DATA_PROCESSED_DIR", tmp_path):
        with pytest.raises(RuntimeError, match="odds"):
            features.build_features(date(2025, 4, 22))


def test_load_features_raises_when_missing(tmp_path):
    """load_features raises FileNotFoundError when file is absent."""
    from mlb_edge_finder import features
    with patch("mlb_edge_finder.features.config.DATA_PROCESSED_DIR", tmp_path):
        with pytest.raises(FileNotFoundError):
            features.load_features(date(2025, 4, 22))
```

- [ ] **Step 2: Run the new test to verify it fails**

```bash
pytest tests/test_features.py::test_build_features_includes_rolling_cols -v
```

Expected: FAIL — `AssertionError: Missing column: home_rolling_runs_scored`

- [ ] **Step 3: Run the updated existing test to verify it still passes (join test needs the new fetch_historical mock)**

```bash
pytest tests/test_features.py::test_build_features_joins_home_and_away -v
```

Expected: FAIL — `TypeError` or `RuntimeError` because `fetch_historical` is now called but not mocked in the old version. This confirms the update was needed.

- [ ] **Step 4: Update `features.py`**

Replace the entire contents of `src/mlb_edge_finder/features.py` with:

```python
"""Merge odds and stats into a model-ready feature DataFrame."""
import logging
from datetime import date

import pandas as pd

from mlb_edge_finder import config
from mlb_edge_finder.historical_ingestion import fetch_historical
from mlb_edge_finder.odds_ingestion import load_cached_odds
from mlb_edge_finder.rolling_stats import latest_rolling_stats
from mlb_edge_finder.stats_ingestion import ODDS_NAME_TO_ABBR, load_cached_stats

logger = logging.getLogger(__name__)


def build_features(game_date: date) -> pd.DataFrame:
    """Join odds, stats, and rolling stats into one row per game.

    Loads cached odds and stats for game_date, fetches the current season's
    completed games (cache-first) to compute rolling team form stats, maps
    Odds API full team names to abbreviations, then joins all three data sources
    with home_/away_ prefixes.

    Args:
        game_date: Date whose cached odds and stats CSVs to load.

    Returns:
        DataFrame with one row per game. Columns include all odds fields plus
        home_<stat>/away_<stat> for every stat column and
        home_rolling_*/away_rolling_* for rolling stats.

    Raises:
        RuntimeError: If the cached odds or stats file for game_date is absent,
            or if the historical data fetch fails.
    """
    try:
        odds_df = load_cached_odds(game_date)
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"No cached odds for {game_date} — run fetch_odds() first"
        ) from exc

    try:
        stats_df = load_cached_stats(game_date)
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"No cached stats for {game_date} — run fetch_stats() first"
        ) from exc

    # Fetch current-season completed games for rolling stats (cache-first)
    try:
        hist_df = fetch_historical(game_date.year)
    except RuntimeError as exc:
        raise RuntimeError(
            f"Could not fetch historical data for {game_date.year} — {exc}"
        ) from exc

    # Map full team names → abbreviations
    odds_df = odds_df.copy()
    odds_df["home_abbr"] = odds_df["home_team"].map(ODDS_NAME_TO_ABBR)
    odds_df["away_abbr"] = odds_df["away_team"].map(ODDS_NAME_TO_ABBR)

    unmapped = pd.concat([
        odds_df.loc[odds_df["home_abbr"].isna(), "home_team"],
        odds_df.loc[odds_df["away_abbr"].isna(), "away_team"],
    ]).unique()
    if len(unmapped):
        logger.warning("Unmapped team names dropped from features: %s", list(unmapped))
    odds_df = odds_df.dropna(subset=["home_abbr", "away_abbr"])

    # Drop data_source; it varies by run and is not a model feature
    stats = stats_df.drop(columns=["data_source"], errors="ignore")

    # Build home/away stat frames with prefixes
    stat_cols = [c for c in stats.columns if c != "team_abbr"]
    home_stats = stats.rename(columns={"team_abbr": "home_abbr"} | {c: f"home_{c}" for c in stat_cols})
    away_stats = stats.rename(columns={"team_abbr": "away_abbr"} | {c: f"away_{c}" for c in stat_cols})

    df = odds_df.merge(home_stats, on="home_abbr", how="inner")
    df = df.merge(away_stats, on="away_abbr", how="inner")

    # Join rolling stats by team_abbr only (today's games haven't been played yet)
    rolling_df = latest_rolling_stats(hist_df)
    rolling_cols = [c for c in rolling_df.columns if c != "team_abbr"]
    home_rolling = rolling_df.rename(
        columns={"team_abbr": "home_abbr"} | {c: f"home_{c}" for c in rolling_cols}
    )
    away_rolling = rolling_df.rename(
        columns={"team_abbr": "away_abbr"} | {c: f"away_{c}" for c in rolling_cols}
    )
    df = df.merge(home_rolling[["home_abbr"] + [f"home_{c}" for c in rolling_cols]], on="home_abbr", how="left")
    df = df.merge(away_rolling[["away_abbr"] + [f"away_{c}" for c in rolling_cols]], on="away_abbr", how="left")

    logger.debug(
        "Built features: %d game(s), %d columns (home stat cols: %d)",
        len(df), len(df.columns), len(stat_cols),
    )

    config.DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.DATA_PROCESSED_DIR / f"features_{game_date}.csv"
    df.to_csv(out_path, index=False)
    logger.info("Wrote %d rows to %s", len(df), out_path)

    return df


def load_features(game_date: date) -> pd.DataFrame:
    """Load a previously built feature DataFrame from DATA_PROCESSED_DIR.

    Args:
        game_date: The date whose features CSV to load.

    Returns:
        DataFrame with the same schema as build_features().

    Raises:
        FileNotFoundError: If no features file exists for the given date.
    """
    path = config.DATA_PROCESSED_DIR / f"features_{game_date}.csv"
    if not path.exists():
        raise FileNotFoundError(f"No cached features for {game_date}: {path}")
    return pd.read_csv(path)
```

- [ ] **Step 5: Run all features tests**

```bash
pytest tests/test_features.py -v
```

Expected: all 6 tests PASS.

- [ ] **Step 6: Run full suite**

```bash
pytest tests/ -v
```

Expected: all 88 tests PASS.

- [ ] **Step 7: Commit**

```bash
git add tests/test_features.py src/mlb_edge_finder/features.py
git commit -m "feat: add rolling stats join to features.build_features()"
```

---

## Task 4: Update CLAUDE.md

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Update the Current Phase section**

In `## Current Phase`, replace:

```markdown
**Phase 5 complete.** Phases 1–3, 4a–4c, and 5 are done. Next: `compute_kelly()`, `__main__.py` CLI entry point.
```

with:

```markdown
**Phase 6 complete.** Phases 1–3, 4a–4c, 5, and 6 are done. Next: starting pitcher features, then `compute_kelly()` + `__main__.py` CLI.
```

And append to the bullet list in that section:

```markdown
- **6 complete:** `rolling_stats.py` — `compute_rolling_stats(historical_df, window=15)` (shift-1, for training) and `latest_rolling_stats(historical_df, window=15)` (no shift, for inference). `HISTORICAL_NAME_TO_ABBR` moved here from `training_data.py` (re-exported for backwards compatibility). `training_data._build_season()` and `features.build_features()` both join 8 new rolling columns: `home_/away_rolling_runs_scored`, `rolling_runs_allowed`, `rolling_win_pct`, `rolling_run_diff`.
```

- [ ] **Step 2: Add `rolling_stats.py` to the Module Responsibilities table**

In `## Module Responsibilities`, add a row:

```markdown
| `rolling_stats.py` | Compute rolling per-team stats from historical game results; owns `HISTORICAL_NAME_TO_ABBR` | — |
```

- [ ] **Step 3: Mark Phase 6 complete in the Roadmap**

Change:

```markdown
- [ ] **6 — Rolling window team stats** — replace end-of-season stat snapshots with per-game rolling N-game averages (e.g. last 15 games of batting/pitching). Affects `training_data.py` and `features.py`. Highest-impact model improvement.
```

to:

```markdown
- [x] **6 — Rolling window team stats** — `rolling_stats.py` computes 4 rolling features (runs_scored, runs_allowed, win_pct, run_diff) from cached historical game results. Joined into both training set and daily features. Window=15, season-only lookback, XGBoost handles NaN for early-season games.
```

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: mark Phase 6 complete in CLAUDE.md"
```
