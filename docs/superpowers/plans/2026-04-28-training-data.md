# Training Data (Phase 4b) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `training_data.build_training_set(seasons)` — joins end-of-season team stats to historical game results, producing a labeled CSV ready for XGBoost training.

**Architecture:** New module `training_data.py` owns two module-level dicts (name mapping, legacy abbr normalization) and two public functions (`build_training_set`, `load_training_set`). Per season: load cached historical games, auto-fetch end-of-season stats via `fetch_stats(date(season, 9, 28))`, normalize abbreviations, double-join stats with `home_`/`away_` prefixes, tag with `season` column. Concatenate all seasons, write to `data/processed/training_{min}-{max}.csv`.

**Tech Stack:** pandas, Python stdlib (`datetime.date`), pytest. Reuses `load_cached_historical` from `historical_ingestion.py` and `fetch_stats` from `stats_ingestion.py`.

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `src/mlb_edge_finder/training_data.py` | Create | New module — name mapping, legacy normalization, build/load functions |
| `tests/test_training_data.py` | Create | Smoke + integration tests |

---

### Task 1: Create `training_data.py` stub with constants

**Files:**
- Create: `src/mlb_edge_finder/training_data.py`
- Test: `tests/test_training_data.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_training_data.py`:

```python
"""Tests for training_data module."""
import inspect
from datetime import date
from unittest.mock import patch

import pandas as pd
import pytest


def test_build_training_set_signature():
    from mlb_edge_finder import training_data
    assert callable(training_data.build_training_set)
    sig = inspect.signature(training_data.build_training_set)
    assert "seasons" in sig.parameters
    assert "force" in sig.parameters


def test_load_training_set_signature():
    from mlb_edge_finder import training_data
    assert callable(training_data.load_training_set)
    sig = inspect.signature(training_data.load_training_set)
    assert "seasons" in sig.parameters


def test_historical_name_to_abbr_covers_all_30_teams():
    from mlb_edge_finder.training_data import HISTORICAL_NAME_TO_ABBR
    # All 30 current franchise abbreviations must be reachable
    expected_abbrs = {
        "ARI", "ATL", "BAL", "BOS", "CHC", "CWS", "CIN", "CLE", "COL", "DET",
        "HOU", "KC", "LAA", "LAD", "MIA", "MIL", "MIN", "NYM", "NYY", "ATH",
        "PHI", "PIT", "SD", "SF", "SEA", "STL", "TB", "TEX", "TOR", "WSH",
    }
    assert expected_abbrs == set(HISTORICAL_NAME_TO_ABBR.values())


def test_historical_name_to_abbr_maps_oakland():
    from mlb_edge_finder.training_data import HISTORICAL_NAME_TO_ABBR
    # Both statsapi names for the Athletics franchise map to ATH
    assert HISTORICAL_NAME_TO_ABBR["Oakland Athletics"] == "ATH"
    assert HISTORICAL_NAME_TO_ABBR["Athletics"] == "ATH"


def test_legacy_abbr_normalize_maps_oak_to_ath():
    from mlb_edge_finder.training_data import _LEGACY_ABBR_NORMALIZE
    assert _LEGACY_ABBR_NORMALIZE["OAK"] == "ATH"
```

- [ ] **Step 2: Run to verify they fail**

```bash
pytest tests/test_training_data.py -v
```

Expected: all 5 tests fail with `ModuleNotFoundError` or `ImportError`.

- [ ] **Step 3: Create `training_data.py` with constants and stubs**

Create `src/mlb_edge_finder/training_data.py`:

```python
"""Build and cache a labeled training dataset for XGBoost model training."""
import logging
from datetime import date

import pandas as pd

from mlb_edge_finder import config
from mlb_edge_finder.historical_ingestion import load_cached_historical
from mlb_edge_finder.stats_ingestion import fetch_stats

logger = logging.getLogger(__name__)

# statsapi full team names → current franchise abbreviations.
# Always use the current abbreviation regardless of historical team name,
# so training features are consistent with inference-time features.
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
    "Oakland Athletics": "ATH",   # pre-2025 statsapi name; franchise moved to Sacramento
    "Athletics": "ATH",           # 2025+ statsapi name
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

# FanGraphs abbreviations that changed between seasons → current abbreviation.
# Applied to the stats DataFrame before joining so both join sides use current identifiers.
_LEGACY_ABBR_NORMALIZE: dict[str, str] = {
    "OAK": "ATH",   # Oakland → Sacramento Athletics
}

_SNAPSHOT_MONTH = 9
_SNAPSHOT_DAY = 28


def build_training_set(seasons: list[int], force: bool = False) -> pd.DataFrame:
    raise NotImplementedError


def load_training_set(seasons: list[int]) -> pd.DataFrame:
    raise NotImplementedError
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_training_data.py -v
```

Expected: all 5 pass.

- [ ] **Step 5: Commit**

```bash
git add src/mlb_edge_finder/training_data.py tests/test_training_data.py
git commit -m "feat: scaffold training_data module with name mapping constants"
```

---

### Task 2: Implement `load_training_set`

**Files:**
- Modify: `src/mlb_edge_finder/training_data.py`
- Modify: `tests/test_training_data.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_training_data.py`:

```python
def test_load_training_set_raises_when_missing(tmp_path):
    from mlb_edge_finder import training_data
    with patch("mlb_edge_finder.training_data.config.DATA_PROCESSED_DIR", tmp_path):
        with pytest.raises(FileNotFoundError):
            training_data.load_training_set([2023, 2024, 2025])
```

- [ ] **Step 2: Run to verify it fails**

```bash
pytest tests/test_training_data.py::test_load_training_set_raises_when_missing -v
```

Expected: FAIL — `NotImplementedError` from the stub.

- [ ] **Step 3: Implement `load_training_set`**

Replace the `load_training_set` stub in `training_data.py`:

```python
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

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_training_data.py -v
```

Expected: all 6 pass.

- [ ] **Step 5: Commit**

```bash
git add src/mlb_edge_finder/training_data.py tests/test_training_data.py
git commit -m "feat: implement load_training_set with FileNotFoundError"
```

---

### Task 3: Implement `build_training_set` — core join logic

**Files:**
- Modify: `src/mlb_edge_finder/training_data.py`
- Modify: `tests/test_training_data.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_training_data.py`:

```python
def _make_hist(home="New York Yankees", away="Boston Red Sox"):
    return pd.DataFrame([{
        "game_date": "2024-04-01",
        "home_name": home,
        "away_name": away,
        "home_score": 5,
        "away_score": 3,
        "home_win": 1,
    }])


def _make_stats():
    return pd.DataFrame([
        {
            "team_abbr": "NYY", "bat_avg": 0.260, "obp": 0.330, "slg": 0.420,
            "ops": 0.750, "runs_per_game": 4.8, "era": 3.80, "whip": 1.20,
            "k_per_9": 9.0, "bb_per_9": 3.0, "data_source": "mlb_api",
        },
        {
            "team_abbr": "BOS", "bat_avg": 0.255, "obp": 0.320, "slg": 0.410,
            "ops": 0.730, "runs_per_game": 4.5, "era": 4.10, "whip": 1.30,
            "k_per_9": 8.5, "bb_per_9": 3.2, "data_source": "mlb_api",
        },
    ])


def test_build_training_set_joins_home_and_away_stats(tmp_path):
    from mlb_edge_finder import training_data
    with patch("mlb_edge_finder.training_data.load_cached_historical", return_value=_make_hist()), \
         patch("mlb_edge_finder.training_data.fetch_stats", return_value=_make_stats()), \
         patch("mlb_edge_finder.training_data.config.DATA_PROCESSED_DIR", tmp_path):
        df = training_data.build_training_set([2024])
    assert len(df) == 1
    assert "home_bat_avg" in df.columns
    assert "away_bat_avg" in df.columns
    assert "home_era" in df.columns
    assert "away_era" in df.columns
    assert "data_source" not in df.columns


def test_build_training_set_includes_season_column(tmp_path):
    from mlb_edge_finder import training_data
    with patch("mlb_edge_finder.training_data.load_cached_historical", return_value=_make_hist()), \
         patch("mlb_edge_finder.training_data.fetch_stats", return_value=_make_stats()), \
         patch("mlb_edge_finder.training_data.config.DATA_PROCESSED_DIR", tmp_path):
        df = training_data.build_training_set([2024])
    assert "season" in df.columns
    assert df["season"].iloc[0] == 2024


def test_build_training_set_preserves_home_win(tmp_path):
    from mlb_edge_finder import training_data
    with patch("mlb_edge_finder.training_data.load_cached_historical", return_value=_make_hist()), \
         patch("mlb_edge_finder.training_data.fetch_stats", return_value=_make_stats()), \
         patch("mlb_edge_finder.training_data.config.DATA_PROCESSED_DIR", tmp_path):
        df = training_data.build_training_set([2024])
    assert "home_win" in df.columns
    assert df["home_win"].iloc[0] == 1


def test_build_training_set_keeps_name_and_abbr_columns(tmp_path):
    from mlb_edge_finder import training_data
    with patch("mlb_edge_finder.training_data.load_cached_historical", return_value=_make_hist()), \
         patch("mlb_edge_finder.training_data.fetch_stats", return_value=_make_stats()), \
         patch("mlb_edge_finder.training_data.config.DATA_PROCESSED_DIR", tmp_path):
        df = training_data.build_training_set([2024])
    for col in ("home_name", "away_name", "home_abbr", "away_abbr"):
        assert col in df.columns
```

- [ ] **Step 2: Run to verify they fail**

```bash
pytest tests/test_training_data.py::test_build_training_set_joins_home_and_away_stats \
       tests/test_training_data.py::test_build_training_set_includes_season_column \
       tests/test_training_data.py::test_build_training_set_preserves_home_win \
       tests/test_training_data.py::test_build_training_set_keeps_name_and_abbr_columns -v
```

Expected: all FAIL with `NotImplementedError`.

- [ ] **Step 3: Implement the private `_build_season` helper and `build_training_set`**

Replace the `build_training_set` stub in `training_data.py` (add `_build_season` above it):

```python
def _build_season(season: int) -> pd.DataFrame:
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

    # Double-join with home_/away_ prefixes (same pattern as features.py)
    stat_cols = [c for c in stats.columns if c != "team_abbr"]
    home_stats = stats.rename(columns={"team_abbr": "home_abbr"} | {c: f"home_{c}" for c in stat_cols})
    away_stats = stats.rename(columns={"team_abbr": "away_abbr"} | {c: f"away_{c}" for c in stat_cols})

    df = hist.merge(home_stats, on="home_abbr", how="inner")
    df = df.merge(away_stats, on="away_abbr", how="inner")
    df["season"] = season

    logger.debug("Season %d: %d games, %d columns", season, len(df), len(df.columns))
    return df


def build_training_set(seasons: list[int], force: bool = False) -> pd.DataFrame:
    """Build and cache a labeled training set by joining historical games with end-of-season stats.

    For each season, loads historical game results and fetches end-of-season stats
    (September 28 snapshot), normalizes abbreviations, joins stats twice with home_/away_
    prefixes, and tags rows with a season column. Concatenates all seasons.

    Args:
        seasons: List of season years to include (e.g. [2023, 2024, 2025]).
        force: If True, rebuild even if a cache file exists.

    Returns:
        DataFrame with one row per game. Columns: game_date, season, home_name, away_name,
        home_abbr, away_abbr, home_win, plus home_<stat> and away_<stat> for every stat column.
        FanGraphs-specific columns (home_w_oba, home_bat_wrc_plus, home_fip) appear when present.

    Raises:
        RuntimeError: If historical data is missing for any season (stats are auto-fetched).
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
```

- [ ] **Step 4: Run all tests to verify they pass**

```bash
pytest tests/test_training_data.py -v
```

Expected: all 10 pass.

- [ ] **Step 5: Commit**

```bash
git add src/mlb_edge_finder/training_data.py tests/test_training_data.py
git commit -m "feat: implement build_training_set with double-join and season column"
```

---

### Task 4: Error handling and edge case tests

**Files:**
- Modify: `tests/test_training_data.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_training_data.py`:

```python
def test_build_training_set_raises_runtime_error_when_historical_missing(tmp_path):
    from mlb_edge_finder import training_data
    with patch("mlb_edge_finder.training_data.load_cached_historical",
               side_effect=FileNotFoundError("no file")), \
         patch("mlb_edge_finder.training_data.config.DATA_PROCESSED_DIR", tmp_path):
        with pytest.raises(RuntimeError, match="fetch_historical"):
            training_data.build_training_set([2024])


def test_build_training_set_drops_unmapped_teams(tmp_path):
    from mlb_edge_finder import training_data
    hist = pd.DataFrame([
        {"game_date": "2024-04-01", "home_name": "New York Yankees",
         "away_name": "Boston Red Sox", "home_score": 5, "away_score": 3, "home_win": 1},
        {"game_date": "2024-04-01", "home_name": "Unknown Team",
         "away_name": "Boston Red Sox", "home_score": 2, "away_score": 1, "home_win": 1},
    ])
    with patch("mlb_edge_finder.training_data.load_cached_historical", return_value=hist), \
         patch("mlb_edge_finder.training_data.fetch_stats", return_value=_make_stats()), \
         patch("mlb_edge_finder.training_data.config.DATA_PROCESSED_DIR", tmp_path):
        df = training_data.build_training_set([2024])
    assert len(df) == 1
    assert df["home_name"].iloc[0] == "New York Yankees"


def test_build_training_set_applies_legacy_abbr_normalization(tmp_path):
    from mlb_edge_finder import training_data
    hist = _make_hist(home="Oakland Athletics", away="New York Yankees")
    stats = pd.DataFrame([
        {"team_abbr": "OAK", "bat_avg": 0.240, "obp": 0.310, "slg": 0.390,
         "ops": 0.700, "runs_per_game": 4.0, "era": 4.50, "whip": 1.35,
         "k_per_9": 8.0, "bb_per_9": 3.5, "data_source": "fangraphs"},
        {"team_abbr": "NYY", "bat_avg": 0.260, "obp": 0.330, "slg": 0.420,
         "ops": 0.750, "runs_per_game": 4.8, "era": 3.80, "whip": 1.20,
         "k_per_9": 9.0, "bb_per_9": 3.0, "data_source": "fangraphs"},
    ])
    with patch("mlb_edge_finder.training_data.load_cached_historical", return_value=hist), \
         patch("mlb_edge_finder.training_data.fetch_stats", return_value=stats), \
         patch("mlb_edge_finder.training_data.config.DATA_PROCESSED_DIR", tmp_path):
        df = training_data.build_training_set([2024])
    # Oakland → ATH via _LEGACY_ABBR_NORMALIZE; should match and produce 1 row
    assert len(df) == 1
    assert df["home_abbr"].iloc[0] == "ATH"


def test_build_training_set_cache_first(tmp_path):
    from mlb_edge_finder import training_data
    # Pre-write a fake cache file
    out_path = tmp_path / "training_2024-2024.csv"
    cached_df = pd.DataFrame([{"game_date": "2024-04-01", "season": 2024, "home_win": 1}])
    cached_df.to_csv(out_path, index=False)
    with patch("mlb_edge_finder.training_data.load_cached_historical") as mock_hist, \
         patch("mlb_edge_finder.training_data.config.DATA_PROCESSED_DIR", tmp_path):
        df = training_data.build_training_set([2024])
    # Should NOT call load_cached_historical at all (served from cache)
    mock_hist.assert_not_called()
    assert len(df) == 1


def test_build_training_set_multi_season_concatenates(tmp_path):
    from mlb_edge_finder import training_data

    def mock_hist(season):
        return pd.DataFrame([{
            "game_date": f"{season}-04-01",
            "home_name": "New York Yankees",
            "away_name": "Boston Red Sox",
            "home_score": 5, "away_score": 3, "home_win": 1,
        }])

    with patch("mlb_edge_finder.training_data.load_cached_historical", side_effect=mock_hist), \
         patch("mlb_edge_finder.training_data.fetch_stats", return_value=_make_stats()), \
         patch("mlb_edge_finder.training_data.config.DATA_PROCESSED_DIR", tmp_path):
        df = training_data.build_training_set([2023, 2024])

    assert len(df) == 2
    assert set(df["season"]) == {2023, 2024}
    # Output file uses min-max range
    assert (tmp_path / "training_2023-2024.csv").exists()
```

- [ ] **Step 2: Run to verify they fail**

```bash
pytest tests/test_training_data.py::test_build_training_set_raises_runtime_error_when_historical_missing \
       tests/test_training_data.py::test_build_training_set_drops_unmapped_teams \
       tests/test_training_data.py::test_build_training_set_applies_legacy_abbr_normalization \
       tests/test_training_data.py::test_build_training_set_cache_first \
       tests/test_training_data.py::test_build_training_set_multi_season_concatenates -v
```

Expected: all FAIL (the implementation exists but these test new behaviour).

- [ ] **Step 3: Run the full test suite to verify all pass (no new code needed)**

These tests exercise the code written in Task 3. If any fail, fix the implementation in `training_data.py` before proceeding.

```bash
pytest tests/test_training_data.py -v
```

Expected: all 15 pass.

- [ ] **Step 4: Run the full project test suite to check for regressions**

```bash
pytest tests/ -v
```

Expected: all tests pass (41 existing + 15 new).

- [ ] **Step 5: Commit**

```bash
git add tests/test_training_data.py
git commit -m "test: add error handling and edge case tests for training_data"
```

---

### Task 5: Update CLAUDE.md

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Mark Phase 4b complete and update the module table**

In `CLAUDE.md`, find the `## Current Phase` section and update it:

```markdown
## Current Phase

**Phase 4 — Model training in progress.** Phases 1–3 and 4a–4b are complete.
```

In the module responsibilities table, update `training_data.py` row from stub to complete:

```markdown
| `training_data.py` | Join end-of-season stats to game results for model training | `data/processed/training_YYYY-YYYY.csv` |
```

In the Roadmap section, mark 4b done:

```markdown
- [x] **4b** — `training_data.build_training_set(seasons)` joining end-of-season stats to game results
```

- [ ] **Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: mark Phase 4b complete in CLAUDE.md"
```
