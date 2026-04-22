# Odds Ingestion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `fetch_odds()`, `load_cached_odds()`, and private `_parse_response()` in `odds_ingestion.py` so the module fetches MLB moneyline odds from The Odds API v4, caches them to a dated CSV, and returns a clean long-format DataFrame.

**Architecture:** A private `_parse_response()` helper flattens the nested API JSON (bookmakers → markets → outcomes) into one row per bookmaker per game. `load_cached_odds()` reads a dated CSV from `data/raw/`. `fetch_odds(game_date, force=False)` checks for a cached file first; if absent or `force=True`, it calls the API, parses, writes the CSV, and returns the DataFrame.

**Tech Stack:** `requests` for HTTP, `pandas` for DataFrame construction and CSV I/O, `python-dotenv` / `config.py` for API key and path constants, `pytest` with `unittest.mock` for tests.

---

## File Map

| Action | Path | Responsibility |
|---|---|---|
| Modify | `src/mlb_edge_finder/odds_ingestion.py` | All three functions |
| Modify | `tests/test_odds_ingestion.py` | Unit tests for all new logic |

---

### Task 1: Implement `_parse_response()`

**Files:**
- Modify: `src/mlb_edge_finder/odds_ingestion.py`
- Modify: `tests/test_odds_ingestion.py`

- [ ] **Step 1: Write failing tests for `_parse_response`**

Add the following to `tests/test_odds_ingestion.py` (preserve the existing two smoke tests):

```python
import datetime
import pandas as pd
from mlb_edge_finder.odds_ingestion import _parse_response

GAME_DATE = datetime.date(2026, 4, 21)

SAMPLE_GAMES = [
    {
        "id": "game-abc",
        "commence_time": "2026-04-21T18:00:00Z",
        "home_team": "New York Yankees",
        "away_team": "Boston Red Sox",
        "bookmakers": [
            {
                "key": "draftkings",
                "markets": [
                    {
                        "key": "h2h",
                        "outcomes": [
                            {"name": "New York Yankees", "price": -150},
                            {"name": "Boston Red Sox", "price": 130},
                        ],
                    }
                ],
            },
            {
                "key": "fanduel",
                "markets": [
                    {
                        "key": "h2h",
                        "outcomes": [
                            {"name": "New York Yankees", "price": -145},
                            {"name": "Boston Red Sox", "price": 125},
                        ],
                    }
                ],
            },
        ],
    },
    {
        "id": "game-xyz",
        "commence_time": "2026-04-22T20:00:00Z",  # wrong date — should be excluded
        "home_team": "Chicago Cubs",
        "away_team": "St. Louis Cardinals",
        "bookmakers": [],
    },
]


def test_parse_response_columns():
    df = _parse_response(SAMPLE_GAMES, GAME_DATE)
    assert set(df.columns) == {
        "game_id", "home_team", "away_team",
        "home_odds_american", "away_odds_american",
        "bookmaker", "commence_time",
    }


def test_parse_response_row_count():
    # 1 game on the right date × 2 bookmakers = 2 rows
    df = _parse_response(SAMPLE_GAMES, GAME_DATE)
    assert len(df) == 2


def test_parse_response_filters_by_date():
    df = _parse_response(SAMPLE_GAMES, GAME_DATE)
    assert all(df["game_id"] == "game-abc")


def test_parse_response_odds_values():
    df = _parse_response(SAMPLE_GAMES, GAME_DATE)
    dk_row = df[df["bookmaker"] == "draftkings"].iloc[0]
    assert dk_row["home_odds_american"] == -150
    assert dk_row["away_odds_american"] == 130


def test_parse_response_empty_when_no_matching_date():
    df = _parse_response(SAMPLE_GAMES, datetime.date(2099, 1, 1))
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_odds_ingestion.py -v -k "parse_response"
```

Expected: ImportError or AttributeError — `_parse_response` does not exist yet.

- [ ] **Step 3: Implement `_parse_response` in `odds_ingestion.py`**

Replace the contents of `src/mlb_edge_finder/odds_ingestion.py` with:

```python
"""Fetch and cache MLB moneyline odds from The Odds API."""
import logging
from datetime import date

import pandas as pd
import requests

from mlb_edge_finder import config

logger = logging.getLogger(__name__)


def _parse_response(games: list[dict], game_date: date) -> pd.DataFrame:
    rows = []
    date_str = str(game_date)
    for game in games:
        if game["commence_time"][:10] != date_str:
            continue
        for bookmaker in game.get("bookmakers", []):
            h2h = next(
                (m for m in bookmaker.get("markets", []) if m["key"] == "h2h"),
                None,
            )
            if h2h is None:
                logger.debug("No h2h market for game %s / bookmaker %s", game["id"], bookmaker["key"])
                continue
            outcomes = {o["name"]: o["price"] for o in h2h["outcomes"]}
            home_odds = outcomes.get(game["home_team"])
            away_odds = outcomes.get(game["away_team"])
            if home_odds is None or away_odds is None:
                logger.debug("Missing outcome for game %s bookmaker %s", game["id"], bookmaker["key"])
                continue
            rows.append({
                "game_id": game["id"],
                "home_team": game["home_team"],
                "away_team": game["away_team"],
                "home_odds_american": int(home_odds),
                "away_odds_american": int(away_odds),
                "bookmaker": bookmaker["key"],
                "commence_time": game["commence_time"],
            })
    if not rows:
        logger.warning("No games found for %s after filtering by date", game_date)
    return pd.DataFrame(rows)


def fetch_odds(game_date: date, force: bool = False) -> pd.DataFrame:
    """Fetch MLB moneyline odds from The Odds API for a given date.

    Calls GET /v4/sports/{sport}/odds with market=h2h for the configured
    region. Writes the raw response to DATA_RAW_DIR/odds_YYYY-MM-DD.csv
    before returning. If a cached file already exists and force=False,
    returns the cached data without making an API call.

    Args:
        game_date: The date for which to fetch odds.
        force: If True, re-fetch from the API even if a cache file exists.

    Returns:
        DataFrame with columns: game_id, home_team, away_team,
        home_odds_american, away_odds_american, bookmaker, commence_time.

    Raises:
        RuntimeError: If ODDS_API_KEY is not set or the API returns non-200.
    """
    raise NotImplementedError


def load_cached_odds(game_date: date) -> pd.DataFrame:
    """Load previously fetched odds from DATA_RAW_DIR/odds_YYYY-MM-DD.csv.

    Args:
        game_date: The date whose cached CSV to load.

    Returns:
        DataFrame with the same schema as fetch_odds().

    Raises:
        FileNotFoundError: If no cached file exists for the given date.
    """
    raise NotImplementedError
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_odds_ingestion.py -v -k "parse_response"
```

Expected: 5 PASSED.

- [ ] **Step 5: Commit**

```bash
git add src/mlb_edge_finder/odds_ingestion.py tests/test_odds_ingestion.py
git commit -m "feat: implement _parse_response for odds ingestion"
```

---

### Task 2: Implement `load_cached_odds()`

**Files:**
- Modify: `src/mlb_edge_finder/odds_ingestion.py`
- Modify: `tests/test_odds_ingestion.py`

- [ ] **Step 1: Write failing tests for `load_cached_odds`**

Add to `tests/test_odds_ingestion.py`:

```python
import datetime
from pathlib import Path
import pandas as pd
import pytest
from mlb_edge_finder import odds_ingestion, config


def test_load_cached_odds_returns_dataframe(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_RAW_DIR", tmp_path)
    game_date = datetime.date(2026, 4, 21)
    csv_path = tmp_path / f"odds_{game_date}.csv"
    sample = pd.DataFrame([{
        "game_id": "game-abc",
        "home_team": "New York Yankees",
        "away_team": "Boston Red Sox",
        "home_odds_american": -150,
        "away_odds_american": 130,
        "bookmaker": "draftkings",
        "commence_time": "2026-04-21T18:00:00Z",
    }])
    sample.to_csv(csv_path, index=False)
    df = odds_ingestion.load_cached_odds(game_date)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    assert df.iloc[0]["game_id"] == "game-abc"


def test_load_cached_odds_raises_when_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_RAW_DIR", tmp_path)
    with pytest.raises(FileNotFoundError):
        odds_ingestion.load_cached_odds(datetime.date(2026, 4, 21))
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_odds_ingestion.py -v -k "load_cached"
```

Expected: FAIL — `NotImplementedError`.

- [ ] **Step 3: Implement `load_cached_odds` in `odds_ingestion.py`**

Replace the `load_cached_odds` stub:

```python
def load_cached_odds(game_date: date) -> pd.DataFrame:
    """Load previously fetched odds from DATA_RAW_DIR/odds_YYYY-MM-DD.csv.

    Args:
        game_date: The date whose cached CSV to load.

    Returns:
        DataFrame with the same schema as fetch_odds().

    Raises:
        FileNotFoundError: If no cached file exists for the given date.
    """
    cache_path = config.DATA_RAW_DIR / f"odds_{game_date}.csv"
    if not cache_path.exists():
        raise FileNotFoundError(f"No cached odds for {game_date}: {cache_path}")
    return pd.read_csv(cache_path)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_odds_ingestion.py -v -k "load_cached"
```

Expected: 2 PASSED.

- [ ] **Step 5: Commit**

```bash
git add src/mlb_edge_finder/odds_ingestion.py tests/test_odds_ingestion.py
git commit -m "feat: implement load_cached_odds"
```

---

### Task 3: Implement `fetch_odds()` — caching logic

**Files:**
- Modify: `src/mlb_edge_finder/odds_ingestion.py`
- Modify: `tests/test_odds_ingestion.py`

- [ ] **Step 1: Write failing tests for the caching behaviour of `fetch_odds`**

Add to `tests/test_odds_ingestion.py`:

```python
import datetime
from pathlib import Path
from unittest.mock import patch, MagicMock
import pandas as pd
import pytest
from mlb_edge_finder import odds_ingestion, config


def _write_cache(tmp_path: Path, game_date: datetime.date) -> Path:
    csv_path = tmp_path / f"odds_{game_date}.csv"
    pd.DataFrame([{
        "game_id": "cached-game",
        "home_team": "A",
        "away_team": "B",
        "home_odds_american": -110,
        "away_odds_american": -110,
        "bookmaker": "draftkings",
        "commence_time": f"{game_date}T18:00:00Z",
    }]).to_csv(csv_path, index=False)
    return csv_path


def test_fetch_odds_returns_cache_when_exists(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_RAW_DIR", tmp_path)
    monkeypatch.setattr(config, "ODDS_API_KEY", "test-key")
    game_date = datetime.date(2026, 4, 21)
    _write_cache(tmp_path, game_date)
    with patch("mlb_edge_finder.odds_ingestion.requests.get") as mock_get:
        df = odds_ingestion.fetch_odds(game_date)
        mock_get.assert_not_called()
    assert df.iloc[0]["game_id"] == "cached-game"


def test_fetch_odds_force_bypasses_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_RAW_DIR", tmp_path)
    monkeypatch.setattr(config, "ODDS_API_KEY", "test-key")
    game_date = datetime.date(2026, 4, 21)
    _write_cache(tmp_path, game_date)
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = []
    with patch("mlb_edge_finder.odds_ingestion.requests.get", return_value=mock_response):
        df = odds_ingestion.fetch_odds(game_date, force=True)
    assert len(df) == 0  # empty because API returned []


def test_fetch_odds_writes_csv_on_api_call(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_RAW_DIR", tmp_path)
    monkeypatch.setattr(config, "ODDS_API_KEY", "test-key")
    game_date = datetime.date(2026, 4, 21)
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = [
        {
            "id": "game-new",
            "commence_time": "2026-04-21T18:00:00Z",
            "home_team": "Houston Astros",
            "away_team": "Texas Rangers",
            "bookmakers": [
                {
                    "key": "draftkings",
                    "markets": [
                        {
                            "key": "h2h",
                            "outcomes": [
                                {"name": "Houston Astros", "price": -120},
                                {"name": "Texas Rangers", "price": 100},
                            ],
                        }
                    ],
                }
            ],
        }
    ]
    with patch("mlb_edge_finder.odds_ingestion.requests.get", return_value=mock_response):
        df = odds_ingestion.fetch_odds(game_date)
    cache_path = tmp_path / f"odds_{game_date}.csv"
    assert cache_path.exists()
    assert df.iloc[0]["game_id"] == "game-new"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_odds_ingestion.py -v -k "fetch_odds"
```

Expected: FAIL — `NotImplementedError`.

- [ ] **Step 3: Implement `fetch_odds` in `odds_ingestion.py`**

Replace the `fetch_odds` stub:

```python
def fetch_odds(game_date: date, force: bool = False) -> pd.DataFrame:
    """Fetch MLB moneyline odds from The Odds API for a given date.

    Calls GET /v4/sports/{sport}/odds with market=h2h for the configured
    region. Writes the raw response to DATA_RAW_DIR/odds_YYYY-MM-DD.csv
    before returning. If a cached file already exists and force=False,
    returns the cached data without making an API call.

    Args:
        game_date: The date for which to fetch odds.
        force: If True, re-fetch from the API even if a cache file exists.

    Returns:
        DataFrame with columns: game_id, home_team, away_team,
        home_odds_american, away_odds_american, bookmaker, commence_time.

    Raises:
        RuntimeError: If ODDS_API_KEY is not set or the API returns non-200.
    """
    cache_path = config.DATA_RAW_DIR / f"odds_{game_date}.csv"
    if cache_path.exists() and not force:
        logger.debug("Cache hit for %s, loading from disk", game_date)
        return load_cached_odds(game_date)

    if not config.ODDS_API_KEY:
        msg = "ODDS_API_KEY is not set"
        logger.error(msg)
        raise RuntimeError(msg)

    url = f"https://api.the-odds-api.com/v4/sports/{config.SPORT}/odds"
    params = {
        "apiKey": config.ODDS_API_KEY,
        "regions": config.REGION,
        "markets": config.MARKET,
        "dateFormat": "iso",
        "oddsFormat": "american",
    }

    response = requests.get(url, params=params, timeout=30)
    if response.status_code != 200:
        msg = f"Odds API returned {response.status_code}: {response.text}"
        logger.error(msg)
        raise RuntimeError(msg)

    df = _parse_response(response.json(), game_date)

    config.DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path, index=False)
    logger.info("Wrote %d rows to %s", len(df), cache_path)

    return df
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_odds_ingestion.py -v -k "fetch_odds"
```

Expected: 3 PASSED.

- [ ] **Step 5: Commit**

```bash
git add src/mlb_edge_finder/odds_ingestion.py tests/test_odds_ingestion.py
git commit -m "feat: implement fetch_odds with caching and force flag"
```

---

### Task 4: Error handling for `fetch_odds()`

**Files:**
- Modify: `tests/test_odds_ingestion.py`

- [ ] **Step 1: Write failing tests for error paths**

Add to `tests/test_odds_ingestion.py`:

```python
import datetime
from unittest.mock import patch, MagicMock
import pytest
from mlb_edge_finder import odds_ingestion, config


def test_fetch_odds_raises_when_no_api_key(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_RAW_DIR", tmp_path)
    monkeypatch.setattr(config, "ODDS_API_KEY", "")
    with pytest.raises(RuntimeError, match="ODDS_API_KEY is not set"):
        odds_ingestion.fetch_odds(datetime.date(2026, 4, 21))


def test_fetch_odds_raises_on_non_200(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_RAW_DIR", tmp_path)
    monkeypatch.setattr(config, "ODDS_API_KEY", "test-key")
    mock_response = MagicMock()
    mock_response.status_code = 401
    mock_response.text = "Unauthorized"
    with patch("mlb_edge_finder.odds_ingestion.requests.get", return_value=mock_response):
        with pytest.raises(RuntimeError, match="Odds API returned 401"):
            odds_ingestion.fetch_odds(datetime.date(2026, 4, 21))
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_odds_ingestion.py -v -k "raises"
```

Expected: FAIL — `NotImplementedError` (the error paths aren't reached yet since `fetch_odds` isn't implemented; after Task 3 they should fail for the right reasons).

> Note: If Task 3 is already complete, these tests may already pass. Run them anyway to confirm.

- [ ] **Step 3: Run all odds ingestion tests**

```bash
pytest tests/test_odds_ingestion.py -v
```

Expected: All tests PASSED (the implementation from Task 3 already handles these paths).

- [ ] **Step 4: Run the full test suite to confirm no regressions**

```bash
pytest tests/ -v
```

Expected: All 17 existing smoke tests plus new tests PASSED. Zero failures.

- [ ] **Step 5: Commit**

```bash
git add tests/test_odds_ingestion.py
git commit -m "test: add error path tests for fetch_odds"
```

---

### Task 5: Update CLAUDE.md

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Update Current Phase and Roadmap in `CLAUDE.md`**

In `CLAUDE.md`, update the **Current Phase** section to:

```markdown
## Current Phase

**Phase 2 — Odds ingestion complete.** `odds_ingestion.fetch_odds()` and `load_cached_odds()` are fully implemented with caching (`force=False` skips re-fetch), date filtering, and error handling. `_parse_response()` flattens bookmaker JSON to a long-format DataFrame.

**Next:** Implement `stats_ingestion.fetch_stats()` and `load_cached_stats()` — those two complete data ingestion and unlock `features.build_features()`.
```

Update the **Roadmap** checkboxes:

```markdown
- [x] Implement `odds_ingestion.fetch_odds()` and `load_cached_odds()`
- [ ] Implement `stats_ingestion.fetch_stats()` and `load_cached_stats()`
```

- [ ] **Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md — odds ingestion complete, phase 2"
```
