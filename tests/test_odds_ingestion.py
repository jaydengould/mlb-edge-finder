"""Smoke tests: odds_ingestion exposes expected public API."""
import datetime
import inspect

import pandas as pd
import pytest

from mlb_win_probability import config, odds_ingestion
from mlb_win_probability.odds_ingestion import _parse_response


def test_fetch_odds_signature():
    """fetch_odds should accept game_date and return a DataFrame."""
    assert callable(odds_ingestion.fetch_odds)
    sig = inspect.signature(odds_ingestion.fetch_odds)
    assert "game_date" in sig.parameters


def test_load_cached_odds_signature():
    """load_cached_odds should accept game_date and return a DataFrame."""
    assert callable(odds_ingestion.load_cached_odds)
    sig = inspect.signature(odds_ingestion.load_cached_odds)
    assert "game_date" in sig.parameters


# ---------------------------------------------------------------------------
# _parse_response
# ---------------------------------------------------------------------------

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
        "commence_time",
    }


def test_parse_response_row_count():
    # 2 bookmakers for 1 matching game collapse to 1 deduplicated row
    df = _parse_response(SAMPLE_GAMES, GAME_DATE)
    assert len(df) == 1


def test_parse_response_filters_by_date():
    df = _parse_response(SAMPLE_GAMES, GAME_DATE)
    assert all(df["game_id"] == "game-abc")


def test_parse_response_odds_values():
    # Best available = max American odds across bookmakers:
    # home: max(-150, -145) = -145; away: max(130, 125) = 130
    df = _parse_response(SAMPLE_GAMES, GAME_DATE)
    row = df.iloc[0]
    assert row["home_odds_american"] == -145
    assert row["away_odds_american"] == 130


def test_parse_response_empty_when_no_matching_date():
    df = _parse_response(SAMPLE_GAMES, datetime.date(2099, 1, 1))
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0


# ---------------------------------------------------------------------------
# load_cached_odds
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# fetch_odds — caching behaviour
# ---------------------------------------------------------------------------

from pathlib import Path
from unittest.mock import MagicMock, patch


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
    with patch("mlb_win_probability.odds_ingestion.requests.get") as mock_get:
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
    with patch("mlb_win_probability.odds_ingestion.requests.get", return_value=mock_response):
        df = odds_ingestion.fetch_odds(game_date, force=True)
    assert len(df) == 0  # empty because API returned []


def test_fetch_odds_writes_csv_on_api_call(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_RAW_DIR", tmp_path)
    monkeypatch.setattr(config, "ODDS_API_KEY", "test-key")
    game_date = datetime.date(2099, 1, 1)  # matches far-future commence_time in mock
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = [
        {
            "id": "game-new",
            "commence_time": "2099-01-01T18:00:00Z",  # far future — passes pre-game filter
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
    with patch("mlb_win_probability.odds_ingestion.requests.get", return_value=mock_response):
        df = odds_ingestion.fetch_odds(game_date)
    cache_path = tmp_path / f"odds_{game_date}.csv"
    assert cache_path.exists()
    assert df.iloc[0]["game_id"] == "game-new"


# ---------------------------------------------------------------------------
# fetch_odds — error paths
# ---------------------------------------------------------------------------


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
    with patch("mlb_win_probability.odds_ingestion.requests.get", return_value=mock_response):
        with pytest.raises(RuntimeError, match="Odds API returned 401"):
            odds_ingestion.fetch_odds(datetime.date(2026, 4, 21))
