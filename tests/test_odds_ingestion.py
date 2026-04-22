"""Smoke tests: odds_ingestion exposes expected public API."""
import datetime
import inspect

import pandas as pd
import pytest

from mlb_edge_finder import config, odds_ingestion
from mlb_edge_finder.odds_ingestion import _parse_response


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
