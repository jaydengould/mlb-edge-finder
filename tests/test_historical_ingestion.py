"""Tests for historical_ingestion module."""
import inspect
from unittest.mock import patch

import pandas as pd
import pytest


def _make_game(home, away, home_score, away_score, game_type="R", status="Final"):
    return {
        "game_id": 1,
        "game_date": "2024-04-01",
        "game_datetime": "2024-04-01T18:00:00Z",
        "game_type": game_type,
        "status": status,
        "home_name": home,
        "away_name": away,
        "home_id": 1,
        "away_id": 2,
        "home_score": home_score,
        "away_score": away_score,
        "doubleheader": "N",
        "game_num": 1,
        "home_probable_pitcher": "",
        "away_probable_pitcher": "",
        "home_pitcher_note": "",
        "away_pitcher_note": "",
        "current_inning": 9,
        "inning_state": "End",
        "venue_id": 1,
        "venue_name": "Stadium",
        "national_broadcasts": [],
        "series_status": "",
        "summary": "",
    }


def test_fetch_historical_signature():
    from mlb_edge_finder import historical_ingestion
    assert callable(historical_ingestion.fetch_historical)
    sig = inspect.signature(historical_ingestion.fetch_historical)
    assert "season" in sig.parameters
    assert "force" in sig.parameters


def test_load_cached_historical_signature():
    from mlb_edge_finder import historical_ingestion
    assert callable(historical_ingestion.load_cached_historical)
    sig = inspect.signature(historical_ingestion.load_cached_historical)
    assert "season" in sig.parameters


def test_fetch_all_historical_signature():
    from mlb_edge_finder import historical_ingestion
    assert callable(historical_ingestion.fetch_all_historical)
    sig = inspect.signature(historical_ingestion.fetch_all_historical)
    assert "force" in sig.parameters


def test_load_cached_historical_raises_when_missing(tmp_path):
    from mlb_edge_finder import historical_ingestion
    with patch("mlb_edge_finder.historical_ingestion.config.DATA_RAW_DIR", tmp_path):
        with pytest.raises(FileNotFoundError):
            historical_ingestion.load_cached_historical(2024)


def test_fetch_historical_filters_and_derives_home_win(tmp_path):
    games = [
        _make_game("Yankees", "Red Sox", 5, 3),           # R/Final, home win
        _make_game("Cubs", "Cardinals", 1, 4),             # R/Final, away win
        _make_game("Dodgers", "Giants", 2, 2, status="Postponed"),  # excluded
        _make_game("Mets", "Phillies", 3, 1, game_type="S"),        # excluded spring
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


def test_fetch_historical_uses_cache(tmp_path):
    import pandas as pd
    from mlb_edge_finder import historical_ingestion
    cached = pd.DataFrame([{
        "game_date": "2024-04-01", "home_name": "Yankees", "away_name": "Red Sox",
        "home_score": 5, "away_score": 3, "home_win": 1,
    }])
    cache_file = tmp_path / "historical_2024.csv"
    cached.to_csv(cache_file, index=False)

    with patch("mlb_edge_finder.historical_ingestion.config.DATA_RAW_DIR", tmp_path), \
         patch("mlb_edge_finder.historical_ingestion.statsapi.schedule") as mock_api:
        df = historical_ingestion.fetch_historical(2024)
        mock_api.assert_not_called()

    assert len(df) == 1


def test_fetch_historical_raises_on_api_failure(tmp_path):
    from mlb_edge_finder import historical_ingestion
    with patch("mlb_edge_finder.historical_ingestion.statsapi.schedule", side_effect=Exception("timeout")), \
         patch("mlb_edge_finder.historical_ingestion.config.DATA_RAW_DIR", tmp_path):
        with pytest.raises(RuntimeError, match="statsapi.schedule failed"):
            historical_ingestion.fetch_historical(2024, force=True)


def test_fetch_all_historical_concatenates(tmp_path):
    from mlb_edge_finder import historical_ingestion
    one_game = [_make_game("Yankees", "Red Sox", 5, 3)]
    with patch("mlb_edge_finder.historical_ingestion.statsapi.schedule", return_value=one_game), \
         patch("mlb_edge_finder.historical_ingestion.config.DATA_RAW_DIR", tmp_path):
        df = historical_ingestion.fetch_all_historical(force=True)
    # 3 seasons × 1 game each
    assert len(df) == 3
    assert "home_starter_name" in df.columns
    assert "away_starter_name" in df.columns


# helper used in test_fetch_historical_filters_and_derives_home_win
def historical_ingestion_module():
    from mlb_edge_finder import historical_ingestion
    return historical_ingestion
