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


def test_load_training_set_raises_when_missing(tmp_path):
    from mlb_edge_finder import training_data
    with patch("mlb_edge_finder.training_data.config.DATA_PROCESSED_DIR", tmp_path):
        with pytest.raises(FileNotFoundError):
            training_data.load_training_set([2023, 2024, 2025])


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
