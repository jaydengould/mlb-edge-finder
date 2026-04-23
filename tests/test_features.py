"""Smoke tests: features exposes expected public API."""
import inspect
from datetime import date
from unittest.mock import patch

import pandas as pd
import pytest


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

    odds = pd.DataFrame([{
        "game_id": "abc",
        "home_team": "New York Yankees",
        "away_team": "Boston Red Sox",
        "home_odds_american": -150,
        "away_odds_american": 130,
        "commence_time": "2025-04-22T18:05:00Z",
    }])
    stats = pd.DataFrame([
        {"team_abbr": "NYY", "bat_avg": ".260", "obp": ".330", "slg": ".420",
         "ops": ".750", "runs_per_game": 4.8, "era": "3.80", "whip": "1.20",
         "k_per_9": "9.0", "bb_per_9": "3.0", "data_source": "mlb_api"},
        {"team_abbr": "BOS", "bat_avg": ".255", "obp": ".320", "slg": ".410",
         "ops": ".730", "runs_per_game": 4.5, "era": "4.10", "whip": "1.30",
         "k_per_9": "8.5", "bb_per_9": "3.2", "data_source": "mlb_api"},
    ])

    with patch("mlb_edge_finder.features.load_cached_odds", return_value=odds), \
         patch("mlb_edge_finder.features.load_cached_stats", return_value=stats), \
         patch("mlb_edge_finder.features.config.DATA_PROCESSED_DIR") as mock_dir:
        mock_dir.__truediv__ = lambda self, other: __import__("pathlib").Path("/tmp") / other
        df = features.build_features(date(2025, 4, 22))

    assert len(df) == 1
    assert "home_bat_avg" in df.columns
    assert "away_bat_avg" in df.columns
    assert "home_era" in df.columns
    assert "away_era" in df.columns
    assert "data_source" not in df.columns


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
