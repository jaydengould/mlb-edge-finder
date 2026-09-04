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


def test_build_features_signature():
    """build_features should accept game_date."""
    from mlb_win_probability import features
    assert callable(features.build_features)
    sig = inspect.signature(features.build_features)
    assert "game_date" in sig.parameters


def test_load_features_signature():
    """load_features should accept game_date."""
    from mlb_win_probability import features
    assert callable(features.load_features)
    sig = inspect.signature(features.load_features)
    assert "game_date" in sig.parameters


def test_build_features_joins_home_and_away():
    """build_features should produce home_ and away_ prefixed stat columns."""
    from mlb_win_probability import features

    with patch("mlb_win_probability.features.load_cached_odds", return_value=_make_odds()), \
         patch("mlb_win_probability.features.load_cached_stats", return_value=_make_stats()), \
         patch("mlb_win_probability.features.fetch_historical", return_value=_make_hist()), \
         patch("mlb_win_probability.features.fetch_probable_starters",
               return_value=_make_probable_starters()), \
         patch("mlb_win_probability.features.load_cached_pitcher_stats",
               return_value=_make_pitcher_stats()), \
         patch("mlb_win_probability.features.config.DATA_PROCESSED_DIR") as mock_dir:
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
    from mlb_win_probability import features

    with patch("mlb_win_probability.features.load_cached_odds", return_value=_make_odds()), \
         patch("mlb_win_probability.features.load_cached_stats", return_value=_make_stats()), \
         patch("mlb_win_probability.features.fetch_historical", return_value=_make_hist()), \
         patch("mlb_win_probability.features.fetch_probable_starters",
               return_value=_make_probable_starters()), \
         patch("mlb_win_probability.features.load_cached_pitcher_stats",
               return_value=_make_pitcher_stats()), \
         patch("mlb_win_probability.features.config.DATA_PROCESSED_DIR") as mock_dir:
        mock_dir.__truediv__ = lambda self, other: __import__("pathlib").Path("/tmp") / other
        df = features.build_features(date(2025, 4, 22))

    for col in ("home_rolling_runs_scored", "away_rolling_runs_scored",
                "home_rolling_run_diff", "away_rolling_run_diff"):
        assert col in df.columns, f"Missing column: {col}"

    assert not df["home_rolling_runs_scored"].isna().any()


def test_build_features_raises_on_missing_odds(tmp_path):
    """build_features raises RuntimeError when the odds cache is absent."""
    from mlb_win_probability import features
    with patch("mlb_win_probability.features.config.DATA_RAW_DIR", tmp_path), \
         patch("mlb_win_probability.features.config.DATA_PROCESSED_DIR", tmp_path):
        with pytest.raises(RuntimeError, match="odds"):
            features.build_features(date(2025, 4, 22))


def test_load_features_raises_when_missing(tmp_path):
    """load_features raises FileNotFoundError when file is absent."""
    from mlb_win_probability import features
    with patch("mlb_win_probability.features.config.DATA_PROCESSED_DIR", tmp_path):
        with pytest.raises(FileNotFoundError):
            features.load_features(date(2025, 4, 22))


# --- Pitcher join tests ---

def test_build_features_includes_pitcher_sp_cols():
    from mlb_win_probability import features
    with patch("mlb_win_probability.features.load_cached_odds", return_value=_make_odds()), \
         patch("mlb_win_probability.features.load_cached_stats", return_value=_make_stats()), \
         patch("mlb_win_probability.features.fetch_historical", return_value=_make_hist()), \
         patch("mlb_win_probability.features.fetch_probable_starters",
               return_value=_make_probable_starters()), \
         patch("mlb_win_probability.features.load_cached_pitcher_stats",
               return_value=_make_pitcher_stats()), \
         patch("mlb_win_probability.features.config.DATA_PROCESSED_DIR") as mock_dir:
        mock_dir.__truediv__ = lambda self, other: __import__("pathlib").Path("/tmp") / other
        df = features.build_features(date(2025, 4, 22))
    for col in ("home_sp_era", "away_sp_era", "home_sp_fip_computed", "away_sp_fip_computed"):
        assert col in df.columns, f"Missing column: {col}"


def test_build_features_pitcher_join_values_correct():
    from mlb_win_probability import features
    with patch("mlb_win_probability.features.load_cached_odds", return_value=_make_odds()), \
         patch("mlb_win_probability.features.load_cached_stats", return_value=_make_stats()), \
         patch("mlb_win_probability.features.fetch_historical", return_value=_make_hist()), \
         patch("mlb_win_probability.features.fetch_probable_starters",
               return_value=_make_probable_starters()), \
         patch("mlb_win_probability.features.load_cached_pitcher_stats",
               return_value=_make_pitcher_stats()), \
         patch("mlb_win_probability.features.config.DATA_PROCESSED_DIR") as mock_dir:
        mock_dir.__truediv__ = lambda self, other: __import__("pathlib").Path("/tmp") / other
        df = features.build_features(date(2025, 4, 22))
    assert abs(df.iloc[0]["home_sp_era"] - 3.20) < 0.01
    assert abs(df.iloc[0]["away_sp_era"] - 4.10) < 0.01


def test_build_features_pitcher_nan_when_no_probable_starter():
    from mlb_win_probability import features
    no_starters = pd.DataFrame([{
        "home_abbr": "NYY", "away_abbr": "BOS",
        "home_starter_name": None, "away_starter_name": None,
    }])
    with patch("mlb_win_probability.features.load_cached_odds", return_value=_make_odds()), \
         patch("mlb_win_probability.features.load_cached_stats", return_value=_make_stats()), \
         patch("mlb_win_probability.features.fetch_historical", return_value=_make_hist()), \
         patch("mlb_win_probability.features.fetch_probable_starters", return_value=no_starters), \
         patch("mlb_win_probability.features.load_cached_pitcher_stats",
               return_value=_make_pitcher_stats()), \
         patch("mlb_win_probability.features.config.DATA_PROCESSED_DIR") as mock_dir:
        mock_dir.__truediv__ = lambda self, other: __import__("pathlib").Path("/tmp") / other
        df = features.build_features(date(2025, 4, 22))
    assert pd.isna(df.iloc[0]["home_sp_era"])
    assert pd.isna(df.iloc[0]["away_sp_era"])


def test_build_features_raises_on_missing_pitcher_stats():
    from mlb_win_probability import features
    with patch("mlb_win_probability.features.load_cached_odds", return_value=_make_odds()), \
         patch("mlb_win_probability.features.load_cached_stats", return_value=_make_stats()), \
         patch("mlb_win_probability.features.fetch_historical", return_value=_make_hist()), \
         patch("mlb_win_probability.features.fetch_probable_starters",
               return_value=_make_probable_starters()), \
         patch("mlb_win_probability.features.load_cached_pitcher_stats",
               side_effect=FileNotFoundError("no file")), \
         patch("mlb_win_probability.features.config.DATA_PROCESSED_DIR") as mock_dir:
        mock_dir.__truediv__ = lambda self, other: __import__("pathlib").Path("/tmp") / other
        with pytest.raises(RuntimeError, match="fetch_pitcher_stats"):
            features.build_features(date(2025, 4, 22))
