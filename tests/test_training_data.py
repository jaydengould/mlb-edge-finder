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
    expected_abbrs = {
        "ARI", "ATL", "BAL", "BOS", "CHC", "CWS", "CIN", "CLE", "COL", "DET",
        "HOU", "KC", "LAA", "LAD", "MIA", "MIL", "MIN", "NYM", "NYY", "ATH",
        "PHI", "PIT", "SD", "SF", "SEA", "STL", "TB", "TEX", "TOR", "WSH",
    }
    assert expected_abbrs == set(HISTORICAL_NAME_TO_ABBR.values())


def test_historical_name_to_abbr_maps_oakland():
    from mlb_edge_finder.training_data import HISTORICAL_NAME_TO_ABBR
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


def test_build_training_set_joins_home_and_away_stats(tmp_path):
    from mlb_edge_finder import training_data
    with patch("mlb_edge_finder.training_data.load_cached_historical", return_value=_make_hist()), \
         patch("mlb_edge_finder.training_data.fetch_stats", return_value=_make_stats()), \
         patch("mlb_edge_finder.training_data.fetch_pitcher_stats",
               return_value=_make_pitcher_stats()), \
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
         patch("mlb_edge_finder.training_data.fetch_pitcher_stats",
               return_value=_make_pitcher_stats()), \
         patch("mlb_edge_finder.training_data.config.DATA_PROCESSED_DIR", tmp_path):
        df = training_data.build_training_set([2024])
    assert "season" in df.columns
    assert df["season"].iloc[0] == 2024


def test_build_training_set_preserves_home_win(tmp_path):
    from mlb_edge_finder import training_data
    with patch("mlb_edge_finder.training_data.load_cached_historical", return_value=_make_hist()), \
         patch("mlb_edge_finder.training_data.fetch_stats", return_value=_make_stats()), \
         patch("mlb_edge_finder.training_data.fetch_pitcher_stats",
               return_value=_make_pitcher_stats()), \
         patch("mlb_edge_finder.training_data.config.DATA_PROCESSED_DIR", tmp_path):
        df = training_data.build_training_set([2024])
    assert "home_win" in df.columns
    assert df["home_win"].iloc[0] == 1


def test_build_training_set_keeps_name_and_abbr_columns(tmp_path):
    from mlb_edge_finder import training_data
    with patch("mlb_edge_finder.training_data.load_cached_historical", return_value=_make_hist()), \
         patch("mlb_edge_finder.training_data.fetch_stats", return_value=_make_stats()), \
         patch("mlb_edge_finder.training_data.fetch_pitcher_stats",
               return_value=_make_pitcher_stats()), \
         patch("mlb_edge_finder.training_data.config.DATA_PROCESSED_DIR", tmp_path):
        df = training_data.build_training_set([2024])
    for col in ("home_name", "away_name", "home_abbr", "away_abbr"):
        assert col in df.columns


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
         patch("mlb_edge_finder.training_data.fetch_pitcher_stats",
               return_value=_make_pitcher_stats()), \
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
         patch("mlb_edge_finder.training_data.fetch_pitcher_stats",
               return_value=_make_pitcher_stats()), \
         patch("mlb_edge_finder.training_data.config.DATA_PROCESSED_DIR", tmp_path):
        df = training_data.build_training_set([2024])
    assert len(df) == 1
    assert df["home_abbr"].iloc[0] == "ATH"


def test_build_training_set_cache_first(tmp_path):
    from mlb_edge_finder import training_data
    out_path = tmp_path / "training_2024-2024.csv"
    cached_df = pd.DataFrame([{"game_date": "2024-04-01", "season": 2024, "home_win": 1}])
    cached_df.to_csv(out_path, index=False)
    with patch("mlb_edge_finder.training_data.load_cached_historical") as mock_hist, \
         patch("mlb_edge_finder.training_data.config.DATA_PROCESSED_DIR", tmp_path):
        df = training_data.build_training_set([2024])
    mock_hist.assert_not_called()
    assert len(df) == 1


def test_build_training_set_includes_rolling_cols(tmp_path):
    """build_training_set output includes home_ and away_ rolling stat columns."""
    from mlb_edge_finder import training_data
    with patch("mlb_edge_finder.training_data.load_cached_historical", return_value=_make_hist()), \
         patch("mlb_edge_finder.training_data.fetch_stats", return_value=_make_stats()), \
         patch("mlb_edge_finder.training_data.fetch_pitcher_stats",
               return_value=_make_pitcher_stats()), \
         patch("mlb_edge_finder.training_data.config.DATA_PROCESSED_DIR", tmp_path):
        df = training_data.build_training_set([2024], force=True)
    for col in ("home_rolling_runs_scored", "away_rolling_runs_scored",
                "home_rolling_run_diff", "away_rolling_run_diff"):
        assert col in df.columns, f"Missing column: {col}"


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
         patch("mlb_edge_finder.training_data.fetch_pitcher_stats",
               return_value=_make_pitcher_stats()), \
         patch("mlb_edge_finder.training_data.config.DATA_PROCESSED_DIR", tmp_path):
        df = training_data.build_training_set([2023, 2024])

    assert len(df) == 2
    assert set(df["season"]) == {2023, 2024}
    assert (tmp_path / "training_2023-2024.csv").exists()


# --- Pitcher join tests ---

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
