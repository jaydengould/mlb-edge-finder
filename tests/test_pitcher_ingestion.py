"""Tests for pitcher_ingestion module."""
import inspect
from datetime import date
from unittest.mock import patch

import pandas as pd
import pytest


def _make_stats_response(era="3.50", whip="1.20", k9="9.0", bb9="3.0",
                          ip="150.0", hr=15, bb=45, k=135):
    return {
        "stats": [{
            "splits": [{
                "player": {"id": 592789, "fullName": "Gerrit Cole"},
                "stat": {
                    "era": era, "whip": whip,
                    "strikeoutsPer9Inn": k9, "walksPer9Inn": bb9,
                    "inningsPitched": ip,
                    "homeRuns": hr, "baseOnBalls": bb, "strikeOuts": k,
                },
            }]
        }]
    }


def test_fetch_pitcher_stats_signature():
    from mlb_edge_finder import pitcher_ingestion
    assert callable(pitcher_ingestion.fetch_pitcher_stats)
    sig = inspect.signature(pitcher_ingestion.fetch_pitcher_stats)
    assert "game_date" in sig.parameters
    assert "force" in sig.parameters


def test_load_cached_pitcher_stats_signature():
    from mlb_edge_finder import pitcher_ingestion
    assert callable(pitcher_ingestion.load_cached_pitcher_stats)
    sig = inspect.signature(pitcher_ingestion.load_cached_pitcher_stats)
    assert "game_date" in sig.parameters


def test_load_cached_pitcher_stats_raises_when_missing(tmp_path):
    from mlb_edge_finder import pitcher_ingestion
    with patch("mlb_edge_finder.pitcher_ingestion.config.DATA_RAW_DIR", tmp_path):
        with pytest.raises(FileNotFoundError):
            pitcher_ingestion.load_cached_pitcher_stats(date(2025, 9, 28))


def test_fetch_pitcher_stats_returns_expected_columns(tmp_path):
    from mlb_edge_finder import pitcher_ingestion
    with patch("mlb_edge_finder.pitcher_ingestion.statsapi.get",
               return_value=_make_stats_response()), \
         patch("mlb_edge_finder.pitcher_ingestion.config.DATA_RAW_DIR", tmp_path):
        df = pitcher_ingestion.fetch_pitcher_stats(date(2025, 9, 28), force=True)
    assert set(df.columns) == {
        "pitcher_id", "pitcher_name", "era", "whip",
        "k_per_9", "bb_per_9", "ip", "fip_computed",
    }


def test_fetch_pitcher_stats_computes_fip(tmp_path):
    from mlb_edge_finder import pitcher_ingestion
    # fip = (13*15 + 3*45 - 2*135) / 150.0 + 3.15
    # = (195 + 135 - 270) / 150 + 3.15 = 60/150 + 3.15 = 0.4 + 3.15 = 3.55
    with patch("mlb_edge_finder.pitcher_ingestion.statsapi.get",
               return_value=_make_stats_response(ip="150.0", hr=15, bb=45, k=135)), \
         patch("mlb_edge_finder.pitcher_ingestion.config.DATA_RAW_DIR", tmp_path):
        df = pitcher_ingestion.fetch_pitcher_stats(date(2025, 9, 28), force=True)
    assert abs(df.iloc[0]["fip_computed"] - 3.55) < 0.01


def test_fetch_pitcher_stats_skips_zero_ip(tmp_path):
    from mlb_edge_finder import pitcher_ingestion
    response = {
        "stats": [{
            "splits": [
                {"player": {"id": 1, "fullName": "A"}, "stat": {"inningsPitched": "0", "era": "0", "whip": "0", "strikeoutsPer9Inn": "0", "walksPer9Inn": "0", "homeRuns": 0, "baseOnBalls": 0, "strikeOuts": 0}},
                {"player": {"id": 2, "fullName": "B"}, "stat": {"inningsPitched": "50.1", "era": "3.50", "whip": "1.20", "strikeoutsPer9Inn": "9.0", "walksPer9Inn": "3.0", "homeRuns": 5, "baseOnBalls": 15, "strikeOuts": 50}},
            ]
        }]
    }
    with patch("mlb_edge_finder.pitcher_ingestion.statsapi.get", return_value=response), \
         patch("mlb_edge_finder.pitcher_ingestion.config.DATA_RAW_DIR", tmp_path):
        df = pitcher_ingestion.fetch_pitcher_stats(date(2025, 9, 28), force=True)
    assert len(df) == 1
    assert df.iloc[0]["pitcher_name"] == "B"


def test_fetch_pitcher_stats_cache_first(tmp_path):
    from mlb_edge_finder import pitcher_ingestion
    cached = pd.DataFrame([{
        "pitcher_id": 1, "pitcher_name": "Cached Cole",
        "era": 3.0, "whip": 1.1, "k_per_9": 9.0, "bb_per_9": 2.0,
        "ip": 100.0, "fip_computed": 3.2,
    }])
    cache_path = tmp_path / "pitcher_stats_2025-09-28.csv"
    cached.to_csv(cache_path, index=False)
    with patch("mlb_edge_finder.pitcher_ingestion.statsapi.get") as mock_get, \
         patch("mlb_edge_finder.pitcher_ingestion.config.DATA_RAW_DIR", tmp_path):
        df = pitcher_ingestion.fetch_pitcher_stats(date(2025, 9, 28))
    mock_get.assert_not_called()
    assert df.iloc[0]["pitcher_name"] == "Cached Cole"


def test_fetch_pitcher_stats_raises_on_api_failure(tmp_path):
    from mlb_edge_finder import pitcher_ingestion
    with patch("mlb_edge_finder.pitcher_ingestion.statsapi.get",
               side_effect=Exception("timeout")), \
         patch("mlb_edge_finder.pitcher_ingestion.config.DATA_RAW_DIR", tmp_path):
        with pytest.raises(RuntimeError, match="statsapi failed"):
            pitcher_ingestion.fetch_pitcher_stats(date(2025, 9, 28), force=True)


def test_fetch_pitcher_stats_raises_on_empty_response(tmp_path):
    from mlb_edge_finder import pitcher_ingestion
    with patch("mlb_edge_finder.pitcher_ingestion.statsapi.get",
               return_value={"stats": [{"splits": []}]}), \
         patch("mlb_edge_finder.pitcher_ingestion.config.DATA_RAW_DIR", tmp_path):
        with pytest.raises(RuntimeError, match="no pitcher stats"):
            pitcher_ingestion.fetch_pitcher_stats(date(2025, 9, 28), force=True)


def test_fetch_pitcher_stats_excludes_low_ip_pitchers(tmp_path):
    """Pitchers with ip < MIN_PITCHER_IP are excluded from fetch_pitcher_stats output."""
    from mlb_edge_finder import pitcher_ingestion, config
    response = {
        "stats": [{
            "splits": [
                {
                    "player": {"id": 1, "fullName": "Low IP Pitcher"},
                    "stat": {
                        "inningsPitched": str(config.MIN_PITCHER_IP - 1),
                        "era": "2.50", "whip": "1.10",
                        "strikeoutsPer9Inn": "9.0", "walksPer9Inn": "2.0",
                        "homeRuns": 1, "baseOnBalls": 5, "strikeOuts": 25,
                    },
                },
                {
                    "player": {"id": 2, "fullName": "Qualified Pitcher"},
                    "stat": {
                        "inningsPitched": str(config.MIN_PITCHER_IP),
                        "era": "3.50", "whip": "1.20",
                        "strikeoutsPer9Inn": "9.0", "walksPer9Inn": "3.0",
                        "homeRuns": 3, "baseOnBalls": 10, "strikeOuts": 50,
                    },
                },
            ]
        }]
    }
    with patch("mlb_edge_finder.pitcher_ingestion.statsapi.get", return_value=response), \
         patch("mlb_edge_finder.pitcher_ingestion.config.DATA_RAW_DIR", tmp_path):
        df = pitcher_ingestion.fetch_pitcher_stats(date(2026, 5, 26), force=True)
    assert len(df) == 1
    assert df.iloc[0]["pitcher_name"] == "Qualified Pitcher"


# --- fetch_probable_starters tests ---

def _make_schedule(home_name="New York Yankees", away_name="Boston Red Sox",
                   home_pitcher="Gerrit Cole", away_pitcher="Brayan Bello",
                   game_type="R"):
    return [{
        "game_id": 12345,
        "game_type": game_type,
        "home_name": home_name,
        "away_name": away_name,
        "home_probable_pitcher": home_pitcher,
        "away_probable_pitcher": away_pitcher,
    }]


def test_fetch_probable_starters_signature():
    from mlb_edge_finder import pitcher_ingestion
    assert callable(pitcher_ingestion.fetch_probable_starters)
    sig = inspect.signature(pitcher_ingestion.fetch_probable_starters)
    assert "game_date" in sig.parameters


def test_fetch_probable_starters_returns_expected_columns():
    from mlb_edge_finder import pitcher_ingestion
    with patch("mlb_edge_finder.pitcher_ingestion.statsapi.schedule",
               return_value=_make_schedule()):
        df = pitcher_ingestion.fetch_probable_starters(date(2025, 4, 22))
    assert set(df.columns) == {"home_abbr", "away_abbr", "home_starter_name", "away_starter_name"}


def test_fetch_probable_starters_maps_team_names():
    from mlb_edge_finder import pitcher_ingestion
    with patch("mlb_edge_finder.pitcher_ingestion.statsapi.schedule",
               return_value=_make_schedule()):
        df = pitcher_ingestion.fetch_probable_starters(date(2025, 4, 22))
    assert len(df) == 1
    assert df.iloc[0]["home_abbr"] == "NYY"
    assert df.iloc[0]["away_abbr"] == "BOS"
    assert df.iloc[0]["home_starter_name"] == "Gerrit Cole"
    assert df.iloc[0]["away_starter_name"] == "Brayan Bello"


def test_fetch_probable_starters_none_when_empty_string():
    from mlb_edge_finder import pitcher_ingestion
    with patch("mlb_edge_finder.pitcher_ingestion.statsapi.schedule",
               return_value=_make_schedule(home_pitcher="", away_pitcher="")):
        df = pitcher_ingestion.fetch_probable_starters(date(2025, 4, 22))
    assert pd.isna(df.iloc[0]["home_starter_name"])
    assert pd.isna(df.iloc[0]["away_starter_name"])


def test_fetch_probable_starters_skips_non_regular_season():
    from mlb_edge_finder import pitcher_ingestion
    with patch("mlb_edge_finder.pitcher_ingestion.statsapi.schedule",
               return_value=_make_schedule(game_type="S")):
        df = pitcher_ingestion.fetch_probable_starters(date(2025, 4, 22))
    assert len(df) == 0


def test_fetch_probable_starters_returns_empty_on_no_games():
    from mlb_edge_finder import pitcher_ingestion
    with patch("mlb_edge_finder.pitcher_ingestion.statsapi.schedule", return_value=[]):
        df = pitcher_ingestion.fetch_probable_starters(date(2025, 4, 22))
    assert len(df) == 0
    assert list(df.columns) == ["home_abbr", "away_abbr", "home_starter_name", "away_starter_name"]


def test_fetch_probable_starters_raises_on_api_failure():
    from mlb_edge_finder import pitcher_ingestion
    with patch("mlb_edge_finder.pitcher_ingestion.statsapi.schedule",
               side_effect=Exception("timeout")):
        with pytest.raises(RuntimeError, match="statsapi.schedule failed"):
            pitcher_ingestion.fetch_probable_starters(date(2025, 4, 22))


# --- fetch_pitcher_snapshot tests ---

def test_fetch_pitcher_snapshot_signature():
    from mlb_edge_finder import pitcher_ingestion
    import inspect
    assert callable(pitcher_ingestion.fetch_pitcher_snapshot)
    sig = inspect.signature(pitcher_ingestion.fetch_pitcher_snapshot)
    assert "snapshot_date" in sig.parameters
    assert "force" in sig.parameters


def test_fetch_pitcher_snapshot_writes_to_snapshot_path(tmp_path):
    """fetch_pitcher_snapshot writes to pitcher_snapshot_*.csv, not pitcher_stats_*.csv."""
    from mlb_edge_finder import pitcher_ingestion
    with patch("mlb_edge_finder.pitcher_ingestion.statsapi.get",
               return_value=_make_stats_response(ip="150.0")), \
         patch("mlb_edge_finder.pitcher_ingestion.config.DATA_RAW_DIR", tmp_path):
        pitcher_ingestion.fetch_pitcher_snapshot(date(2026, 4, 30), force=True)
    assert (tmp_path / "pitcher_snapshot_2026-04-30.csv").exists()
    assert not (tmp_path / "pitcher_stats_2026-04-30.csv").exists()


def test_fetch_pitcher_snapshot_excludes_low_ip(tmp_path):
    """fetch_pitcher_snapshot excludes pitchers below MIN_PITCHER_IP."""
    from mlb_edge_finder import pitcher_ingestion, config
    response = {
        "stats": [{
            "splits": [
                {
                    "player": {"id": 1, "fullName": "Low IP"},
                    "stat": {
                        "inningsPitched": str(config.MIN_PITCHER_IP - 1),
                        "era": "2.00", "whip": "1.00",
                        "strikeoutsPer9Inn": "10.0", "walksPer9Inn": "2.0",
                        "homeRuns": 0, "baseOnBalls": 3, "strikeOuts": 20,
                    },
                },
                {
                    "player": {"id": 2, "fullName": "Qualified"},
                    "stat": {
                        "inningsPitched": str(config.MIN_PITCHER_IP + 10),
                        "era": "3.50", "whip": "1.20",
                        "strikeoutsPer9Inn": "9.0", "walksPer9Inn": "3.0",
                        "homeRuns": 3, "baseOnBalls": 12, "strikeOuts": 45,
                    },
                },
            ]
        }]
    }
    with patch("mlb_edge_finder.pitcher_ingestion.statsapi.get", return_value=response), \
         patch("mlb_edge_finder.pitcher_ingestion.config.DATA_RAW_DIR", tmp_path):
        df = pitcher_ingestion.fetch_pitcher_snapshot(date(2026, 4, 30), force=True)
    assert len(df) == 1
    assert df.iloc[0]["pitcher_name"] == "Qualified"


def test_fetch_pitcher_snapshot_cache_first(tmp_path):
    """fetch_pitcher_snapshot returns cached file without calling statsapi."""
    from mlb_edge_finder import pitcher_ingestion
    cached = pd.DataFrame([{
        "pitcher_id": 99, "pitcher_name": "Cached Ace",
        "era": 2.5, "whip": 1.0, "k_per_9": 11.0, "bb_per_9": 2.0,
        "ip": 50.0, "fip_computed": 2.8,
    }])
    (tmp_path / "pitcher_snapshot_2026-04-30.csv").write_text(cached.to_csv(index=False))
    with patch("mlb_edge_finder.pitcher_ingestion.statsapi.get") as mock_get, \
         patch("mlb_edge_finder.pitcher_ingestion.config.DATA_RAW_DIR", tmp_path):
        df = pitcher_ingestion.fetch_pitcher_snapshot(date(2026, 4, 30))
    mock_get.assert_not_called()
    assert df.iloc[0]["pitcher_name"] == "Cached Ace"


def test_fetch_pitcher_snapshot_falls_back_to_season_stats_when_no_splits(tmp_path):
    """When byDateRange returns empty splits, falls back to full-season stats."""
    from mlb_edge_finder import pitcher_ingestion
    empty_response = {"stats": [{"splits": []}]}
    full_season_response = _make_stats_response(ip="150.0")
    responses = [empty_response, full_season_response]
    with patch("mlb_edge_finder.pitcher_ingestion.statsapi.get", side_effect=responses), \
         patch("mlb_edge_finder.pitcher_ingestion.config.DATA_RAW_DIR", tmp_path):
        df = pitcher_ingestion.fetch_pitcher_snapshot(date(2024, 4, 30), force=True)
    assert len(df) == 1
    assert df.iloc[0]["pitcher_name"] == "Gerrit Cole"
