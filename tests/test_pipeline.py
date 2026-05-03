"""Tests for pipeline.run()."""
import pickle
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


def _make_mock_clf():
    clf = MagicMock()
    clf.feature_names_in_ = np.array(["home_bat_avg", "away_bat_avg"])
    clf.predict_proba.return_value = np.array([[0.4, 0.6]])
    return clf


def _write_pkl(path: Path, clf) -> None:
    with open(path, "wb") as f:
        pickle.dump(clf, f)


def test_run_returns_edges(tmp_path):
    """pipeline.run() returns the edges DataFrame produced by find_edges."""
    from mlb_edge_finder import pipeline

    # Create a dummy pkl file so the glob finds it (load_model is patched below)
    model_date = date(2025, 9, 28)
    (tmp_path / f"xgb_{model_date}.pkl").touch()

    clf = _make_mock_clf()
    features_df = pd.DataFrame([{
        "game_id": "game_1",
        "home_team": "New York Yankees",
        "away_team": "Boston Red Sox",
        "home_odds_american": 130,
        "away_odds_american": 200,
        "commence_time": "2025-07-01T18:00:00Z",
        "home_bat_avg": 0.260,
        "away_bat_avg": 0.255,
    }])
    expected_edges = pd.DataFrame([{
        "game_id": "game_1", "home_team": "New York Yankees",
        "away_team": "Boston Red Sox", "bet_side": "home",
        "american_odds": 130, "model_prob": 0.6, "ev": 0.38,
    }])

    with patch("mlb_edge_finder.pipeline.odds_ingestion.fetch_odds", return_value=pd.DataFrame()), \
         patch("mlb_edge_finder.pipeline.stats_ingestion.fetch_stats", return_value=pd.DataFrame()), \
         patch("mlb_edge_finder.pipeline.features.build_features", return_value=features_df), \
         patch("mlb_edge_finder.pipeline.config.MODELS_DIR", tmp_path), \
         patch("mlb_edge_finder.pipeline.model.load_model", return_value=clf), \
         patch("mlb_edge_finder.pipeline.edge_finder.find_edges", return_value=expected_edges) as mock_edges:
        result = pipeline.run(date(2025, 7, 1))

    mock_edges.assert_called_once()
    assert len(result) == 1


def test_run_defaults_to_today():
    """pipeline.run() defaults game_date to date.today() when not provided."""
    from mlb_edge_finder import pipeline
    from datetime import date as date_cls

    captured = {}

    def fake_fetch_odds(game_date):
        captured["game_date"] = game_date
        return pd.DataFrame()

    with patch("mlb_edge_finder.pipeline.odds_ingestion.fetch_odds", side_effect=fake_fetch_odds), \
         patch("mlb_edge_finder.pipeline.stats_ingestion.fetch_stats", return_value=pd.DataFrame()), \
         patch("mlb_edge_finder.pipeline.features.build_features", return_value=pd.DataFrame()), \
         patch("mlb_edge_finder.pipeline.config.MODELS_DIR") as mock_dir, \
         patch("mlb_edge_finder.pipeline.edge_finder.find_edges", return_value=pd.DataFrame()):
        mock_dir.glob.return_value = []
        with pytest.raises(FileNotFoundError):
            pipeline.run()

    assert captured["game_date"] == date_cls.today()


def test_run_raises_when_no_models(tmp_path):
    """pipeline.run() raises FileNotFoundError when MODELS_DIR has no pkl files."""
    from mlb_edge_finder import pipeline

    with patch("mlb_edge_finder.pipeline.odds_ingestion.fetch_odds", return_value=pd.DataFrame()), \
         patch("mlb_edge_finder.pipeline.stats_ingestion.fetch_stats", return_value=pd.DataFrame()), \
         patch("mlb_edge_finder.pipeline.pitcher_ingestion.fetch_pitcher_stats",
               return_value=pd.DataFrame()), \
         patch("mlb_edge_finder.pipeline.features.build_features", return_value=pd.DataFrame()), \
         patch("mlb_edge_finder.pipeline.config.MODELS_DIR", tmp_path):
        with pytest.raises(FileNotFoundError, match="No trained models"):
            pipeline.run(date(2025, 7, 1))


def test_pipeline_calls_fetch_pitcher_stats_before_build_features():
    from mlb_edge_finder import pipeline
    from unittest.mock import call

    mock_clf = MagicMock()
    mock_clf.feature_names_in_ = []
    mock_edges = pd.DataFrame()

    with patch("mlb_edge_finder.pipeline.odds_ingestion.fetch_odds"), \
         patch("mlb_edge_finder.pipeline.stats_ingestion.fetch_stats"), \
         patch("mlb_edge_finder.pipeline.pitcher_ingestion.fetch_pitcher_stats") as mock_pitcher, \
         patch("mlb_edge_finder.pipeline.features.build_features", return_value=pd.DataFrame()), \
         patch("mlb_edge_finder.pipeline.model.load_model", return_value=mock_clf), \
         patch("mlb_edge_finder.pipeline.edge_finder.find_edges", return_value=mock_edges), \
         patch("mlb_edge_finder.pipeline.config.MODELS_DIR") as mock_models_dir:
        mock_models_dir.glob.return_value = [
            Path("models/xgb_2025-01-01.pkl")
        ]
        pipeline.run(date(2025, 4, 22))

    mock_pitcher.assert_called_once_with(date(2025, 4, 22))
