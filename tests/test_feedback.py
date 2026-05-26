# tests/test_feedback.py
"""Tests for feedback module."""
from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


def test_retrain_threshold_constant():
    from mlb_edge_finder import config
    assert config.RETRAIN_THRESHOLD == 15


def _make_historical_df(game_dates: list[str]) -> pd.DataFrame:
    return pd.DataFrame({
        "game_date": game_dates,
        "home_name": "Yankees",
        "away_name": "Red Sox",
        "home_score": 5,
        "away_score": 3,
        "home_win": 1,
        "home_starter_name": None,
        "away_starter_name": None,
    })


def test_refresh_historical_calls_fetch_with_force():
    from mlb_edge_finder import feedback
    mock_df = _make_historical_df(["2026-04-01"])
    with patch("mlb_edge_finder.feedback.fetch_historical", return_value=mock_df) as mock_fetch:
        result = feedback.refresh_historical(2026)
    mock_fetch.assert_called_once_with(2026, force=True)
    assert len(result) == 1


def test_games_since_last_train_counts_correctly():
    from mlb_edge_finder import feedback
    cutoff = date(2026, 4, 10)
    df = _make_historical_df([
        "2026-04-08", "2026-04-09",
        "2026-04-11", "2026-04-12", "2026-04-13",
    ])
    assert feedback.games_since_last_train(df, cutoff) == 3


def test_games_since_last_train_zero_when_all_before():
    from mlb_edge_finder import feedback
    cutoff = date(2026, 5, 1)
    df = _make_historical_df(["2026-04-01", "2026-04-02", "2026-04-03"])
    assert feedback.games_since_last_train(df, cutoff) == 0


def test_games_since_last_train_excludes_exact_cutoff_date():
    from mlb_edge_finder import feedback
    cutoff = date(2026, 4, 10)
    # game on exactly the cutoff date should NOT count
    df = _make_historical_df(["2026-04-10", "2026-04-11"])
    assert feedback.games_since_last_train(df, cutoff) == 1


def test_run_feedback_loop_no_retrain(tmp_path):
    """Fewer than 15 new games — should skip retrain."""
    from mlb_edge_finder import feedback

    model_date = date(2026, 4, 1)
    (tmp_path / f"xgb_{model_date}.pkl").touch()

    # 14 games after model_date — one short of threshold
    game_dates = [str(model_date + timedelta(days=i + 1)) for i in range(14)]
    hist_df = _make_historical_df(game_dates)

    with patch("mlb_edge_finder.feedback.refresh_historical", return_value=hist_df), \
         patch("mlb_edge_finder.feedback.config.MODELS_DIR", tmp_path), \
         patch("mlb_edge_finder.feedback.config.RETRAIN_THRESHOLD", 15), \
         patch("mlb_edge_finder.feedback.model.save_model") as mock_save:
        result = feedback.run_feedback_loop(2026)

    assert result["retrained"] is False
    assert result["new_games"] == 14
    assert result["season"] == 2026
    mock_save.assert_not_called()


def test_run_feedback_loop_retrain(tmp_path):
    """Exactly 15 new games — should retrain and save model."""
    from mlb_edge_finder import feedback

    model_date = date(2026, 4, 1)
    (tmp_path / f"xgb_{model_date}.pkl").touch()

    # 15 games after model_date — hits threshold
    game_dates = [str(model_date + timedelta(days=i + 1)) for i in range(15)]
    hist_df = _make_historical_df(game_dates)

    mock_clf = MagicMock()
    mock_training_df = pd.DataFrame({"home_win": [1, 0]})

    with patch("mlb_edge_finder.feedback.refresh_historical", return_value=hist_df), \
         patch("mlb_edge_finder.feedback.config.MODELS_DIR", tmp_path), \
         patch("mlb_edge_finder.feedback.config.RETRAIN_THRESHOLD", 15), \
         patch("mlb_edge_finder.feedback.build_training_set", return_value=mock_training_df) as mock_build, \
         patch("mlb_edge_finder.feedback.model.train", return_value=(mock_clf, MagicMock(), MagicMock(), MagicMock(), MagicMock())), \
         patch("mlb_edge_finder.feedback.model.calibrate", return_value=mock_clf), \
         patch("mlb_edge_finder.feedback.model.evaluate", return_value={}), \
         patch("mlb_edge_finder.feedback.model.save_model") as mock_save:
        result = feedback.run_feedback_loop(2026)

    assert result["retrained"] is True
    assert result["new_games"] == 15
    mock_build.assert_called_once_with([2019, 2021, 2022, 2023, 2024, 2025, 2026], force=True)
    mock_save.assert_called_once()


def test_run_feedback_loop_no_existing_model(tmp_path):
    """No model on disk — should always retrain regardless of game count."""
    from mlb_edge_finder import feedback

    # Only 2 games — well below threshold, but no model exists
    hist_df = _make_historical_df(["2026-04-01", "2026-04-02"])

    mock_clf = MagicMock()

    with patch("mlb_edge_finder.feedback.refresh_historical", return_value=hist_df), \
         patch("mlb_edge_finder.feedback.config.MODELS_DIR", tmp_path), \
         patch("mlb_edge_finder.feedback.config.RETRAIN_THRESHOLD", 15), \
         patch("mlb_edge_finder.feedback.build_training_set", return_value=pd.DataFrame({"home_win": [1]})), \
         patch("mlb_edge_finder.feedback.model.train", return_value=(mock_clf, MagicMock(), MagicMock(), MagicMock(), MagicMock())), \
         patch("mlb_edge_finder.feedback.model.calibrate", return_value=mock_clf), \
         patch("mlb_edge_finder.feedback.model.evaluate", return_value={}), \
         patch("mlb_edge_finder.feedback.model.save_model") as mock_save:
        result = feedback.run_feedback_loop(2026)

    assert result["retrained"] is True
    mock_save.assert_called_once()
