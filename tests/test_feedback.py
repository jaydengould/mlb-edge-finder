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
