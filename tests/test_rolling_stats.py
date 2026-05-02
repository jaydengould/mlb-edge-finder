"""Tests for rolling_stats module."""
import pandas as pd
import pytest


def _make_hist(games: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(games)


def test_compute_rolling_stats_shift():
    """First game of season has NaN rolling stats; game 3 reflects only games 1-2."""
    from mlb_edge_finder.rolling_stats import compute_rolling_stats

    hist = _make_hist([
        {"game_date": "2024-04-01", "home_name": "New York Yankees",
         "away_name": "Boston Red Sox", "home_score": 5, "away_score": 3, "home_win": 1},
        {"game_date": "2024-04-03", "home_name": "Boston Red Sox",
         "away_name": "New York Yankees", "home_score": 2, "away_score": 4, "home_win": 0},
        {"game_date": "2024-04-05", "home_name": "New York Yankees",
         "away_name": "Boston Red Sox", "home_score": 7, "away_score": 2, "home_win": 1},
    ])
    result = compute_rolling_stats(hist, window=15)
    nyy = result[result["team_abbr"] == "NYY"].sort_values("game_date").reset_index(drop=True)

    # First game has no prior games → NaN
    assert pd.isna(nyy.iloc[0]["rolling_runs_scored"])

    # Third game (Apr 5) reflects Apr 1 (scored 5) and Apr 3 (scored 4 as away) → avg 4.5
    assert abs(nyy.iloc[2]["rolling_runs_scored"] - 4.5) < 1e-6


def test_compute_rolling_stats_window():
    """Window of 15 limits rolling average to last 15 prior games."""
    from mlb_edge_finder.rolling_stats import compute_rolling_stats

    # 17 games for NYY (always home, away_score=0), runs scored = game number
    games = [
        {"game_date": f"2024-04-{i+1:02d}", "home_name": "New York Yankees",
         "away_name": "Boston Red Sox", "home_score": i + 1, "away_score": 0, "home_win": 1}
        for i in range(17)
    ]
    hist = _make_hist(games)
    result = compute_rolling_stats(hist, window=15)
    nyy = result[result["team_abbr"] == "NYY"].sort_values("game_date").reset_index(drop=True)

    # Game 17 (index 16): shift(1) means rolling = avg of games 1-16 capped at window=15
    # = avg of games 2-16 = (2+3+...+16)/15 = 9.0
    assert abs(nyy.iloc[16]["rolling_runs_scored"] - 9.0) < 1e-6


def test_compute_rolling_stats_min_periods():
    """min_periods=1 allows rolling with fewer than window games without error."""
    from mlb_edge_finder.rolling_stats import compute_rolling_stats

    hist = _make_hist([
        {"game_date": "2024-04-01", "home_name": "New York Yankees",
         "away_name": "Boston Red Sox", "home_score": 5, "away_score": 3, "home_win": 1},
        {"game_date": "2024-04-03", "home_name": "New York Yankees",
         "away_name": "Boston Red Sox", "home_score": 3, "away_score": 2, "home_win": 1},
        {"game_date": "2024-04-05", "home_name": "New York Yankees",
         "away_name": "Boston Red Sox", "home_score": 4, "away_score": 1, "home_win": 1},
    ])
    # window=15 but only 3 games — should not raise
    result = compute_rolling_stats(hist, window=15)
    nyy = result[result["team_abbr"] == "NYY"].sort_values("game_date").reset_index(drop=True)

    # Game 2 (index 1): shift(1) → rolling = avg of game 1 only = 5.0
    assert abs(nyy.iloc[1]["rolling_runs_scored"] - 5.0) < 1e-6


def test_latest_rolling_stats_one_row_per_team():
    """Returns exactly one row per team with no duplicate team_abbr."""
    from mlb_edge_finder.rolling_stats import latest_rolling_stats

    hist = _make_hist([
        {"game_date": "2024-04-01", "home_name": "New York Yankees",
         "away_name": "Boston Red Sox", "home_score": 5, "away_score": 3, "home_win": 1},
        {"game_date": "2024-04-03", "home_name": "New York Yankees",
         "away_name": "Boston Red Sox", "home_score": 3, "away_score": 2, "home_win": 1},
    ])
    result = latest_rolling_stats(hist, window=15)

    assert len(result) == 2
    assert result["team_abbr"].nunique() == 2
    assert set(result["team_abbr"]) == {"NYY", "BOS"}


def test_latest_rolling_stats_includes_last_game():
    """Latest stats include the most recent completed game (no shift applied)."""
    from mlb_edge_finder.rolling_stats import latest_rolling_stats

    hist = _make_hist([
        {"game_date": "2024-04-01", "home_name": "New York Yankees",
         "away_name": "Boston Red Sox", "home_score": 2, "away_score": 1, "home_win": 1},
        {"game_date": "2024-04-03", "home_name": "New York Yankees",
         "away_name": "Boston Red Sox", "home_score": 8, "away_score": 1, "home_win": 1},
    ])
    result = latest_rolling_stats(hist, window=15)
    nyy = result[result["team_abbr"] == "NYY"].iloc[0]

    # No shift: avg of both games = (2 + 8) / 2 = 5.0
    assert abs(nyy["rolling_runs_scored"] - 5.0) < 1e-6
