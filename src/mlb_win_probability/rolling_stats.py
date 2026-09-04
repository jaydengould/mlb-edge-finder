"""Compute rolling per-team stats from historical game results."""
import logging

import pandas as pd

logger = logging.getLogger(__name__)

# statsapi full team names → current franchise abbreviations.
# Moved here from training_data so both training and inference paths
# share the same mapping without circular imports.
HISTORICAL_NAME_TO_ABBR: dict[str, str] = {
    "Arizona Diamondbacks": "ARI",
    "Atlanta Braves": "ATL",
    "Baltimore Orioles": "BAL",
    "Boston Red Sox": "BOS",
    "Chicago Cubs": "CHC",
    "Chicago White Sox": "CWS",
    "Cincinnati Reds": "CIN",
    "Cleveland Guardians": "CLE",
    "Colorado Rockies": "COL",
    "Detroit Tigers": "DET",
    "Houston Astros": "HOU",
    "Kansas City Royals": "KC",
    "Los Angeles Angels": "LAA",
    "Los Angeles Dodgers": "LAD",
    "Miami Marlins": "MIA",
    "Milwaukee Brewers": "MIL",
    "Minnesota Twins": "MIN",
    "New York Mets": "NYM",
    "New York Yankees": "NYY",
    "Oakland Athletics": "ATH",
    "Athletics": "ATH",
    "Philadelphia Phillies": "PHI",
    "Pittsburgh Pirates": "PIT",
    "San Diego Padres": "SD",
    "San Francisco Giants": "SF",
    "Seattle Mariners": "SEA",
    "St. Louis Cardinals": "STL",
    "Tampa Bay Rays": "TB",
    "Texas Rangers": "TEX",
    "Toronto Blue Jays": "TOR",
    "Washington Nationals": "WSH",
    # Legacy names for pre-rename seasons
    "Cleveland Indians": "CLE",
    "Florida Marlins": "MIA",
    "Tampa Bay Devil Rays": "TB",
    "Montreal Expos": "WSH",
}

_ROLLING_COLS = [
    "rolling_runs_scored",
    "rolling_runs_allowed",
    "rolling_win_pct",
    "rolling_run_diff",
]


def _reshape_to_team_games(historical_df: pd.DataFrame) -> pd.DataFrame:
    """Reshape from one-row-per-game to one-row-per-team-game.

    Each game becomes two rows — one for the home team and one for the away team.
    Maps full statsapi team names to abbreviations via HISTORICAL_NAME_TO_ABBR.
    Drops rows with unmapped team names and logs a warning.
    """
    home = pd.DataFrame({
        "team_abbr": historical_df["home_name"].map(HISTORICAL_NAME_TO_ABBR),
        "game_date": historical_df["game_date"].values,
        "runs_scored": historical_df["home_score"].astype(float).values,
        "runs_allowed": historical_df["away_score"].astype(float).values,
        "win": historical_df["home_win"].astype(float).values,
    })
    away = pd.DataFrame({
        "team_abbr": historical_df["away_name"].map(HISTORICAL_NAME_TO_ABBR),
        "game_date": historical_df["game_date"].values,
        "runs_scored": historical_df["away_score"].astype(float).values,
        "runs_allowed": historical_df["home_score"].astype(float).values,
        "win": (1 - historical_df["home_win"]).astype(float).values,
    })
    long_df = pd.concat([home, away], ignore_index=True)
    n_unmapped = long_df["team_abbr"].isna().sum()
    if n_unmapped:
        logger.warning("Rolling stats: dropped %d rows with unmapped team names", n_unmapped)
    long_df = long_df.dropna(subset=["team_abbr"])
    return long_df.sort_values(["team_abbr", "game_date"]).reset_index(drop=True)


def _roll(long_df: pd.DataFrame, window: int, shift: bool) -> pd.DataFrame:
    """Apply rolling aggregation per team, optionally shifting by 1."""
    df = long_df.copy()
    df["run_diff"] = df["runs_scored"] - df["runs_allowed"]
    result = df[["team_abbr", "game_date"]].copy()
    for raw_col, out_col in [
        ("runs_scored", "rolling_runs_scored"),
        ("runs_allowed", "rolling_runs_allowed"),
        ("win", "rolling_win_pct"),
        ("run_diff", "rolling_run_diff"),
    ]:
        result[out_col] = (
            df.groupby("team_abbr")[raw_col]
            .transform(lambda x: x.rolling(window, min_periods=1).mean())
        )
    if shift:
        for col in _ROLLING_COLS:
            result[col] = result.groupby("team_abbr")[col].transform(lambda x: x.shift(1))
    return result


def compute_rolling_stats(historical_df: pd.DataFrame, window: int = 15) -> pd.DataFrame:
    """Compute per-game pregame rolling stats for training.

    Rolling stats for each game reflect up to `window` completed games BEFORE
    that game date. The current game is excluded via shift(1). First game of
    the season per team has NaN rolling stats — XGBoost handles NaN natively.

    Args:
        historical_df: Output of fetch_historical() or load_cached_historical().
            Columns: game_date, home_name, away_name, home_score, away_score, home_win.
        window: Number of prior games to average over. Default 15.

    Returns:
        DataFrame with columns: team_abbr, game_date, rolling_runs_scored,
        rolling_runs_allowed, rolling_win_pct, rolling_run_diff.
    """
    long_df = _reshape_to_team_games(historical_df)
    return _roll(long_df, window, shift=True)


def latest_rolling_stats(historical_df: pd.DataFrame, window: int = 15) -> pd.DataFrame:
    """Compute current rolling stats per team for inference.

    All completed games are included (no shift). Returns one row per team
    reflecting their current form going into today's games.

    Args:
        historical_df: Output of fetch_historical() for the current season.
            Columns: game_date, home_name, away_name, home_score, away_score, home_win.
        window: Number of prior games to average over. Default 15.

    Returns:
        DataFrame with one row per team_abbr. Columns: team_abbr,
        rolling_runs_scored, rolling_runs_allowed, rolling_win_pct, rolling_run_diff.
    """
    long_df = _reshape_to_team_games(historical_df)
    rolled = _roll(long_df, window, shift=False)
    return rolled.groupby("team_abbr").last().reset_index()
