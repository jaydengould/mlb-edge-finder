"""Fetch and cache team and pitcher stats via pybaseball."""
import logging
from datetime import date

import pandas as pd
from pybaseball import team_batting, team_pitching

from mlb_edge_finder import config

logger = logging.getLogger(__name__)

ODDS_NAME_TO_ABBR: dict[str, str] = {
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
}


def _build_stats_df(season: int) -> pd.DataFrame:
    raise NotImplementedError


def fetch_stats(game_date: date, force: bool = False) -> pd.DataFrame:
    raise NotImplementedError


def load_cached_stats(game_date: date) -> pd.DataFrame:
    """Load previously fetched stats from DATA_RAW_DIR/stats_YYYY-MM-DD.csv.

    Args:
        game_date: The date whose cached CSV to load.

    Returns:
        DataFrame with the same schema as fetch_stats().

    Raises:
        FileNotFoundError: If no cached file exists for the given date.
    """
    cache_path = config.DATA_RAW_DIR / f"stats_{game_date}.csv"
    if not cache_path.exists():
        raise FileNotFoundError(f"No cached stats for {game_date}: {cache_path}")
    return pd.read_csv(cache_path)
