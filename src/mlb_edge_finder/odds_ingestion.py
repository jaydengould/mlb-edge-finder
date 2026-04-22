"""Fetch and cache MLB moneyline odds from The Odds API."""
import logging
from datetime import date

import pandas as pd
import requests

from mlb_edge_finder import config

logger = logging.getLogger(__name__)


def fetch_odds(game_date: date) -> pd.DataFrame:
    """Fetch MLB moneyline odds from The Odds API for a given date.

    Calls GET /v4/sports/{sport}/odds with market=h2h for the configured
    region. Writes the raw response to DATA_RAW_DIR/odds_YYYY-MM-DD.csv
    before returning.

    Args:
        game_date: The date for which to fetch odds.

    Returns:
        DataFrame with columns: game_id, home_team, away_team,
        home_odds_american, away_odds_american, bookmaker, commence_time.

    Raises:
        RuntimeError: If the API request returns a non-200 status.
    """
    raise NotImplementedError


def load_cached_odds(game_date: date) -> pd.DataFrame:
    """Load previously fetched odds from DATA_RAW_DIR/odds_YYYY-MM-DD.csv.

    Args:
        game_date: The date whose cached CSV to load.

    Returns:
        DataFrame with the same schema as fetch_odds().

    Raises:
        FileNotFoundError: If no cached file exists for the given date.
    """
    raise NotImplementedError
