"""Fetch and cache team and pitcher stats via pybaseball."""
import logging
from datetime import date

import pandas as pd

from mlb_edge_finder import config

logger = logging.getLogger(__name__)


def fetch_stats(start_date: date, end_date: date) -> pd.DataFrame:
    """Fetch team batting and starting pitcher stats for a date range.

    Uses pybaseball.team_batting() and pybaseball.pitching_stats() to pull
    season-to-date aggregates. Writes result to
    DATA_RAW_DIR/stats_YYYY-MM-DD.csv (keyed by end_date).

    Args:
        start_date: First date of the window (inclusive).
        end_date: Last date of the window (inclusive).

    Returns:
        DataFrame with columns: team, era, whip, batting_avg, ops,
        runs_per_game, home_away (placeholder columns — finalize during
        feature engineering design).

    Raises:
        RuntimeError: If pybaseball fails to return data.
    """
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
    raise NotImplementedError
