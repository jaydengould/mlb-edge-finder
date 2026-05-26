"""Refresh current-season historical data and conditionally retrain the model."""
import logging
from datetime import date

import pandas as pd

from mlb_edge_finder import config, model
from mlb_edge_finder.historical_ingestion import fetch_historical
from mlb_edge_finder.training_data import build_training_set

logger = logging.getLogger(__name__)

_TRAINING_SEASONS = [2019, 2021, 2022, 2023, 2024, 2025, 2026]


def refresh_historical(season: int) -> pd.DataFrame:
    """Force-fetch the latest completed games for the given season.

    Always bypasses the local cache to ensure today's completed games
    are included. Overwrites data/raw/historical_YYYY.csv.

    Args:
        season: The season year to refresh (e.g. 2026).

    Returns:
        DataFrame of all completed regular-season games for the season.
    """
    return fetch_historical(season, force=True)


def games_since_last_train(historical_df: pd.DataFrame, last_train_date: date) -> int:
    """Count completed games that occurred after last_train_date.

    Args:
        historical_df: Output of refresh_historical() or fetch_historical().
        last_train_date: The date of the most recently saved model.

    Returns:
        Number of rows in historical_df with game_date > last_train_date.
    """
    game_dates = pd.to_datetime(historical_df["game_date"]).dt.date
    return int((game_dates > last_train_date).sum())
