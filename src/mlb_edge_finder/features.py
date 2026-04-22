"""Merge odds and stats into a model-ready feature DataFrame."""
import logging
from datetime import date

import pandas as pd

from mlb_edge_finder import config

logger = logging.getLogger(__name__)


def build_features(odds_df: pd.DataFrame, stats_df: pd.DataFrame) -> pd.DataFrame:
    """Join odds and stats on team name and engineer model features.

    Computes implied probability from American odds, merges team-level
    stats for both home and away sides, and writes the result to
    DATA_PROCESSED_DIR/features_YYYY-MM-DD.csv.

    Args:
        odds_df: Output of odds_ingestion.fetch_odds() or load_cached_odds().
        stats_df: Output of stats_ingestion.fetch_stats() or load_cached_stats().

    Returns:
        DataFrame with one row per game and feature columns ready for
        XGBoost training or inference. Includes implied_prob_home,
        implied_prob_away, and all engineered stat differentials.

    Raises:
        ValueError: If odds_df or stats_df are empty.
    """
    raise NotImplementedError


def load_features(game_date: date) -> pd.DataFrame:
    """Load a previously built feature DataFrame from DATA_PROCESSED_DIR.

    Args:
        game_date: The date whose features CSV to load.

    Returns:
        DataFrame with the same schema as build_features().

    Raises:
        FileNotFoundError: If no features file exists for the given date.
    """
    raise NotImplementedError
