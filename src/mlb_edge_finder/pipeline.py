"""Orchestrate all pipeline stages from odds ingestion to edge output."""
import logging
from datetime import date

import pandas as pd

from mlb_edge_finder import config, edge_finder, features, model, odds_ingestion, stats_ingestion

logger = logging.getLogger(__name__)


def run(game_date: date | None = None) -> pd.DataFrame:
    """Run the full MLB edge-finding pipeline for a single game date.

    Stages (in order):
      1. Fetch or load moneyline odds for game_date.
      2. Fetch or load team/pitcher stats up to game_date.
      3. Build feature DataFrame from odds + stats.
      4. Load the most recently saved model from MODELS_DIR.
      5. Run edge_finder.find_edges() and return the result.

    Args:
        game_date: Date to run the pipeline for. Defaults to today.

    Returns:
        DataFrame of flagged edges (may be empty if none found).
        Same schema as edge_finder.find_edges().
    """
    raise NotImplementedError
