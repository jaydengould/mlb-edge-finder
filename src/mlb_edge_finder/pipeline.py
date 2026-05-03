"""Orchestrate all pipeline stages from odds ingestion to edge output."""
import logging
from datetime import date

import pandas as pd

from mlb_edge_finder import (
    config, edge_finder, features, model,
    odds_ingestion, pitcher_ingestion, stats_ingestion,
)

logger = logging.getLogger(__name__)


def run(game_date: date | None = None) -> pd.DataFrame:
    """Run the full MLB edge-finding pipeline for a single game date.

    Stages (in order):
      1. Fetch or load moneyline odds for game_date.
      2. Fetch or load team stats up to game_date.
      3. Build feature DataFrame from odds + stats.
      4. Auto-discover and load the most recently saved model from MODELS_DIR.
      5. Run edge_finder.find_edges() and return the result.

    Args:
        game_date: Date to run the pipeline for. Defaults to today.

    Returns:
        DataFrame of flagged edges (may be empty if none found).
        Same schema as edge_finder.find_edges().

    Raises:
        FileNotFoundError: If no trained models exist in MODELS_DIR.
    """
    if game_date is None:
        game_date = date.today()

    logger.info("Running pipeline for %s", game_date)

    odds_ingestion.fetch_odds(game_date)
    stats_ingestion.fetch_stats(game_date)
    pitcher_ingestion.fetch_pitcher_stats(game_date)
    features_df = features.build_features(game_date)

    pkls = sorted(config.MODELS_DIR.glob("xgb_*.pkl"))
    if not pkls:
        raise FileNotFoundError(
            "No trained models found in MODELS_DIR — run model.train() and save_model() first"
        )
    # Filenames are xgb_YYYY-MM-DD.pkl; lexicographic sort puts latest last
    latest_date = date.fromisoformat(pkls[-1].stem[4:])  # strip "xgb_"
    clf = model.load_model(latest_date)
    logger.info("Loaded model from %s", pkls[-1].name)

    return edge_finder.find_edges(features_df, clf, game_date)
