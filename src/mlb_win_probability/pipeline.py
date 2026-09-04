"""Orchestrate all pipeline stages from odds ingestion to flagged-game output."""
import logging
from datetime import date

import pandas as pd

from mlb_win_probability import (
    config, win_probability, features, model,
    odds_ingestion, pitcher_ingestion, stats_ingestion,
)

logger = logging.getLogger(__name__)


def run(game_date: date | None = None, force: bool = False) -> pd.DataFrame:
    """Run the full MLB win-probability pipeline for a single game date.

    Stages (in order):
      1. Fetch or load moneyline odds for game_date.
      2. Fetch or load team stats up to game_date.
      3. Build feature DataFrame from odds + stats.
      4. Auto-discover and load the most recently saved model from MODELS_DIR.
      5. Run win_probability.select_flagged_games() and return the result.

    Args:
        game_date: Date to run the pipeline for. Defaults to today.
        force: If True, re-fetch all data bypassing caches.

    Returns:
        DataFrame of flagged games (may be empty if nothing clears the thresholds).
        Same schema as win_probability.select_flagged_games().

    Raises:
        FileNotFoundError: If no trained models exist in MODELS_DIR.
    """
    if game_date is None:
        game_date = date.today()

    logger.info("Running pipeline for %s", game_date)

    odds_ingestion.fetch_odds(game_date, force=force)
    stats_ingestion.fetch_stats(game_date, force=force)
    pitcher_ingestion.fetch_pitcher_stats(game_date, force=force)
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

    return win_probability.select_flagged_games(features_df, clf, game_date)
