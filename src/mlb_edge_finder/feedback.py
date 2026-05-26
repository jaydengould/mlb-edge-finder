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


def run_feedback_loop(season: int) -> dict:
    """Refresh historical data for the season and retrain if enough new games exist.

    Always refreshes historical_YYYY.csv from the MLB Stats API. Checks how many
    games have been played since the most recent saved model. If the count reaches
    config.RETRAIN_THRESHOLD (or no model exists), rebuilds the training set for
    all seasons and retrains + calibrates + saves a new model.

    Args:
        season: The current season year (e.g. 2026).

    Returns:
        Dict with keys: season (int), games_in_season (int), new_games (int),
        retrained (bool).
    """
    historical_df = refresh_historical(season)

    pkls = sorted(
        p for p in config.MODELS_DIR.glob("xgb_*.pkl")
        if len(p.stem) == 14  # "xgb_YYYY-MM-DD"
    )
    if pkls:
        last_train_date = date.fromisoformat(pkls[-1].stem[4:])  # strip "xgb_"
        new_games = games_since_last_train(historical_df, last_train_date)
        do_retrain = new_games >= config.RETRAIN_THRESHOLD
    else:
        last_train_date = None
        new_games = len(historical_df)
        do_retrain = True

    retrained = False
    if do_retrain:
        logger.info(
            "Retraining model: %d new games since %s (threshold: %d)",
            new_games, last_train_date, config.RETRAIN_THRESHOLD,
        )
        try:
            training_df = build_training_set(_TRAINING_SEASONS, force=True)
            clf, X_val, X_test, y_val, y_test = model.train(training_df)
            clf = model.calibrate(clf, X_val, y_val)
            metrics = model.evaluate(clf, X_test, y_test)
            model.save_model(clf, metrics, date.today())
            retrained = True
        except Exception as exc:
            logger.error("Retrain failed: %s — skipping model update", exc)
    else:
        logger.info(
            "Skipping retrain: %d new games since %s (threshold: %d)",
            new_games, last_train_date, config.RETRAIN_THRESHOLD,
        )

    return {
        "season": season,
        "games_in_season": len(historical_df),
        "new_games": new_games,
        "retrained": retrained,
    }
