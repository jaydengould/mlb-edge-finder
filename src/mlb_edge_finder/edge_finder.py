"""Compute expected value and identify positive-EV betting edges."""
import logging
from datetime import date

import pandas as pd
from xgboost import XGBClassifier

from mlb_edge_finder import config

logger = logging.getLogger(__name__)


def compute_ev(prob: float, american_odds: int) -> float:
    """Compute expected value of a bet given model probability and American odds.

    Formula:
        Favorites (negative odds): EV = prob * (100 / abs(odds)) - (1 - prob)
        Underdogs (positive odds): EV = prob * (odds / 100) - (1 - prob)

    Args:
        prob: Model-predicted win probability for the team (0.0 – 1.0).
        american_odds: Bookmaker's American moneyline for the same team.

    Returns:
        Expected value per unit wagered. Positive = profitable edge.
    """
    if american_odds < 0:
        payout = 100 / abs(american_odds)
    else:
        payout = american_odds / 100
    return prob * payout - (1 - prob)


def find_edges(features_df: pd.DataFrame, clf: XGBClassifier) -> pd.DataFrame:
    """Run inference and return games with positive expected value.

    Uses clf to predict home-win probabilities, computes EV for both sides
    via compute_ev(), then filters to rows where:
      - EV > config.EV_THRESHOLD
      - The relevant team's American odds >= config.MIN_AMERICAN_ODDS

    Logs a warning and returns an empty DataFrame if no edges are found.
    Writes results to DATA_PROCESSED_DIR/edges_YYYY-MM-DD.csv.

    Args:
        features_df: Output of features.load_features() or build_features().
            Must contain home_odds_american and away_odds_american columns.
        clf: Fitted XGBClassifier from model.load_model() or train().

    Returns:
        DataFrame with columns: game_id, home_team, away_team,
        bet_side, american_odds, model_prob, ev — one row per flagged edge.
    """
    raise NotImplementedError
