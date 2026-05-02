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


def find_edges(features_df: pd.DataFrame, clf: XGBClassifier, game_date: date) -> pd.DataFrame:
    """Run inference and return games with positive expected value.

    Uses clf.feature_names_in_ to select exactly the columns the model was
    trained on, then runs two passes (home, away) to find bets where:
      - EV > config.EV_THRESHOLD
      - The relevant team's American odds >= config.MIN_AMERICAN_ODDS

    Logs a warning and returns an empty DataFrame (with correct columns) if no
    edges are found. Writes results to DATA_PROCESSED_DIR/edges_YYYY-MM-DD.csv.

    Args:
        features_df: Output of features.load_features() or build_features().
            Must contain all columns in clf.feature_names_in_, plus
            game_id, home_team, away_team, home_odds_american, away_odds_american.
        clf: Fitted XGBClassifier from model.load_model() or train().
        game_date: Used to name the output CSV.

    Returns:
        DataFrame with columns: game_id, home_team, away_team,
        bet_side, american_odds, model_prob, ev — one row per flagged edge.

    Raises:
        ValueError: If features_df is missing any column in clf.feature_names_in_.
    """
    output_cols = ["game_id", "home_team", "away_team", "bet_side", "american_odds", "model_prob", "ev"]

    feature_cols = list(clf.feature_names_in_)
    missing = [c for c in feature_cols if c not in features_df.columns]
    if missing:
        raise ValueError(f"features_df missing columns required by model: {missing}")

    df = features_df.reset_index(drop=True)
    X = df[feature_cols]
    home_prob = clf.predict_proba(X)[:, 1]
    away_prob = 1.0 - home_prob

    # Home pass
    home_ev = pd.Series(
        [compute_ev(float(p), int(o)) for p, o in zip(home_prob, df["home_odds_american"])]
    )
    home_mask = (home_ev > config.EV_THRESHOLD) & (df["home_odds_american"] >= config.MIN_AMERICAN_ODDS)
    home_edges = df.loc[home_mask, ["game_id", "home_team", "away_team"]].copy()
    home_edges["bet_side"] = "home"
    home_edges["american_odds"] = df.loc[home_mask, "home_odds_american"].values
    home_edges["model_prob"] = home_prob[home_mask.values]
    home_edges["ev"] = home_ev[home_mask].values

    # Away pass
    away_ev = pd.Series(
        [compute_ev(float(p), int(o)) for p, o in zip(away_prob, df["away_odds_american"])]
    )
    away_mask = (away_ev > config.EV_THRESHOLD) & (df["away_odds_american"] >= config.MIN_AMERICAN_ODDS)
    away_edges = df.loc[away_mask, ["game_id", "home_team", "away_team"]].copy()
    away_edges["bet_side"] = "away"
    away_edges["american_odds"] = df.loc[away_mask, "away_odds_american"].values
    away_edges["model_prob"] = away_prob[away_mask.values]
    away_edges["ev"] = away_ev[away_mask].values

    edges = pd.concat([home_edges[output_cols], away_edges[output_cols]], ignore_index=True)

    if edges.empty:
        logger.warning("No edges found for %s", game_date)
        return pd.DataFrame(columns=output_cols)

    config.DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.DATA_PROCESSED_DIR / f"edges_{game_date}.csv"
    edges.to_csv(out_path, index=False)
    logger.info("Found %d edge(s) for %s → %s", len(edges), game_date, out_path)

    return edges
