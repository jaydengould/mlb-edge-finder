"""Compute expected value and identify positive-EV betting edges."""
import logging
from datetime import date

import numpy as np
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


def market_implied_prob(american_odds: int) -> float:
    """Convert American odds to raw bookmaker-implied probability (vig included).

    Args:
        american_odds: Bookmaker's American moneyline. 0 is treated as even money.

    Returns:
        Implied probability in [0.0, 1.0].
    """
    if american_odds == 0:
        logger.warning("market_implied_prob received odds=0, returning 0.5")
        return 0.5
    if american_odds < 0:
        return abs(american_odds) / (abs(american_odds) + 100)
    return 100 / (american_odds + 100)


def compute_kelly(prob: float, american_odds: int) -> float:
    """Compute half-Kelly bet size as a fraction of bankroll.

    Uses the same payout derivation as compute_ev, then applies:
        full_kelly = EV / payout
        half_kelly = full_kelly / 2

    Returns 0.0 for zero or negative EV (no edge, don't bet).
    Result is clamped to [0.0, 1.0].

    Args:
        prob: Model-predicted win probability for the team (0.0 – 1.0).
        american_odds: Bookmaker's American moneyline for the same team.

    Returns:
        Fraction of bankroll to wager (0.0 = no bet, 1.0 = full bankroll).
    """
    if american_odds < 0:
        b = 100 / abs(american_odds)
    else:
        b = american_odds / 100
    ev = prob * b - (1 - prob)
    if ev <= 0:
        return 0.0
    return min(ev / b / 2, 1.0)


def find_edges(
    features_df: pd.DataFrame,
    clf: XGBClassifier,
    game_date: date,
    min_prob_edge: float | None = None,
) -> pd.DataFrame:
    """Run inference and return games with positive expected value.

    Uses clf.feature_names_in_ to select exactly the columns the model was
    trained on, then runs two passes (home, away) to find bets where:
      - EV > config.EV_THRESHOLD
      - The relevant team's American odds >= config.MIN_AMERICAN_ODDS
      - model_prob - market_implied_prob(odds) > min_prob_edge

    Logs a warning and returns an empty DataFrame (with correct columns) if no
    edges are found. Writes results to DATA_PROCESSED_DIR/edges_YYYY-MM-DD.csv.

    Args:
        features_df: Output of features.load_features() or build_features().
            Must contain all columns in clf.feature_names_in_, plus
            game_id, home_team, away_team, home_odds_american, away_odds_american.
        clf: Fitted XGBClassifier from model.load_model() or train().
        game_date: Used to name the output CSV.
        min_prob_edge: Minimum required gap between model_prob and
            market_implied_prob. Defaults to config.MIN_PROB_EDGE.

    Returns:
        DataFrame with columns: game_id, home_team, away_team,
        bet_side, american_odds, model_prob, ev, kelly_fraction, prob_flag —
        one row per flagged edge. prob_flag=True when model_prob > 0.80.

    Raises:
        ValueError: If features_df is missing any column in clf.feature_names_in_.
    """
    if min_prob_edge is None:
        min_prob_edge = config.MIN_PROB_EDGE

    output_cols = [
        "game_id", "home_team", "away_team", "bet_side",
        "american_odds", "model_prob", "ev", "kelly_fraction", "prob_flag",
    ]

    if features_df.empty:
        logger.warning("No features available for %s — returning empty edges", game_date)
        return pd.DataFrame(columns=output_cols)

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
    home_implied = np.array(
        [market_implied_prob(int(o)) for o in df["home_odds_american"]]
    )
    home_mask = (
        (home_ev > config.EV_THRESHOLD)
        & (df["home_odds_american"] >= config.MIN_AMERICAN_ODDS)
        & ((home_prob - home_implied) > min_prob_edge)
    )
    home_edges = df.loc[home_mask, ["game_id", "home_team", "away_team"]].copy()
    home_edges["bet_side"] = "home"
    home_edges["american_odds"] = df.loc[home_mask, "home_odds_american"].values
    home_edges["model_prob"] = home_prob[home_mask.values]
    home_edges["ev"] = home_ev[home_mask].values
    home_edges["kelly_fraction"] = [
        compute_kelly(float(p), int(o))
        for p, o in zip(home_prob[home_mask.values], df.loc[home_mask, "home_odds_american"].values)
    ]
    home_edges["prob_flag"] = home_prob[home_mask.values] > 0.80

    # Away pass
    away_ev = pd.Series(
        [compute_ev(float(p), int(o)) for p, o in zip(away_prob, df["away_odds_american"])]
    )
    away_implied = np.array(
        [market_implied_prob(int(o)) for o in df["away_odds_american"]]
    )
    away_mask = (
        (away_ev > config.EV_THRESHOLD)
        & (df["away_odds_american"] >= config.MIN_AMERICAN_ODDS)
        & ((away_prob - away_implied) > min_prob_edge)
    )
    away_edges = df.loc[away_mask, ["game_id", "home_team", "away_team"]].copy()
    away_edges["bet_side"] = "away"
    away_edges["american_odds"] = df.loc[away_mask, "away_odds_american"].values
    away_edges["model_prob"] = away_prob[away_mask.values]
    away_edges["ev"] = away_ev[away_mask].values
    away_edges["kelly_fraction"] = [
        compute_kelly(float(p), int(o))
        for p, o in zip(away_prob[away_mask.values], df.loc[away_mask, "away_odds_american"].values)
    ]
    away_edges["prob_flag"] = away_prob[away_mask.values] > 0.80

    edges = pd.concat([home_edges[output_cols], away_edges[output_cols]], ignore_index=True)

    logger.info(
        "prob-edge filter (%.0f%%): %d edges kept after all filters for %s",
        min_prob_edge * 100,
        len(edges),
        game_date,
    )

    if edges.empty:
        logger.warning("No edges found for %s", game_date)
        return pd.DataFrame(columns=output_cols)

    config.DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.DATA_PROCESSED_DIR / f"edges_{game_date}.csv"
    edges.to_csv(out_path, index=False)
    logger.info("Found %d edge(s) for %s → %s", len(edges), game_date, out_path)

    return edges
