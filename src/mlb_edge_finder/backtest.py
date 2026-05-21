"""Backtest the edge-finder against held-out test data using synthetic market odds."""
import logging

logger = logging.getLogger(__name__)


def simulate_market_odds(
    home_market_prob: float = 0.5,
    vig: float = 0.0476,
) -> tuple[float, float]:
    """Generate synthetic American odds for both sides of a game.

    Splits the vig additively across home and away implied probabilities,
    then converts each to American odds format.

    At the default home_market_prob=0.5, vig=0.0476, both sides return
    approximately -110.0 (the standard even-money MLB line).

    Args:
        home_market_prob: Market-implied probability that home wins. Default 0.5.
        vig: Bookmaker overround (sum of implied probs minus 1). Default 0.0476.

    Returns:
        (home_american, away_american) as floats.
    """
    home_implied = home_market_prob + vig / 2
    away_implied = (1.0 - home_market_prob) + vig / 2

    def _to_american(p: float) -> float:
        if p >= 0.5:
            return -(p / (1.0 - p)) * 100.0
        return ((1.0 - p) / p) * 100.0

    return _to_american(home_implied), _to_american(away_implied)
