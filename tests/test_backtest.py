import pytest
from mlb_edge_finder.backtest import simulate_market_odds


def test_simulate_market_odds_default_is_110_110():
    home_american, away_american = simulate_market_odds()
    assert abs(home_american - (-110.0)) < 1.0
    assert abs(away_american - (-110.0)) < 1.0


def test_simulate_market_odds_favored_home():
    home_american, away_american = simulate_market_odds(home_market_prob=0.6)
    assert home_american < 0   # home is favorite
    assert away_american > 0   # away is underdog


def test_simulate_market_odds_implied_probs_sum_to_1_plus_vig():
    vig = 0.05
    home_american, away_american = simulate_market_odds(home_market_prob=0.55, vig=vig)

    def to_implied(american: float) -> float:
        if american < 0:
            return abs(american) / (abs(american) + 100)
        return 100 / (american + 100)

    total = to_implied(home_american) + to_implied(away_american)
    assert abs(total - (1 + vig)) < 0.01


def test_simulate_market_odds_even_home_prob():
    home_american, away_american = simulate_market_odds(home_market_prob=0.5, vig=0.0476)
    assert abs(home_american - away_american) < 0.01


def test_simulate_market_odds_zero_vig_gives_fair_odds():
    home_american, away_american = simulate_market_odds(home_market_prob=0.6, vig=0.0)
    assert abs(home_american - (-150.0)) < 1.0
    assert abs(away_american - (150.0)) < 1.0
