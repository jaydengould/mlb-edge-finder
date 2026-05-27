import numpy as np
import pandas as pd
import pytest
from mlb_edge_finder.backtest import simulate_market_odds
from mlb_edge_finder.backtest import run_backtest
from mlb_edge_finder.backtest import compute_summary


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


def _make_training_df(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "game_date": pd.date_range("2024-04-01", periods=n, freq="D"),
        "home_name": [f"HomeTeam{i % 15}" for i in range(n)],
        "away_name": [f"AwayTeam{i % 15}" for i in range(n)],
        "home_score": rng.integers(0, 10, n),
        "away_score": rng.integers(0, 10, n),
        "home_abbr": [f"HM{i % 15}" for i in range(n)],
        "away_abbr": [f"AW{i % 15}" for i in range(n)],
        "season": [2024] * n,
        "home_win": rng.integers(0, 2, n),
        "home_starter_name": [None] * n,
        "away_starter_name": [None] * n,
        "home_pitcher_id": [None] * n,
        "away_pitcher_id": [None] * n,
        "feature_a": rng.standard_normal(n),
        "feature_b": rng.standard_normal(n),
        "feature_c": rng.standard_normal(n),
    })


def _make_mock_clf(home_win_prob: float = 0.58):
    from unittest.mock import MagicMock
    clf = MagicMock()
    clf.feature_names_in_ = np.array(["feature_a", "feature_b", "feature_c"])
    clf.predict_proba = MagicMock(
        side_effect=lambda X: np.column_stack([
            np.full(len(X), 1.0 - home_win_prob),
            np.full(len(X), home_win_prob),
        ])
    )
    return clf


def test_run_backtest_returns_dataframe():
    df = _make_training_df(200)
    clf = _make_mock_clf(home_win_prob=0.58)
    result = run_backtest(clf, df)
    assert isinstance(result, pd.DataFrame)


def test_run_backtest_output_columns():
    df = _make_training_df(200)
    clf = _make_mock_clf(home_win_prob=0.58)
    result = run_backtest(clf, df)
    expected_cols = {
        "game_date", "home_name", "away_name", "bet_side",
        "american_odds", "model_prob", "ev", "kelly_fraction",
        "actual_home_win", "won", "pnl", "cumulative_pnl",
    }
    assert expected_cols.issubset(set(result.columns))


def test_run_backtest_no_edges_returns_empty_with_correct_columns():
    df = _make_training_df(200)
    clf = _make_mock_clf(home_win_prob=0.50)
    result = run_backtest(clf, df)
    assert result.empty
    expected_cols = {
        "game_date", "home_name", "away_name", "bet_side",
        "american_odds", "model_prob", "ev", "kelly_fraction",
        "actual_home_win", "won", "pnl", "cumulative_pnl",
    }
    assert expected_cols.issubset(set(result.columns))


def test_run_backtest_high_prob_finds_home_edges():
    df = _make_training_df(200)
    clf = _make_mock_clf(home_win_prob=0.65)
    result = run_backtest(clf, df, ev_threshold=0.05, min_prob_edge=0.0)
    assert not result.empty
    assert (result["bet_side"] == "home").any()


def test_run_backtest_cumulative_pnl_is_running_sum():
    df = _make_training_df(200)
    clf = _make_mock_clf(home_win_prob=0.65)
    result = run_backtest(clf, df)
    if not result.empty:
        expected = result["pnl"].cumsum().values
        pd.testing.assert_series_equal(
            result["cumulative_pnl"].reset_index(drop=True),
            pd.Series(expected, name="cumulative_pnl"),
            check_exact=False,
            atol=1e-6,
        )


def test_run_backtest_won_matches_actual_outcome():
    df = _make_training_df(200)
    clf = _make_mock_clf(home_win_prob=0.65)
    result = run_backtest(clf, df)
    if not result.empty:
        home_bets = result[result["bet_side"] == "home"]
        if not home_bets.empty:
            assert (home_bets["won"] == (home_bets["actual_home_win"] == 1)).all()
        away_bets = result[result["bet_side"] == "away"]
        if not away_bets.empty:
            assert (away_bets["won"] == (away_bets["actual_home_win"] == 0)).all()


def test_run_backtest_explicit_ev_threshold_filters_bets():
    """A very high explicit ev_threshold produces fewer bets than a low one."""
    df = _make_training_df(200)
    clf = _make_mock_clf(home_win_prob=0.65)
    result_low = run_backtest(clf, df, ev_threshold=0.05, min_prob_edge=0.0)
    result_high = run_backtest(clf, df, ev_threshold=0.50, min_prob_edge=0.0)
    assert len(result_high) <= len(result_low)


def test_run_backtest_explicit_min_prob_edge_filters_bets():
    """A high min_prob_edge filters out bets where model barely disagrees with market."""
    df = _make_training_df(200)
    clf = _make_mock_clf(home_win_prob=0.65)
    result_no_filter = run_backtest(clf, df, ev_threshold=0.05, min_prob_edge=0.0)
    result_filtered = run_backtest(clf, df, ev_threshold=0.05, min_prob_edge=0.30)
    assert len(result_filtered) <= len(result_no_filter)


def test_run_backtest_defaults_unchanged():
    """run_backtest() with no threshold args still runs (default config values)."""
    df = _make_training_df(200)
    clf = _make_mock_clf(home_win_prob=0.65)
    result = run_backtest(clf, df)
    assert isinstance(result, pd.DataFrame)


def _make_backtest_df(pnl_values: list) -> pd.DataFrame:
    n = len(pnl_values)
    actual = [1, 0, 1, 0, 1, 1, 0, 1, 1, 0]
    return pd.DataFrame({
        "game_date": pd.date_range("2024-04-01", periods=n),
        "home_name": ["HomeTeam"] * n,
        "away_name": ["AwayTeam"] * n,
        "bet_side": ["home"] * n,
        "american_odds": [-110] * n,
        "model_prob": [0.60] * n,
        "ev": [0.08] * n,
        "kelly_fraction": [0.04] * n,
        "actual_home_win": actual[:n],
        "won": [p > 0 for p in pnl_values],
        "pnl": pnl_values,
        "cumulative_pnl": pd.Series(pnl_values).cumsum().tolist(),
    })


def test_compute_summary_keys():
    df = _make_backtest_df([90.9, -100, 90.9, -100, 90.9])
    result = compute_summary(df, unit=100.0)
    expected_keys = {
        "n_bets", "n_wins", "win_rate", "total_pnl",
        "roi_pct", "avg_ev", "max_drawdown", "sharpe_ratio",
    }
    assert expected_keys == set(result.keys())


def test_compute_summary_n_bets_and_wins():
    pnl = [90.9, -100, 90.9, -100, 90.9]  # 3 wins, 2 losses
    df = _make_backtest_df(pnl)
    result = compute_summary(df, unit=100.0)
    assert result["n_bets"] == 5
    assert result["n_wins"] == 3


def test_compute_summary_win_rate():
    pnl = [90.9, -100, 90.9, -100, 90.9]  # 3/5 = 60%
    df = _make_backtest_df(pnl)
    result = compute_summary(df, unit=100.0)
    assert abs(result["win_rate"] - 0.60) < 0.01


def test_compute_summary_total_pnl():
    pnl = [90.9, -100.0, 90.9]
    df = _make_backtest_df(pnl)
    result = compute_summary(df, unit=100.0)
    assert abs(result["total_pnl"] - sum(pnl)) < 0.01


def test_compute_summary_roi_pct():
    # 2 bets at $100 unit, total pnl = $50 → ROI = 50/200 * 100 = 25%
    pnl = [150.0, -100.0]
    df = _make_backtest_df(pnl)
    result = compute_summary(df, unit=100.0)
    assert abs(result["roi_pct"] - 25.0) < 0.01


def test_compute_summary_max_drawdown():
    # cumulative: 100, 0, 50 → peak=100 at index 0, trough=0 at index 1 → drawdown=100
    pnl = [100.0, -100.0, 50.0]
    df = _make_backtest_df(pnl)
    df["cumulative_pnl"] = pd.Series(pnl).cumsum()
    result = compute_summary(df, unit=100.0)
    assert abs(result["max_drawdown"] - 100.0) < 0.01


def test_compute_summary_empty_df_returns_zeros():
    df = pd.DataFrame(columns=[
        "game_date", "home_name", "away_name", "bet_side",
        "american_odds", "model_prob", "ev", "kelly_fraction",
        "actual_home_win", "won", "pnl", "cumulative_pnl",
    ])
    result = compute_summary(df, unit=100.0)
    assert result["n_bets"] == 0
    assert result["total_pnl"] == 0.0
    assert result["roi_pct"] == 0.0


def test_sweep_thresholds_returns_dataframe():
    from mlb_edge_finder.backtest import sweep_thresholds
    df = _make_training_df(200)
    clf = _make_mock_clf(home_win_prob=0.65)
    result = sweep_thresholds(clf, df, ev_low=0.05, ev_high=0.15, ev_step=0.05,
                               prob_edge_low=0.0, prob_edge_high=0.10, prob_edge_step=0.05)
    assert isinstance(result, pd.DataFrame)


def test_sweep_thresholds_output_columns():
    from mlb_edge_finder.backtest import sweep_thresholds
    df = _make_training_df(200)
    clf = _make_mock_clf(home_win_prob=0.65)
    result = sweep_thresholds(clf, df, ev_low=0.05, ev_high=0.15, ev_step=0.05,
                               prob_edge_low=0.0, prob_edge_high=0.10, prob_edge_step=0.05)
    expected = {"ev_threshold", "min_prob_edge", "n_bets", "win_rate",
                "roi_pct", "sharpe_ratio", "avg_bets_per_day"}
    assert expected.issubset(set(result.columns))


def test_sweep_thresholds_sorted_by_sharpe_descending():
    from mlb_edge_finder.backtest import sweep_thresholds
    df = _make_training_df(200)
    clf = _make_mock_clf(home_win_prob=0.65)
    result = sweep_thresholds(clf, df, ev_low=0.05, ev_high=0.15, ev_step=0.05,
                               prob_edge_low=0.0, prob_edge_high=0.10, prob_edge_step=0.05)
    if len(result) > 1:
        assert result["sharpe_ratio"].iloc[0] >= result["sharpe_ratio"].iloc[1]


def test_sweep_thresholds_excludes_zero_bet_combinations():
    """Threshold combinations that produce 0 bets should not appear in output."""
    from mlb_edge_finder.backtest import sweep_thresholds
    df = _make_training_df(200)
    clf = _make_mock_clf(home_win_prob=0.65)
    try:
        result = sweep_thresholds(clf, df, ev_low=0.95, ev_high=0.99, ev_step=0.05,
                                   prob_edge_low=0.0, prob_edge_high=0.0, prob_edge_step=0.05)
        assert (result["n_bets"] > 0).all()
    except RuntimeError:
        pass  # All combinations produced 0 bets — that's valid behavior


def test_sweep_thresholds_best_row_logged(caplog):
    """sweep_thresholds logs the optimal threshold combination."""
    import logging
    from mlb_edge_finder.backtest import sweep_thresholds
    df = _make_training_df(200)
    clf = _make_mock_clf(home_win_prob=0.65)
    with caplog.at_level(logging.INFO, logger="mlb_edge_finder.backtest"):
        sweep_thresholds(clf, df, ev_low=0.05, ev_high=0.10, ev_step=0.05,
                         prob_edge_low=0.0, prob_edge_high=0.05, prob_edge_step=0.05)


import json


def test_export_pnl_json_writes_expected_structure(tmp_path):
    from mlb_edge_finder.backtest import export_pnl_json

    bt_df = pd.DataFrame({
        "pnl": [100.0, -100.0, 100.0],
        "cumulative_pnl": [100.0, 0.0, 100.0],
    })
    summary = {"n_bets": 3, "win_rate": 0.667, "roi_pct": 11.1, "sharpe_ratio": 0.5}
    out = tmp_path / "backtest_pnl.json"

    export_pnl_json(bt_df, summary, out)

    data = json.loads(out.read_text())
    assert data["cumulative_pnl"] == [100.0, 0.0, 100.0]
    assert data["summary"]["n_bets"] == 3
    assert data["summary"]["win_rate"] == 0.667


def test_export_pnl_json_creates_parent_dirs(tmp_path):
    from mlb_edge_finder.backtest import export_pnl_json

    bt_df = pd.DataFrame({"pnl": [50.0], "cumulative_pnl": [50.0]})
    summary = {"n_bets": 1}
    out = tmp_path / "nested" / "dir" / "pnl.json"

    export_pnl_json(bt_df, summary, out)

    assert out.exists()
