import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


def _make_training_df(n_per_season: int = 60) -> pd.DataFrame:
    """Minimal training DataFrame with two seasons (2023 and 2025)."""
    rng = np.random.default_rng(42)
    n = n_per_season * 2
    seasons = [2023] * n_per_season + [2025] * n_per_season
    return pd.DataFrame({
        "season": seasons,
        "game_date": ["2023-04-01"] * n_per_season + ["2025-04-01"] * n_per_season,
        "home_name": "TeamA",
        "away_name": "TeamB",
        "home_score": rng.integers(0, 10, n),
        "away_score": rng.integers(0, 10, n),
        "home_abbr": "TA",
        "away_abbr": "TB",
        "home_win": rng.integers(0, 2, n),
        "home_starter_name": None,
        "away_starter_name": None,
        "home_pitcher_id": None,
        "away_pitcher_id": None,
        "feature_a": rng.random(n),
        "feature_b": rng.random(n),
    })


def _make_mock_clf(n_features: int = 2) -> MagicMock:
    clf = MagicMock()
    clf.feature_names_in_ = np.array([f"feature_{chr(97+i)}" for i in range(n_features)])
    clf.predict_proba = MagicMock(
        side_effect=lambda X: np.column_stack([np.full(len(X), 0.35), np.full(len(X), 0.65)])
    )
    clf.predict = MagicMock(return_value=np.ones(60, dtype=int))
    return clf


# --- _temporal_split ---

def test_temporal_split_train_has_no_holdout_season():
    from mlb_edge_finder.temporal_eval import _temporal_split
    df = _make_training_df()
    train_df, _ = _temporal_split(df, holdout_season=2025)
    assert (train_df["season"] < 2025).all()


def test_temporal_split_test_is_only_holdout_season():
    from mlb_edge_finder.temporal_eval import _temporal_split
    df = _make_training_df()
    _, test_df = _temporal_split(df, holdout_season=2025)
    assert (test_df["season"] == 2025).all()


def test_temporal_split_raises_if_no_train():
    from mlb_edge_finder.temporal_eval import _temporal_split
    df = pd.DataFrame({"season": [2025] * 10, "home_win": [1] * 10, "f": [0.5] * 10})
    with pytest.raises(RuntimeError, match="No training data"):
        _temporal_split(df, holdout_season=2025)


def test_temporal_split_raises_if_no_test():
    from mlb_edge_finder.temporal_eval import _temporal_split
    df = pd.DataFrame({"season": [2023] * 10, "home_win": [1] * 10, "f": [0.5] * 10})
    with pytest.raises(RuntimeError, match="No test data"):
        _temporal_split(df, holdout_season=2025)


# --- run() ---

def _run_with_mocks(tmp_path: Path, training_df: pd.DataFrame, force: bool = False) -> dict:
    """Call temporal_eval.run() with all expensive operations mocked."""
    import mlb_edge_finder.temporal_eval as te
    mock_clf = _make_mock_clf()
    empty_backtest = pd.DataFrame(columns=[
        "game_date", "home_name", "away_name", "bet_side", "american_odds",
        "model_prob", "ev", "kelly_fraction", "actual_home_win", "won", "pnl", "cumulative_pnl",
    ])
    fake_sweep = pd.DataFrame({
        "alpha": [0.0, 0.5, 1.0],
        "roi_pct": [18.0, 4.0, -4.0],
        "n_bets": [200, 90, 8],
        "win_rate": [0.61, 0.55, 0.50],
    })

    with patch.object(te, "_load_training_csv", return_value=training_df), \
         patch("mlb_edge_finder.temporal_eval.XGBClassifier") as MockXGB, \
         patch("mlb_edge_finder.temporal_eval.calibrate", return_value=mock_clf), \
         patch("mlb_edge_finder.temporal_eval.evaluate", return_value={
             "accuracy": 0.57, "roc_auc": 0.60, "log_loss": 0.68, "brier_score": 0.24,
             "n_test_samples": 60,
         }), \
         patch("mlb_edge_finder.temporal_eval.simulate_bets", return_value=empty_backtest), \
         patch("mlb_edge_finder.temporal_eval.sweep_market_efficiency", return_value=fake_sweep), \
         patch("mlb_edge_finder.temporal_eval.compute_summary", return_value={
             "n_bets": 0, "n_wins": 0, "win_rate": 0.0, "total_pnl": 0.0,
             "roi_pct": 0.0, "avg_ev": 0.0, "max_drawdown": 0.0, "sharpe_ratio": 0.0,
         }), \
         patch("mlb_edge_finder.temporal_eval.config") as mock_config:
        mock_config.MODELS_DIR = tmp_path
        mock_config.DATA_PROCESSED_DIR = tmp_path
        mock_config.XGB_N_ESTIMATORS = 10
        mock_config.XGB_MAX_DEPTH = 3
        mock_xgb_instance = MagicMock()
        mock_xgb_instance.feature_names_in_ = np.array(["feature_a", "feature_b"])
        MockXGB.return_value = mock_xgb_instance
        result = te.run(holdout_season=2025, force=force)
    return result


def test_run_writes_json(tmp_path):
    df = _make_training_df()
    _run_with_mocks(tmp_path, df)
    assert (tmp_path / "temporal_eval_2025.json").exists()


def test_run_json_has_required_keys(tmp_path):
    df = _make_training_df()
    result = _run_with_mocks(tmp_path, df)
    required = {
        "holdout_season", "train_seasons", "n_train", "n_test",
        "accuracy", "roc_auc", "log_loss", "brier_score",
        "n_bets", "win_rate", "roi_pct", "sharpe_ratio",
        "total_pnl", "avg_ev", "max_drawdown",
        "market_efficiency_sweep", "break_even_alpha",
    }
    assert required.issubset(set(result.keys()))
    assert "pnl_series" not in result


def test_run_market_efficiency_sweep_is_list(tmp_path):
    df = _make_training_df()
    result = _run_with_mocks(tmp_path, df)
    sweep = result["market_efficiency_sweep"]
    assert isinstance(sweep, list)
    assert sweep and set(sweep[0].keys()) == {"alpha", "roi_pct", "n_bets"}


def test_run_break_even_alpha_interpolated(tmp_path):
    # fake_sweep crosses 0 between alpha=0.5 (roi 4.0) and alpha=1.0 (roi -4.0)
    # crossing at 0.5 + 0.5 * (4.0 / (4.0 - -4.0)) = 0.75
    df = _make_training_df()
    result = _run_with_mocks(tmp_path, df)
    assert result["break_even_alpha"] == 0.75


def test_break_even_alpha_returns_none_when_never_negative():
    from mlb_edge_finder.temporal_eval import _break_even_alpha
    sweep = pd.DataFrame({"alpha": [0.0, 0.5, 1.0], "roi_pct": [10.0, 5.0, 1.0]})
    assert _break_even_alpha(sweep) is None


def test_run_skips_if_exists(tmp_path):
    existing = {"holdout_season": 2025, "roc_auc": 0.999, "market_efficiency_sweep": []}
    (tmp_path / "temporal_eval_2025.json").write_text(json.dumps(existing))
    import mlb_edge_finder.temporal_eval as te
    with patch("mlb_edge_finder.temporal_eval.config") as mock_config:
        mock_config.MODELS_DIR = tmp_path
        result = te.run(holdout_season=2025, force=False)
    assert result["roc_auc"] == 0.999


def test_run_force_overwrites(tmp_path):
    existing = {"holdout_season": 2025, "roc_auc": 0.999, "market_efficiency_sweep": []}
    (tmp_path / "temporal_eval_2025.json").write_text(json.dumps(existing))
    df = _make_training_df()
    result = _run_with_mocks(tmp_path, df, force=True)
    assert result["roc_auc"] != 0.999


def test_run_raises_if_no_training_csv(tmp_path):
    import mlb_edge_finder.temporal_eval as te
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    with patch("mlb_edge_finder.temporal_eval.config") as mock_config:
        mock_config.MODELS_DIR = tmp_path
        mock_config.DATA_PROCESSED_DIR = empty_dir  # no training_*.csv files
        with pytest.raises(RuntimeError, match="No training set found"):
            te.run(holdout_season=2025)
