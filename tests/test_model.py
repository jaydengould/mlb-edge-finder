"""Smoke tests: model exposes expected public API."""
import inspect
from datetime import date

import numpy as np
import pandas as pd
import pytest


def _make_df(n=20):
    """Minimal training DataFrame with correct schema for testing."""
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "game_date": ["2024-04-01"] * n,
        "home_name": ["Team A"] * n,
        "away_name": ["Team B"] * n,
        "home_score": rng.integers(0, 10, n),
        "away_score": rng.integers(0, 10, n),
        "home_abbr": ["AAA"] * n,
        "away_abbr": ["BBB"] * n,
        "season": [2024] * n,
        "home_win": [1, 0] * (n // 2),
        "home_bat_avg": rng.uniform(0.220, 0.280, n),
        "home_obp": rng.uniform(0.300, 0.380, n),
        "home_slg": rng.uniform(0.380, 0.500, n),
        "home_ops": rng.uniform(0.680, 0.880, n),
        "home_runs_per_game": rng.uniform(3.5, 6.0, n),
        "home_era": rng.uniform(3.0, 5.5, n),
        "home_whip": rng.uniform(1.1, 1.5, n),
        "home_k_per_9": rng.uniform(7.0, 10.5, n),
        "home_bb_per_9": rng.uniform(2.5, 4.0, n),
        "home_fip_computed": rng.uniform(3.5, 5.0, n),
        "away_bat_avg": rng.uniform(0.220, 0.280, n),
        "away_obp": rng.uniform(0.300, 0.380, n),
        "away_slg": rng.uniform(0.380, 0.500, n),
        "away_ops": rng.uniform(0.680, 0.880, n),
        "away_runs_per_game": rng.uniform(3.5, 6.0, n),
        "away_era": rng.uniform(3.0, 5.5, n),
        "away_whip": rng.uniform(1.1, 1.5, n),
        "away_k_per_9": rng.uniform(7.0, 10.5, n),
        "away_bb_per_9": rng.uniform(2.5, 4.0, n),
        "away_fip_computed": rng.uniform(3.5, 5.0, n),
    })


def test_split_shapes():
    from mlb_edge_finder.model import _split
    df = _make_df(20)
    X_train, X_test, y_train, y_test = _split(df)
    assert len(X_train) == 16
    assert len(X_test) == 4
    assert len(y_train) == 16
    assert len(y_test) == 4


def test_split_no_metadata_columns():
    from mlb_edge_finder.model import _split, NON_FEATURE_COLS
    df = _make_df(20)
    X_train, X_test, y_train, y_test = _split(df)
    for col in NON_FEATURE_COLS:
        assert col not in X_train.columns
        assert col not in X_test.columns


def test_split_missing_target_raises():
    from mlb_edge_finder.model import _split
    df = _make_df(20).drop(columns=["home_win"])
    with pytest.raises(ValueError, match="home_win"):
        _split(df)


def test_split_empty_df_raises():
    from mlb_edge_finder.model import _split
    df = _make_df(20).iloc[0:0]
    with pytest.raises(FileNotFoundError):
        _split(df)


def test_train_returns_classifier_and_test_split():
    from mlb_edge_finder.model import train
    from xgboost import XGBClassifier
    df = _make_df(20)
    clf, X_test, y_test = train(df)
    assert isinstance(clf, XGBClassifier)
    assert len(X_test) == 4
    assert len(y_test) == 4


def test_train_clf_can_predict_proba():
    from mlb_edge_finder.model import train
    df = _make_df(20)
    clf, X_test, y_test = train(df)
    proba = clf.predict_proba(X_test)
    assert proba.shape == (4, 2)
    assert (proba >= 0).all() and (proba <= 1).all()


def test_train_raises_on_missing_target():
    from mlb_edge_finder.model import train
    df = _make_df(20).drop(columns=["home_win"])
    with pytest.raises(ValueError):
        train(df)


def test_train_raises_on_empty_df():
    from mlb_edge_finder.model import train
    df = _make_df(20).iloc[0:0]
    with pytest.raises(FileNotFoundError):
        train(df)


def test_train_baseline_returns_logistic_regression_and_test_split():
    from mlb_edge_finder.model import train_baseline
    from sklearn.linear_model import LogisticRegression
    df = _make_df(20)
    clf, X_test, y_test = train_baseline(df)
    assert isinstance(clf, LogisticRegression)
    assert len(X_test) == 4
    assert len(y_test) == 4


def test_train_baseline_same_test_split_as_train():
    from mlb_edge_finder.model import train, train_baseline
    df = _make_df(20)
    _, X_test_xgb, y_test_xgb = train(df)
    _, X_test_lr, y_test_lr = train_baseline(df)
    pd.testing.assert_frame_equal(X_test_xgb.reset_index(drop=True), X_test_lr.reset_index(drop=True))
    pd.testing.assert_series_equal(y_test_xgb.reset_index(drop=True), y_test_lr.reset_index(drop=True))


def test_train_baseline_can_predict_proba():
    from mlb_edge_finder.model import train_baseline
    df = _make_df(20)
    clf, X_test, y_test = train_baseline(df)
    proba = clf.predict_proba(X_test)
    assert proba.shape == (4, 2)


EXPECTED_METRIC_KEYS = {
    "accuracy", "roc_auc", "log_loss", "brier_score",
    "n_test_samples", "xgb_n_estimators", "xgb_max_depth",
}


def test_evaluate_xgb_returns_all_keys():
    from mlb_edge_finder.model import train, evaluate
    df = _make_df(20)
    clf, X_test, y_test = train(df)
    metrics = evaluate(clf, X_test, y_test)
    assert set(metrics.keys()) == EXPECTED_METRIC_KEYS


def test_evaluate_xgb_metric_ranges():
    from mlb_edge_finder.model import train, evaluate
    df = _make_df(20)
    clf, X_test, y_test = train(df)
    metrics = evaluate(clf, X_test, y_test)
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert metrics["log_loss"] >= 0.0
    assert 0.0 <= metrics["brier_score"] <= 1.0
    assert metrics["n_test_samples"] == 4


def test_evaluate_xgb_hyperparams_populated():
    from mlb_edge_finder.model import train, evaluate
    from mlb_edge_finder import config
    df = _make_df(20)
    clf, X_test, y_test = train(df)
    metrics = evaluate(clf, X_test, y_test)
    assert metrics["xgb_n_estimators"] == config.XGB_N_ESTIMATORS
    assert metrics["xgb_max_depth"] == config.XGB_MAX_DEPTH


def test_evaluate_baseline_hyperparam_keys_are_none():
    from mlb_edge_finder.model import train_baseline, evaluate
    df = _make_df(20)
    clf, X_test, y_test = train_baseline(df)
    metrics = evaluate(clf, X_test, y_test)
    assert set(metrics.keys()) == EXPECTED_METRIC_KEYS
    assert metrics["xgb_n_estimators"] is None
    assert metrics["xgb_max_depth"] is None


def test_save_and_load_model_roundtrip(tmp_path, monkeypatch):
    from mlb_edge_finder import config
    from mlb_edge_finder.model import evaluate, load_model, save_model, train
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    df = _make_df(20)
    clf, X_test, y_test = train(df)
    metrics = evaluate(clf, X_test, y_test)
    game_date = date(2024, 4, 1)
    save_model(clf, metrics, game_date)
    loaded = load_model(game_date)
    import numpy as np
    np.testing.assert_array_equal(
        clf.predict(X_test), loaded.predict(X_test)
    )


def test_save_model_writes_both_files(tmp_path, monkeypatch):
    from mlb_edge_finder import config
    from mlb_edge_finder.model import evaluate, save_model, train
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    df = _make_df(20)
    clf, X_test, y_test = train(df)
    metrics = evaluate(clf, X_test, y_test)
    game_date = date(2024, 4, 1)
    save_model(clf, metrics, game_date)
    assert (tmp_path / "xgb_2024-04-01.pkl").exists()
    assert (tmp_path / "metrics_2024-04-01.json").exists()


def test_load_model_raises_when_missing(tmp_path, monkeypatch):
    from mlb_edge_finder import config
    from mlb_edge_finder.model import load_model
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    with pytest.raises(FileNotFoundError, match="2024-04-01"):
        load_model(date(2024, 4, 1))


def test_train_signature():
    from mlb_edge_finder import model
    assert callable(model.train)
    sig = inspect.signature(model.train)
    assert "features_df" in sig.parameters


def test_evaluate_signature():
    from mlb_edge_finder import model
    assert callable(model.evaluate)
    sig = inspect.signature(model.evaluate)
    assert "clf" in sig.parameters
    assert "X_test" in sig.parameters
    assert "y_test" in sig.parameters


def test_save_model_signature():
    from mlb_edge_finder import model
    assert callable(model.save_model)
    sig = inspect.signature(model.save_model)
    assert "clf" in sig.parameters
    assert "metrics" in sig.parameters
    assert "game_date" in sig.parameters


def test_load_model_signature():
    from mlb_edge_finder import model
    assert callable(model.load_model)
    sig = inspect.signature(model.load_model)
    assert "game_date" in sig.parameters


def test_non_feature_cols_excludes_pitcher_metadata():
    from mlb_edge_finder.model import NON_FEATURE_COLS
    for col in ("home_starter_name", "away_starter_name",
                "home_pitcher_id", "away_pitcher_id"):
        assert col in NON_FEATURE_COLS, f"{col} missing from NON_FEATURE_COLS"
