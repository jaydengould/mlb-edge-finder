"""Smoke tests: model exposes expected public API."""
import inspect

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
