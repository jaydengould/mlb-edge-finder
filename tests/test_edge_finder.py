"""Smoke tests: edge_finder exposes expected public API and EV math is correct."""
import inspect
from datetime import date
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


def test_compute_ev_signature():
    from mlb_edge_finder import edge_finder
    assert callable(edge_finder.compute_ev)
    sig = inspect.signature(edge_finder.compute_ev)
    assert "prob" in sig.parameters
    assert "american_odds" in sig.parameters


def test_compute_ev_favorite():
    """Negative American odds: EV = prob * (100 / abs(odds)) - (1 - prob)."""
    from mlb_edge_finder.edge_finder import compute_ev
    # 60% model prob, -150 line → EV = 0.60*(100/150) - 0.40 = 0.40 - 0.40 = 0.00
    ev = compute_ev(prob=0.60, american_odds=-150)
    assert abs(ev) < 1e-9


def test_compute_ev_underdog():
    """Positive American odds: EV = prob * (odds / 100) - (1 - prob)."""
    from mlb_edge_finder.edge_finder import compute_ev
    # 40% model prob, +150 line → EV = 0.40*(150/100) - 0.60 = 0.60 - 0.60 = 0.00
    ev = compute_ev(prob=0.40, american_odds=150)
    assert abs(ev) < 1e-9


def test_find_edges_signature():
    from mlb_edge_finder import edge_finder
    assert callable(edge_finder.find_edges)
    sig = inspect.signature(edge_finder.find_edges)
    assert "features_df" in sig.parameters
    assert "clf" in sig.parameters


# ---------- helpers ----------

FEATURE_COLS = ["home_bat_avg", "away_bat_avg", "home_era", "away_era"]


def _make_clf(home_proba: float) -> MagicMock:
    clf = MagicMock()
    clf.feature_names_in_ = np.array(FEATURE_COLS)
    clf.predict_proba.side_effect = lambda X: np.column_stack(
        [np.full(len(X), 1.0 - home_proba), np.full(len(X), home_proba)]
    )
    return clf


def _make_features(home_odds: int, away_odds: int) -> pd.DataFrame:
    return pd.DataFrame([{
        "game_id": "game_1",
        "home_team": "New York Yankees",
        "away_team": "Boston Red Sox",
        "home_odds_american": home_odds,
        "away_odds_american": away_odds,
        "commence_time": "2025-07-01T18:00:00Z",
        "home_bat_avg": 0.260,
        "away_bat_avg": 0.255,
        "home_era": 3.80,
        "away_era": 4.10,
    }])


GAME_DATE = date(2025, 7, 1)


# ---------- behavioural tests ----------

def test_find_edges_returns_home_edge(tmp_path):
    """Home side with EV > threshold and odds >= MIN_AMERICAN_ODDS is flagged."""
    from mlb_edge_finder.edge_finder import find_edges
    # home_prob=0.75, home_odds=+110 → EV = 0.75*1.10 - 0.25 = 0.575 > 0.05 ✓
    # away_prob=0.25, away_odds=-140 → EV = 0.25*(100/140) - 0.75 = -0.571 < 0.05 ✗
    features_df = _make_features(home_odds=110, away_odds=-140)
    clf = _make_clf(home_proba=0.75)

    with patch("mlb_edge_finder.edge_finder.config.DATA_PROCESSED_DIR", tmp_path), \
         patch("mlb_edge_finder.edge_finder.config.EV_THRESHOLD", 0.05), \
         patch("mlb_edge_finder.edge_finder.config.MIN_AMERICAN_ODDS", -300):
        result = find_edges(features_df, clf, GAME_DATE)

    assert len(result) == 1
    assert result.iloc[0]["bet_side"] == "home"
    assert result.iloc[0]["american_odds"] == 110
    assert abs(result.iloc[0]["model_prob"] - 0.75) < 1e-6
    assert result.iloc[0]["ev"] > 0.05
    assert set(result.columns) == {
        "game_id", "home_team", "away_team", "bet_side",
        "american_odds", "model_prob", "ev", "kelly_fraction",
    }


def test_find_edges_filters_min_odds(tmp_path):
    """Game where EV > threshold but odds below MIN_AMERICAN_ODDS is excluded."""
    from mlb_edge_finder.edge_finder import find_edges
    # home_prob=0.90, home_odds=-400 → EV = 0.90*(100/400) - 0.10 = 0.125 > 0.05 ✓
    # BUT -400 < -300 (MIN_AMERICAN_ODDS) → excluded
    features_df = _make_features(home_odds=-400, away_odds=310)
    clf = _make_clf(home_proba=0.90)

    with patch("mlb_edge_finder.edge_finder.config.DATA_PROCESSED_DIR", tmp_path), \
         patch("mlb_edge_finder.edge_finder.config.EV_THRESHOLD", 0.05), \
         patch("mlb_edge_finder.edge_finder.config.MIN_AMERICAN_ODDS", -300):
        result = find_edges(features_df, clf, GAME_DATE)

    assert result.empty
    assert set(result.columns) == {
        "game_id", "home_team", "away_team", "bet_side",
        "american_odds", "model_prob", "ev", "kelly_fraction",
    }


def test_find_edges_empty_when_no_edges(tmp_path):
    """Returns empty DataFrame with correct columns when no games pass filters."""
    from mlb_edge_finder.edge_finder import find_edges
    # home_prob=0.50, home_odds=-110 → EV = 0.50*(100/110) - 0.50 = -0.045 < 0.05 ✗
    features_df = _make_features(home_odds=-110, away_odds=-110)
    clf = _make_clf(home_proba=0.50)

    with patch("mlb_edge_finder.edge_finder.config.DATA_PROCESSED_DIR", tmp_path), \
         patch("mlb_edge_finder.edge_finder.config.EV_THRESHOLD", 0.05), \
         patch("mlb_edge_finder.edge_finder.config.MIN_AMERICAN_ODDS", -300):
        result = find_edges(features_df, clf, GAME_DATE)

    assert result.empty
    assert set(result.columns) == {
        "game_id", "home_team", "away_team", "bet_side",
        "american_odds", "model_prob", "ev", "kelly_fraction",
    }


def test_find_edges_both_sides(tmp_path):
    """Both home and away edges are returned when both sides pass filters."""
    from mlb_edge_finder.edge_finder import find_edges
    # home_prob=0.60, home_odds=+130 → EV = 0.60*1.30 - 0.40 = 0.38 > 0.05 ✓
    # away_prob=0.40, away_odds=+200 → EV = 0.40*2.00 - 0.60 = 0.20 > 0.05 ✓
    features_df = _make_features(home_odds=130, away_odds=200)
    clf = _make_clf(home_proba=0.60)

    with patch("mlb_edge_finder.edge_finder.config.DATA_PROCESSED_DIR", tmp_path), \
         patch("mlb_edge_finder.edge_finder.config.EV_THRESHOLD", 0.05), \
         patch("mlb_edge_finder.edge_finder.config.MIN_AMERICAN_ODDS", -300):
        result = find_edges(features_df, clf, GAME_DATE)

    assert len(result) == 2
    assert set(result["bet_side"]) == {"home", "away"}


# ---- compute_kelly tests ----

def test_compute_kelly_signature():
    from mlb_edge_finder import edge_finder
    assert callable(edge_finder.compute_kelly)
    sig = inspect.signature(edge_finder.compute_kelly)
    assert "prob" in sig.parameters
    assert "american_odds" in sig.parameters


def test_compute_kelly_zero_ev_returns_zero():
    """Zero EV → Kelly fraction is 0.0.

    prob=0.60, -150: b=100/150=0.6667, ev=0.60*0.6667-0.40=0.0 → kelly=0.0
    """
    from mlb_edge_finder.edge_finder import compute_kelly
    result = compute_kelly(prob=0.60, american_odds=-150)
    assert abs(result) < 1e-9


def test_compute_kelly_positive_ev_underdog():
    """Positive EV underdog returns correct half-Kelly fraction.

    prob=0.55, +110: b=1.10, ev=0.55*1.10-0.45=0.155
    full_kelly=0.155/1.10=0.14091, half_kelly=0.07045
    """
    from mlb_edge_finder.edge_finder import compute_kelly
    result = compute_kelly(prob=0.55, american_odds=110)
    assert abs(result - 0.0705) < 1e-3


def test_compute_kelly_negative_ev_returns_zero():
    """Negative EV → Kelly fraction is 0.0 (clamp, don't bet).

    prob=0.40, -150: b=0.6667, ev=0.40*0.6667-0.60=-0.333 → kelly=0.0
    """
    from mlb_edge_finder.edge_finder import compute_kelly
    result = compute_kelly(prob=0.40, american_odds=-150)
    assert result == 0.0


def test_compute_kelly_result_in_valid_range():
    """Result is always in [0.0, 1.0] for valid inputs."""
    from mlb_edge_finder.edge_finder import compute_kelly
    for prob, odds in [(0.99, 100), (0.55, 200), (0.60, -120), (0.45, -110)]:
        result = compute_kelly(prob=prob, american_odds=odds)
        assert 0.0 <= result <= 1.0, f"Out of range for prob={prob}, odds={odds}: {result}"


def test_find_edges_includes_kelly_fraction(tmp_path):
    """find_edges output contains kelly_fraction column with a positive value."""
    from mlb_edge_finder.edge_finder import find_edges
    # home_prob=0.75, home_odds=+110 → positive EV → positive Kelly fraction
    features_df = _make_features(home_odds=110, away_odds=-140)
    clf = _make_clf(home_proba=0.75)

    with patch("mlb_edge_finder.edge_finder.config.DATA_PROCESSED_DIR", tmp_path), \
         patch("mlb_edge_finder.edge_finder.config.EV_THRESHOLD", 0.05), \
         patch("mlb_edge_finder.edge_finder.config.MIN_AMERICAN_ODDS", -300):
        result = find_edges(features_df, clf, GAME_DATE)

    assert "kelly_fraction" in result.columns
    assert result.iloc[0]["kelly_fraction"] > 0.0


def test_find_edges_missing_feature_column(tmp_path):
    """Raises ValueError when features_df is missing a column the model needs."""
    from mlb_edge_finder.edge_finder import find_edges
    features_df = pd.DataFrame([{
        "game_id": "game_1",
        "home_team": "New York Yankees",
        "away_team": "Boston Red Sox",
        "home_odds_american": 110,
        "away_odds_american": -130,
        # missing home_bat_avg, away_bat_avg, home_era, away_era
    }])
    clf = _make_clf(home_proba=0.60)

    with pytest.raises(ValueError, match="missing columns"):
        find_edges(features_df, clf, GAME_DATE)
