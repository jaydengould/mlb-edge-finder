"""Smoke tests: edge_finder exposes expected public API and EV math is correct."""
import inspect
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
