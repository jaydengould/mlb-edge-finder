"""Smoke tests: features exposes expected public API."""
import inspect


def test_build_features_signature():
    """build_features should accept odds_df and stats_df DataFrames."""
    from mlb_edge_finder import features
    assert callable(features.build_features)
    sig = inspect.signature(features.build_features)
    assert "odds_df" in sig.parameters
    assert "stats_df" in sig.parameters


def test_load_features_signature():
    """load_features should accept game_date."""
    from mlb_edge_finder import features
    assert callable(features.load_features)
    sig = inspect.signature(features.load_features)
    assert "game_date" in sig.parameters
