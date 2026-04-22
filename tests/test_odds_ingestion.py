"""Smoke tests: odds_ingestion exposes expected public API."""
import inspect


def test_fetch_odds_signature():
    """fetch_odds should accept game_date and return a DataFrame."""
    from mlb_edge_finder import odds_ingestion
    assert callable(odds_ingestion.fetch_odds)
    sig = inspect.signature(odds_ingestion.fetch_odds)
    assert "game_date" in sig.parameters


def test_load_cached_odds_signature():
    """load_cached_odds should accept game_date and return a DataFrame."""
    from mlb_edge_finder import odds_ingestion
    assert callable(odds_ingestion.load_cached_odds)
    sig = inspect.signature(odds_ingestion.load_cached_odds)
    assert "game_date" in sig.parameters
