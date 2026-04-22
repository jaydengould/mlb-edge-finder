"""Smoke tests: stats_ingestion exposes expected public API."""
import inspect


def test_fetch_stats_signature():
    """fetch_stats should accept start_date and end_date."""
    from mlb_edge_finder import stats_ingestion
    assert callable(stats_ingestion.fetch_stats)
    sig = inspect.signature(stats_ingestion.fetch_stats)
    assert "start_date" in sig.parameters
    assert "end_date" in sig.parameters


def test_load_cached_stats_signature():
    """load_cached_stats should accept game_date."""
    from mlb_edge_finder import stats_ingestion
    assert callable(stats_ingestion.load_cached_stats)
    sig = inspect.signature(stats_ingestion.load_cached_stats)
    assert "game_date" in sig.parameters
