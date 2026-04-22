"""Smoke tests: stats_ingestion exposes expected public API."""
import inspect


def test_fetch_stats_signature():
    """fetch_stats should accept game_date and force."""
    from mlb_edge_finder import stats_ingestion
    assert callable(stats_ingestion.fetch_stats)
    sig = inspect.signature(stats_ingestion.fetch_stats)
    assert "game_date" in sig.parameters
    assert "force" in sig.parameters


def test_load_cached_stats_signature():
    """load_cached_stats should accept game_date."""
    from mlb_edge_finder import stats_ingestion
    assert callable(stats_ingestion.load_cached_stats)
    sig = inspect.signature(stats_ingestion.load_cached_stats)
    assert "game_date" in sig.parameters


def test_odds_name_to_abbr_exists():
    """ODDS_NAME_TO_ABBR should be a dict with 30 entries."""
    from mlb_edge_finder import stats_ingestion
    assert isinstance(stats_ingestion.ODDS_NAME_TO_ABBR, dict)
    assert len(stats_ingestion.ODDS_NAME_TO_ABBR) == 30
