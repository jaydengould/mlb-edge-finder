"""Smoke tests: config imports cleanly and exposes expected interface."""
import logging


def test_config_imports():
    from mlb_edge_finder import config
    assert hasattr(config, "ODDS_API_KEY")
    assert hasattr(config, "DATA_RAW_DIR")
    assert hasattr(config, "DATA_PROCESSED_DIR")
    assert hasattr(config, "MODELS_DIR")
    assert hasattr(config, "XGB_N_ESTIMATORS")
    assert hasattr(config, "XGB_MAX_DEPTH")
    assert hasattr(config, "EV_THRESHOLD")
    assert hasattr(config, "MIN_AMERICAN_ODDS")


def test_setup_logging_is_callable():
    from mlb_edge_finder import config
    assert callable(config.setup_logging)


def test_setup_logging_runs():
    from mlb_edge_finder import config
    config.setup_logging(level=logging.DEBUG)
    logger = logging.getLogger("test")
    logger.debug("config smoke test")


def test_min_pitcher_ip_constant():
    from mlb_edge_finder import config
    assert config.MIN_PITCHER_IP == 30
