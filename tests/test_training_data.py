"""Tests for training_data module."""
import inspect
from datetime import date
from unittest.mock import patch

import pandas as pd
import pytest


def test_build_training_set_signature():
    from mlb_edge_finder import training_data
    assert callable(training_data.build_training_set)
    sig = inspect.signature(training_data.build_training_set)
    assert "seasons" in sig.parameters
    assert "force" in sig.parameters


def test_load_training_set_signature():
    from mlb_edge_finder import training_data
    assert callable(training_data.load_training_set)
    sig = inspect.signature(training_data.load_training_set)
    assert "seasons" in sig.parameters


def test_historical_name_to_abbr_covers_all_30_teams():
    from mlb_edge_finder.training_data import HISTORICAL_NAME_TO_ABBR
    # All 30 current franchise abbreviations must be reachable
    expected_abbrs = {
        "ARI", "ATL", "BAL", "BOS", "CHC", "CWS", "CIN", "CLE", "COL", "DET",
        "HOU", "KC", "LAA", "LAD", "MIA", "MIL", "MIN", "NYM", "NYY", "ATH",
        "PHI", "PIT", "SD", "SF", "SEA", "STL", "TB", "TEX", "TOR", "WSH",
    }
    assert expected_abbrs == set(HISTORICAL_NAME_TO_ABBR.values())


def test_historical_name_to_abbr_maps_oakland():
    from mlb_edge_finder.training_data import HISTORICAL_NAME_TO_ABBR
    # Both statsapi names for the Athletics franchise map to ATH
    assert HISTORICAL_NAME_TO_ABBR["Oakland Athletics"] == "ATH"
    assert HISTORICAL_NAME_TO_ABBR["Athletics"] == "ATH"


def test_legacy_abbr_normalize_maps_oak_to_ath():
    from mlb_edge_finder.training_data import _LEGACY_ABBR_NORMALIZE
    assert _LEGACY_ABBR_NORMALIZE["OAK"] == "ATH"


def test_load_training_set_raises_when_missing(tmp_path):
    from mlb_edge_finder import training_data
    with patch("mlb_edge_finder.training_data.config.DATA_PROCESSED_DIR", tmp_path):
        with pytest.raises(FileNotFoundError):
            training_data.load_training_set([2023, 2024, 2025])
