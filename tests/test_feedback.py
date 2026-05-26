# tests/test_feedback.py
"""Tests for feedback module."""
from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


def test_retrain_threshold_constant():
    from mlb_edge_finder import config
    assert config.RETRAIN_THRESHOLD == 15
