"""Build and cache a labeled training dataset for XGBoost model training."""
import logging
from datetime import date

import pandas as pd

from mlb_edge_finder import config
from mlb_edge_finder.historical_ingestion import load_cached_historical
from mlb_edge_finder.stats_ingestion import fetch_stats

logger = logging.getLogger(__name__)

# statsapi full team names → current franchise abbreviations.
# Always use the current abbreviation regardless of historical team name,
# so training features are consistent with inference-time features.
HISTORICAL_NAME_TO_ABBR: dict[str, str] = {
    "Arizona Diamondbacks": "ARI",
    "Atlanta Braves": "ATL",
    "Baltimore Orioles": "BAL",
    "Boston Red Sox": "BOS",
    "Chicago Cubs": "CHC",
    "Chicago White Sox": "CWS",
    "Cincinnati Reds": "CIN",
    "Cleveland Guardians": "CLE",
    "Colorado Rockies": "COL",
    "Detroit Tigers": "DET",
    "Houston Astros": "HOU",
    "Kansas City Royals": "KC",
    "Los Angeles Angels": "LAA",
    "Los Angeles Dodgers": "LAD",
    "Miami Marlins": "MIA",
    "Milwaukee Brewers": "MIL",
    "Minnesota Twins": "MIN",
    "New York Mets": "NYM",
    "New York Yankees": "NYY",
    "Oakland Athletics": "ATH",   # pre-2025 statsapi name; franchise moved to Sacramento
    "Athletics": "ATH",           # 2025+ statsapi name
    "Philadelphia Phillies": "PHI",
    "Pittsburgh Pirates": "PIT",
    "San Diego Padres": "SD",
    "San Francisco Giants": "SF",
    "Seattle Mariners": "SEA",
    "St. Louis Cardinals": "STL",
    "Tampa Bay Rays": "TB",
    "Texas Rangers": "TEX",
    "Toronto Blue Jays": "TOR",
    "Washington Nationals": "WSH",
}

# FanGraphs abbreviations that changed between seasons → current abbreviation.
# Applied to the stats DataFrame before joining so both join sides use current identifiers.
_LEGACY_ABBR_NORMALIZE: dict[str, str] = {
    "OAK": "ATH",   # Oakland → Sacramento Athletics
}

_SNAPSHOT_MONTH = 9
_SNAPSHOT_DAY = 28


def build_training_set(seasons: list[int], force: bool = False) -> pd.DataFrame:
    raise NotImplementedError


def load_training_set(seasons: list[int]) -> pd.DataFrame:
    raise NotImplementedError
