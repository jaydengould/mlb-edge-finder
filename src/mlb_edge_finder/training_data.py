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
# Covers current names (2022+) plus legacy names for pre-rename seasons.
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
    # Legacy names for pre-rename seasons
    "Cleveland Indians": "CLE",       # renamed to Guardians after 2021
    "Florida Marlins": "MIA",         # renamed to Miami Marlins after 2011
    "Tampa Bay Devil Rays": "TB",     # renamed to Rays after 2007
    "Montreal Expos": "WSH",          # relocated to become Nationals in 2005
}

# FanGraphs abbreviations that changed between seasons → current abbreviation.
# Applied to the stats DataFrame before joining so both join sides use current identifiers.
_LEGACY_ABBR_NORMALIZE: dict[str, str] = {
    "OAK": "ATH",   # Oakland → Sacramento Athletics
}

_SNAPSHOT_MONTH = 9
_SNAPSHOT_DAY = 28


def _build_season(season: int) -> pd.DataFrame:
    """Load historical games and end-of-season stats for one season and join them."""
    try:
        hist = load_cached_historical(season)
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"No cached historical data for {season} — run fetch_historical({season}) first"
        ) from exc

    stats = fetch_stats(date(season, _SNAPSHOT_MONTH, _SNAPSHOT_DAY))

    # Normalize FanGraphs abbreviations that changed between seasons
    stats = stats.copy()
    stats["team_abbr"] = stats["team_abbr"].replace(_LEGACY_ABBR_NORMALIZE)

    # Map full team names → current abbreviations
    hist = hist.copy()
    hist["home_abbr"] = hist["home_name"].map(HISTORICAL_NAME_TO_ABBR)
    hist["away_abbr"] = hist["away_name"].map(HISTORICAL_NAME_TO_ABBR)

    unmapped = pd.concat([
        hist.loc[hist["home_abbr"].isna(), "home_name"],
        hist.loc[hist["away_abbr"].isna(), "away_name"],
    ]).unique()
    if len(unmapped):
        logger.warning("Season %d: unmapped team names dropped: %s", season, list(unmapped))
    hist = hist.dropna(subset=["home_abbr", "away_abbr"])

    # Drop data_source — not a model feature
    stats = stats.drop(columns=["data_source"], errors="ignore")

    # Double-join with home_/away_ prefixes (same pattern as features.py)
    stat_cols = [c for c in stats.columns if c != "team_abbr"]
    home_stats = stats.rename(columns={"team_abbr": "home_abbr"} | {c: f"home_{c}" for c in stat_cols})
    away_stats = stats.rename(columns={"team_abbr": "away_abbr"} | {c: f"away_{c}" for c in stat_cols})

    df = hist.merge(home_stats, on="home_abbr", how="inner")
    df = df.merge(away_stats, on="away_abbr", how="inner")
    df["season"] = season

    logger.debug("Season %d: %d games, %d columns", season, len(df), len(df.columns))
    return df


def build_training_set(seasons: list[int], force: bool = False) -> pd.DataFrame:
    """Build and cache a labeled training set by joining historical games with end-of-season stats.

    For each season, loads historical game results and fetches end-of-season stats
    (September 28 snapshot), normalizes abbreviations, joins stats twice with home_/away_
    prefixes, and tags rows with a season column. Concatenates all seasons.

    Args:
        seasons: List of season years to include (e.g. [2023, 2024, 2025]).
        force: If True, rebuild even if a cache file exists.

    Returns:
        DataFrame with one row per game. Columns: game_date, season, home_name, away_name,
        home_abbr, away_abbr, home_win, plus home_<stat> and away_<stat> for every stat column.
        FanGraphs-specific columns (home_w_oba, home_bat_wrc_plus, home_fip) appear when present.

    Raises:
        RuntimeError: If historical data is missing for any season — run fetch_historical(season) first.
            Also propagates RuntimeError from fetch_stats if both FanGraphs and MLB Stats API fail.
    """
    out_path = config.DATA_PROCESSED_DIR / f"training_{min(seasons)}-{max(seasons)}.csv"
    if out_path.exists() and not force:
        logger.debug("Cache hit, loading from %s", out_path)
        return load_training_set(seasons)

    frames = [_build_season(s) for s in seasons]
    df = pd.concat(frames, ignore_index=True)

    config.DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    logger.info("Wrote %d rows (%d seasons) to %s", len(df), len(seasons), out_path)
    return df


def load_training_set(seasons: list[int]) -> pd.DataFrame:
    """Load a previously built training set from DATA_PROCESSED_DIR.

    Args:
        seasons: The seasons list whose training CSV to load (determines filename).

    Returns:
        DataFrame with the same schema as build_training_set().

    Raises:
        FileNotFoundError: If no training set file exists for the given seasons.
    """
    out_path = config.DATA_PROCESSED_DIR / f"training_{min(seasons)}-{max(seasons)}.csv"
    if not out_path.exists():
        raise FileNotFoundError(f"No cached training set for seasons {seasons}: {out_path}")
    return pd.read_csv(out_path)
