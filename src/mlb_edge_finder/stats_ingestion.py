"""Fetch and cache team and pitcher stats via pybaseball."""
import logging
from datetime import date

import pandas as pd
from pybaseball import team_batting, team_pitching

from mlb_edge_finder import config

logger = logging.getLogger(__name__)

ODDS_NAME_TO_ABBR: dict[str, str] = {
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
    "Athletics": "ATH",
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


_BATTING_COLS = ["Team", "AVG", "OBP", "SLG", "OPS", "R", "wOBA", "wRC+"]
_PITCHING_COLS = ["Team", "ERA", "WHIP", "FIP", "K/9", "BB/9"]

_BATTING_RENAME = {
    "Team": "team_abbr",
    "AVG": "bat_avg",
    "OBP": "obp",
    "SLG": "slg",
    "OPS": "ops",
    "R": "runs",
    "wOBA": "w_oba",
    "wRC+": "bat_wrc_plus",
}

_PITCHING_RENAME = {
    "Team": "team_abbr",
    "ERA": "era",
    "WHIP": "whip",
    "FIP": "fip",
    "K/9": "k_per_9",
    "BB/9": "bb_per_9",
}


def _build_stats_df(season: int) -> pd.DataFrame:
    bat = team_batting(season, season, qual=0)
    if bat.empty:
        raise RuntimeError(f"team_batting returned no data for season {season}")
    missing_bat = [c for c in _BATTING_COLS if c not in bat.columns]
    if missing_bat:
        raise RuntimeError(f"team_batting missing columns: {missing_bat}")

    pit = team_pitching(season, season, qual=0)
    if pit.empty:
        raise RuntimeError(f"team_pitching returned no data for season {season}")
    missing_pit = [c for c in _PITCHING_COLS if c not in pit.columns]
    if missing_pit:
        raise RuntimeError(f"team_pitching missing columns: {missing_pit}")

    bat = bat[_BATTING_COLS].rename(columns=_BATTING_RENAME)
    pit = pit[_PITCHING_COLS].rename(columns=_PITCHING_RENAME)

    df = bat.merge(pit, on="team_abbr", how="inner")
    logger.debug("Built stats DataFrame: %d teams, %d columns", len(df), len(df.columns))
    return df


def fetch_stats(game_date: date, force: bool = False) -> pd.DataFrame:
    """Fetch season-to-date team batting and pitching stats for a given date.

    Calls pybaseball.team_batting() and pybaseball.team_pitching() for the
    season year derived from game_date. Writes the result to
    DATA_RAW_DIR/stats_YYYY-MM-DD.csv. If a cached file already exists and
    force=False, returns the cached data without calling pybaseball.

    Args:
        game_date: The date for which to fetch stats. Season year is
            game_date.year.
        force: If True, re-fetch from pybaseball even if a cache file exists.

    Returns:
        DataFrame with columns: team_abbr, bat_avg, obp, slg, ops, runs,
        w_oba, bat_wrc_plus, era, whip, fip, k_per_9, bb_per_9.
        One row per team.

    Raises:
        RuntimeError: If pybaseball returns no data or expected columns are
            missing.
    """
    cache_path = config.DATA_RAW_DIR / f"stats_{game_date}.csv"
    if cache_path.exists() and not force:
        logger.debug("Cache hit for %s, loading from disk", game_date)
        return load_cached_stats(game_date)

    df = _build_stats_df(game_date.year)

    config.DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path, index=False)
    logger.info("Wrote %d rows to %s", len(df), cache_path)

    return df


def load_cached_stats(game_date: date) -> pd.DataFrame:
    """Load previously fetched stats from DATA_RAW_DIR/stats_YYYY-MM-DD.csv.

    Args:
        game_date: The date whose cached CSV to load.

    Returns:
        DataFrame with the same schema as fetch_stats().

    Raises:
        FileNotFoundError: If no cached file exists for the given date.
    """
    cache_path = config.DATA_RAW_DIR / f"stats_{game_date}.csv"
    if not cache_path.exists():
        raise FileNotFoundError(f"No cached stats for {game_date}: {cache_path}")
    return pd.read_csv(cache_path)
