"""Fetch and cache team batting and pitching stats via the MLB Stats API."""
import logging
from datetime import date

import pandas as pd
import requests

from mlb_edge_finder import config

logger = logging.getLogger(__name__)

_MLB_STATS_BASE = "https://statsapi.mlb.com/api/v1"

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

_BATTING_STAT_KEYS = ["avg", "obp", "slg", "ops", "runs"]
_PITCHING_STAT_KEYS = ["era", "whip", "strikeoutsPer9Inn", "walksPer9Inn"]

_BATTING_RENAME = {
    "avg": "bat_avg",
    "strikeoutsPer9Inn": "k_per_9",
    "walksPer9Inn": "bb_per_9",
}

_PITCHING_RENAME = {
    "strikeoutsPer9Inn": "k_per_9",
    "walksPer9Inn": "bb_per_9",
}


def _fetch_team_abbrs(season: int) -> dict[int, str]:
    r = requests.get(
        f"{_MLB_STATS_BASE}/teams",
        params={"sportId": 1, "season": season},
        timeout=15,
    )
    if r.status_code != 200:
        raise RuntimeError(f"MLB API /teams returned {r.status_code}: {r.text[:200]}")
    return {t["id"]: t["abbreviation"] for t in r.json().get("teams", [])}


def _build_stats_df(season: int) -> pd.DataFrame:
    abbr_map = _fetch_team_abbrs(season)

    r_bat = requests.get(
        f"{_MLB_STATS_BASE}/teams/stats",
        params={"stats": "season", "group": "hitting", "season": season, "sportId": 1},
        timeout=15,
    )
    if r_bat.status_code != 200:
        raise RuntimeError(f"MLB API team hitting stats returned {r_bat.status_code}: {r_bat.text[:200]}")
    bat_splits = r_bat.json().get("stats", [{}])[0].get("splits", [])
    if not bat_splits:
        raise RuntimeError(f"MLB API returned no team batting data for season {season}")

    r_pit = requests.get(
        f"{_MLB_STATS_BASE}/teams/stats",
        params={"stats": "season", "group": "pitching", "season": season, "sportId": 1},
        timeout=15,
    )
    if r_pit.status_code != 200:
        raise RuntimeError(f"MLB API team pitching stats returned {r_pit.status_code}: {r_pit.text[:200]}")
    pit_splits = r_pit.json().get("stats", [{}])[0].get("splits", [])
    if not pit_splits:
        raise RuntimeError(f"MLB API returned no team pitching data for season {season}")

    missing_bat = [k for k in _BATTING_STAT_KEYS if k not in bat_splits[0]["stat"]]
    if missing_bat:
        raise RuntimeError(f"MLB API batting response missing keys: {missing_bat}")
    missing_pit = [k for k in _PITCHING_STAT_KEYS if k not in pit_splits[0]["stat"]]
    if missing_pit:
        raise RuntimeError(f"MLB API pitching response missing keys: {missing_pit}")

    bat_rows = []
    for s in bat_splits:
        abbr = abbr_map.get(s["team"]["id"])
        if abbr is None:
            continue
        row = {"team_abbr": abbr}
        row.update({k: s["stat"][k] for k in _BATTING_STAT_KEYS})
        bat_rows.append(row)

    pit_rows = []
    for s in pit_splits:
        abbr = abbr_map.get(s["team"]["id"])
        if abbr is None:
            continue
        row = {"team_abbr": abbr}
        row.update({k: s["stat"][k] for k in _PITCHING_STAT_KEYS})
        pit_rows.append(row)

    bat = pd.DataFrame(bat_rows).rename(columns=_BATTING_RENAME)
    pit = pd.DataFrame(pit_rows).rename(columns=_PITCHING_RENAME)

    df = bat.merge(pit, on="team_abbr", how="inner")
    logger.debug("Built stats DataFrame: %d teams, %d columns", len(df), len(df.columns))
    return df


def fetch_stats(game_date: date, force: bool = False) -> pd.DataFrame:
    """Fetch season-to-date team batting and pitching stats for a given date.

    Calls the MLB Stats API for the season year derived from game_date. Writes
    the result to DATA_RAW_DIR/stats_YYYY-MM-DD.csv. If a cached file already
    exists and force=False, returns the cached data without an API call.

    Args:
        game_date: The date for which to fetch stats. Season year is
            game_date.year.
        force: If True, re-fetch from the MLB API even if a cache file exists.

    Returns:
        DataFrame with columns: team_abbr, bat_avg, obp, slg, ops, runs,
        era, whip, k_per_9, bb_per_9. One row per team.

    Raises:
        RuntimeError: If the MLB API returns an error or unexpected response.
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
