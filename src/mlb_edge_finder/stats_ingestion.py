"""Fetch and cache team batting and pitching stats.

Primary source: FanGraphs via pybaseball (team_batting, team_pitching) with up
to 3 attempts and exponential backoff (2s, 4s, 8s). Falls back to the MLB
Stats API via the statsapi package if FanGraphs remains unavailable. Output
always includes a data_source column ("fangraphs" or "mlb_api") and fip /
fip_computed columns (one is NaN depending on source).
"""
import logging
import time
from datetime import date

import pandas as pd
import statsapi
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

# FanGraphs column selection and rename
_FG_BAT_COLS = ["Team", "G", "AVG", "OBP", "SLG", "OPS", "R", "wOBA", "wRC+"]
_FG_PIT_COLS = ["Team", "ERA", "WHIP", "FIP", "K/9", "BB/9"]

_FG_BAT_RENAME: dict[str, str] = {
    "Team": "team_abbr",
    "AVG": "bat_avg",
    "OBP": "obp",
    "SLG": "slg",
    "OPS": "ops",
    "wOBA": "w_oba",
    "wRC+": "bat_wrc_plus",
}
_FG_PIT_RENAME: dict[str, str] = {
    "Team": "team_abbr",
    "ERA": "era",
    "WHIP": "whip",
    "FIP": "fip",
    "K/9": "k_per_9",
    "BB/9": "bb_per_9",
}

# Normalize source-specific abbreviations to the standard set in ODDS_NAME_TO_ABBR
_FG_ABBR_NORMALIZE: dict[str, str] = {
    "WSN": "WSH",  # Washington
    "KCR": "KC",   # Kansas City
    "TBR": "TB",   # Tampa Bay
}
_FIP_CONSTANT: float = 3.15
_RETRY_DELAYS: list[int] = [2, 4, 8]


def _build_fangraphs(season: int) -> pd.DataFrame:
    bat = team_batting(season, season, qual=0)
    if bat.empty:
        raise RuntimeError(f"team_batting returned no data for season {season}")
    missing = [c for c in _FG_BAT_COLS if c not in bat.columns]
    if missing:
        raise RuntimeError(f"team_batting missing columns: {missing}")

    pit = team_pitching(season, season, qual=0)
    if pit.empty:
        raise RuntimeError(f"team_pitching returned no data for season {season}")
    missing = [c for c in _FG_PIT_COLS if c not in pit.columns]
    if missing:
        raise RuntimeError(f"team_pitching missing columns: {missing}")

    bat_df = bat[_FG_BAT_COLS].copy()
    bat_df["runs_per_game"] = bat_df["R"] / bat_df["G"]
    bat_df = bat_df.drop(columns=["R", "G"]).rename(columns=_FG_BAT_RENAME)
    bat_df["team_abbr"] = bat_df["team_abbr"].replace(_FG_ABBR_NORMALIZE)

    pit_df = pit[_FG_PIT_COLS].rename(columns=_FG_PIT_RENAME)
    pit_df["team_abbr"] = pit_df["team_abbr"].replace(_FG_ABBR_NORMALIZE)

    df = bat_df.merge(pit_df, on="team_abbr", how="inner")
    df["fip_computed"] = float("nan")
    df["data_source"] = "fangraphs"
    logger.debug("FanGraphs: %d teams, %d columns", len(df), len(df.columns))
    return df


def _build_mlb_api(season: int) -> pd.DataFrame:
    # Fetch team abbreviation map: {team_id: abbreviation}
    teams_data = statsapi.get("teams", {"sportId": 1, "season": season})
    abbr_map: dict[int, str] = {t["id"]: t["abbreviation"] for t in teams_data.get("teams", [])}
    if not abbr_map:
        raise RuntimeError(f"MLB Stats API returned no teams for season {season}")

    # Fetch team hitting stats
    hit_data = statsapi.get("teams_stats", {"stats": "season", "group": "hitting", "season": season, "sportIds": 1})
    hit_splits = hit_data.get("stats", [{}])[0].get("splits", [])
    if not hit_splits:
        raise RuntimeError(f"MLB Stats API returned no team hitting stats for season {season}")

    # Fetch team pitching stats
    pit_data = statsapi.get("teams_stats", {"stats": "season", "group": "pitching", "season": season, "sportIds": 1})
    pit_splits = pit_data.get("stats", [{}])[0].get("splits", [])
    if not pit_splits:
        raise RuntimeError(f"MLB Stats API returned no team pitching stats for season {season}")

    bat_rows = []
    for s in hit_splits:
        abbr = abbr_map.get(s["team"]["id"])
        if abbr is None:
            continue
        st = s["stat"]
        bat_rows.append({
            "team_abbr": abbr,
            "bat_avg": st["avg"],
            "obp": st["obp"],
            "slg": st["slg"],
            "ops": st["ops"],
            "runs_per_game": st["runs"] / st["gamesPlayed"],
        })

    pit_rows = []
    for s in pit_splits:
        abbr = abbr_map.get(s["team"]["id"])
        if abbr is None:
            continue
        st = s["stat"]
        ip = float(st["inningsPitched"])
        pit_rows.append({
            "team_abbr": abbr,
            "era": st["era"],
            "whip": st["whip"],
            "k_per_9": st["strikeoutsPer9Inn"],
            "bb_per_9": st["walksPer9Inn"],
            "fip_computed": (13 * st["homeRuns"] + 3 * st["baseOnBalls"] - 2 * st["strikeOuts"]) / ip + _FIP_CONSTANT,
        })

    bat_df = pd.DataFrame(bat_rows)
    pit_df = pd.DataFrame(pit_rows)

    df = bat_df.merge(pit_df, on="team_abbr", how="inner")
    df["data_source"] = "mlb_api"
    logger.debug("MLB Stats API: %d teams, %d columns", len(df), len(df.columns))
    return df


def _build_stats_df(season: int) -> pd.DataFrame:
    last_exc: Exception | None = None
    total = len(_RETRY_DELAYS)
    for attempt, delay in enumerate(_RETRY_DELAYS, start=1):
        try:
            return _build_fangraphs(season)
        except Exception as exc:
            last_exc = exc
            if attempt < total:
                logger.warning(
                    "FanGraphs attempt %d/%d failed: %s — retrying in %ds",
                    attempt, total, exc, delay,
                )
                time.sleep(delay)

    logger.warning(
        "FanGraphs failed after %d attempts (%s) — falling back to MLB Stats API",
        total, last_exc,
    )
    return _build_mlb_api(season)


def fetch_stats(game_date: date, force: bool = False) -> pd.DataFrame:
    """Fetch season-to-date team batting and pitching stats for a given date.

    Tries FanGraphs first (up to 3 attempts, 2s/4s/8s backoff), then falls
    back to the MLB Stats API via statsapi. Writes the result to
    DATA_RAW_DIR/stats_YYYY-MM-DD.csv. Cache-first unless force=True.

    Args:
        game_date: The date for which to fetch stats. Season year is
            game_date.year.
        force: If True, re-fetch even if a cache file exists.

    Returns:
        DataFrame with columns: team_abbr, bat_avg, obp, slg, ops,
        runs_per_game, era, whip, k_per_9, bb_per_9, data_source.
        One row per team. FanGraphs rows also include w_oba, bat_wrc_plus,
        fip. MLB Stats API rows also include fip_computed. Columns absent
        for a given source are not present (not NaN) — features.py must
        check for them with `col in df.columns` before use.

    Raises:
        RuntimeError: If both FanGraphs and the MLB Stats API fail.
    """
    cache_path = config.DATA_RAW_DIR / f"stats_{game_date}.csv"
    if cache_path.exists() and not force:
        logger.debug("Cache hit for %s, loading from disk", game_date)
        return load_cached_stats(game_date)

    df = _build_stats_df(game_date.year)

    config.DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path, index=False)
    logger.info("Wrote %d rows to %s (source: %s)", len(df), cache_path, df["data_source"].iloc[0])

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
