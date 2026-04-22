"""Fetch and cache team batting and pitching stats.

Primary source: FanGraphs via pybaseball (team_batting, team_pitching) with up
to 3 attempts and exponential backoff (2s, 4s, 8s). Falls back to Baseball
Reference (batting_stats_bref, pitching_stats_bref) if FanGraphs remains
unavailable. Output always includes a data_source column ("fangraphs" or
"bbref") and fip / fip_computed columns (one is NaN depending on source).
"""
import logging
import time
from datetime import date

import pandas as pd
from pybaseball import (
    batting_stats_bref,
    pitching_stats_bref,
    team_batting,
    team_pitching,
)

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
_BBREF_ABBR_NORMALIZE: dict[str, str] = {
    "KCR": "KC",   # Kansas City
    "TBR": "TB",   # Tampa Bay
    "SDP": "SD",   # San Diego
    "SFG": "SF",   # San Francisco
}

_FIP_CONSTANT: float = 3.15
_RETRY_DELAYS: list[int] = [2, 4, 8]


def _ip_bbref_to_decimal(ip: pd.Series) -> pd.Series:
    """Convert BBRef IP notation (.1 = ⅓ inning, .2 = ⅔ inning) to decimal."""
    whole = ip.apply(lambda x: int(x) if pd.notna(x) else 0.0)
    frac = (ip - whole).round(1)
    return whole + frac.map({0.0: 0.0, 0.1: 1 / 3, 0.2: 2 / 3}).fillna(0.0)


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


def _build_bbref(season: int) -> pd.DataFrame:
    bat_raw = batting_stats_bref(season)
    pit_raw = pitching_stats_bref(season)

    if bat_raw is None or bat_raw.empty:
        raise RuntimeError(f"batting_stats_bref returned no data for season {season}")
    if pit_raw is None or pit_raw.empty:
        raise RuntimeError(f"pitching_stats_bref returned no data for season {season}")

    # Keep only standard 2-3 letter team codes; exclude multi-team total rows
    bat = bat_raw[bat_raw["Tm"].str.match(r"^[A-Z]{2,3}$", na=False)].copy()
    pit = pit_raw[pit_raw["Tm"].str.match(r"^[A-Z]{2,3}$", na=False)].copy()

    bat["Tm"] = bat["Tm"].replace(_BBREF_ABBR_NORMALIZE)
    pit["Tm"] = pit["Tm"].replace(_BBREF_ABBR_NORMALIZE)

    # Aggregate batting counting stats to team level, then derive rates
    bat_agg = bat.groupby("Tm").agg(
        AB=("AB", "sum"),
        H=("H", "sum"),
        doubles=("2B", "sum"),
        triples=("3B", "sum"),
        HR=("HR", "sum"),
        BB=("BB", "sum"),
        HBP=("HBP", "sum"),
        SF=("SF", "sum"),
        R=("R", "sum"),
        G=("G", "max"),
    ).reset_index()

    bat_agg["bat_avg"] = bat_agg["H"] / bat_agg["AB"]
    bat_agg["obp"] = (
        (bat_agg["H"] + bat_agg["BB"] + bat_agg["HBP"].fillna(0))
        / (bat_agg["AB"] + bat_agg["BB"] + bat_agg["HBP"].fillna(0) + bat_agg["SF"].fillna(0))
    )
    tb = bat_agg["H"] + bat_agg["doubles"] + 2 * bat_agg["triples"] + 3 * bat_agg["HR"]
    bat_agg["slg"] = tb / bat_agg["AB"]
    bat_agg["ops"] = bat_agg["obp"] + bat_agg["slg"]
    bat_agg["runs_per_game"] = bat_agg["R"] / bat_agg["G"]
    bat_agg["w_oba"] = float("nan")
    bat_agg["bat_wrc_plus"] = float("nan")

    bat_df = bat_agg[["Tm", "bat_avg", "obp", "slg", "ops", "runs_per_game", "w_oba", "bat_wrc_plus"]].rename(
        columns={"Tm": "team_abbr"}
    )

    # Aggregate pitching counting stats to team level, then derive rates
    pit_agg = pit.groupby("Tm").agg(
        IP_raw=("IP", "sum"),
        H=("H", "sum"),
        ER=("ER", "sum"),
        BB=("BB", "sum"),
        HR=("HR", "sum"),
        SO=("SO", "sum"),
    ).reset_index()

    pit_agg["IP"] = _ip_bbref_to_decimal(pit_agg["IP_raw"])
    pit_agg["era"] = (pit_agg["ER"] / pit_agg["IP"]) * 9
    pit_agg["whip"] = (pit_agg["H"] + pit_agg["BB"]) / pit_agg["IP"]
    pit_agg["k_per_9"] = (pit_agg["SO"] / pit_agg["IP"]) * 9
    pit_agg["bb_per_9"] = (pit_agg["BB"] / pit_agg["IP"]) * 9
    pit_agg["fip"] = float("nan")
    pit_agg["fip_computed"] = (
        (13 * pit_agg["HR"] + 3 * pit_agg["BB"] - 2 * pit_agg["SO"]) / pit_agg["IP"]
        + _FIP_CONSTANT
    )

    pit_df = pit_agg[["Tm", "era", "whip", "k_per_9", "bb_per_9", "fip", "fip_computed"]].rename(
        columns={"Tm": "team_abbr"}
    )

    df = bat_df.merge(pit_df, on="team_abbr", how="inner")
    df["data_source"] = "bbref"
    logger.debug("BBRef: %d teams, %d columns", len(df), len(df.columns))
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
        "FanGraphs failed after %d attempts (%s) — falling back to Baseball Reference",
        total, last_exc,
    )
    return _build_bbref(season)


def fetch_stats(game_date: date, force: bool = False) -> pd.DataFrame:
    """Fetch season-to-date team batting and pitching stats for a given date.

    Tries FanGraphs first (up to 3 attempts, 2s/4s/8s backoff), then falls
    back to Baseball Reference. Writes the result to
    DATA_RAW_DIR/stats_YYYY-MM-DD.csv. Cache-first unless force=True.

    Args:
        game_date: The date for which to fetch stats. Season year is
            game_date.year.
        force: If True, re-fetch even if a cache file exists.

    Returns:
        DataFrame with columns: team_abbr, bat_avg, obp, slg, ops,
        runs_per_game, w_oba, bat_wrc_plus, era, whip, fip, fip_computed,
        k_per_9, bb_per_9, data_source. One row per team.
        w_oba/bat_wrc_plus/fip are NaN for bbref rows.
        fip_computed is NaN for fangraphs rows.

    Raises:
        RuntimeError: If both FanGraphs and Baseball Reference fail.
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
