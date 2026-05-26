"""Fetch and cache individual pitcher season stats and probable starting pitchers."""
import logging
from datetime import date

import pandas as pd
import statsapi

from mlb_edge_finder import config

logger = logging.getLogger(__name__)

_FIP_CONSTANT: float = 3.15


def _parse_pitcher_splits(splits: list) -> list[dict]:
    """Parse raw statsapi splits into pitcher row dicts. Skips pitchers with ip == 0."""
    rows = []
    for s in splits:
        player = s.get("player", {})
        st = s.get("stat", {})
        ip_str = st.get("inningsPitched", "0") or "0"
        ip = float(ip_str)
        if ip == 0:
            continue
        hr = int(st.get("homeRuns", 0) or 0)
        bb = int(st.get("baseOnBalls", 0) or 0)
        k_out = int(st.get("strikeOuts", 0) or 0)
        fip = (13 * hr + 3 * bb - 2 * k_out) / ip + _FIP_CONSTANT
        rows.append({
            "pitcher_id": player.get("id"),
            "pitcher_name": player.get("fullName"),
            "era": float(st.get("era", 0) or 0),
            "whip": float(st.get("whip", 0) or 0),
            "k_per_9": float(st.get("strikeoutsPer9Inn", 0) or 0),
            "bb_per_9": float(st.get("walksPer9Inn", 0) or 0),
            "ip": ip,
            "fip_computed": fip,
        })
    return rows


def fetch_pitcher_stats(game_date: date, force: bool = False) -> pd.DataFrame:
    """Fetch season-to-date pitching stats for all pitchers via the MLB Stats API.

    Queries the stats endpoint with playerPool=All to get every pitcher's
    season stats in one call. Writes to DATA_RAW_DIR/pitcher_stats_YYYY-MM-DD.csv.
    Cache-first unless force=True.

    Args:
        game_date: Date whose season year to use for the stats fetch.
        force: If True, re-fetch even if a cache file exists.

    Returns:
        DataFrame with columns: pitcher_id, pitcher_name, era, whip,
        k_per_9, bb_per_9, ip, fip_computed. One row per pitcher with IP > 0.

    Raises:
        RuntimeError: If the statsapi call fails or returns no splits.
    """
    cache_path = config.DATA_RAW_DIR / f"pitcher_stats_{game_date}.csv"
    if cache_path.exists() and not force:
        logger.debug("Cache hit for pitcher_stats %s, loading from disk", game_date)
        return load_cached_pitcher_stats(game_date)

    season = game_date.year
    try:
        data = statsapi.get("stats", {
            "stats": "season",
            "group": "pitching",
            "sportId": 1,
            "season": season,
            "playerPool": "All",
            "limit": 5000,
        })
    except Exception as exc:
        raise RuntimeError(
            f"statsapi failed fetching pitcher stats for {season}: {exc}"
        ) from exc

    splits = data.get("stats", [{}])[0].get("splits", [])
    if not splits:
        raise RuntimeError(f"statsapi returned no pitcher stats for season {season}")

    rows = _parse_pitcher_splits(splits)
    df = pd.DataFrame(rows)
    config.DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path, index=False)
    logger.info("Wrote %d pitchers to %s", len(df), cache_path)
    return df


def load_cached_pitcher_stats(game_date: date) -> pd.DataFrame:
    """Load previously fetched pitcher stats from DATA_RAW_DIR/pitcher_stats_YYYY-MM-DD.csv.

    Args:
        game_date: The date whose cached CSV to load.

    Returns:
        DataFrame with the same schema as fetch_pitcher_stats().

    Raises:
        FileNotFoundError: If no cached file exists for the given date.
    """
    cache_path = config.DATA_RAW_DIR / f"pitcher_stats_{game_date}.csv"
    if not cache_path.exists():
        raise FileNotFoundError(
            f"No cached pitcher stats for {game_date}: {cache_path}"
        )
    return pd.read_csv(cache_path)


def fetch_probable_starters(game_date: date) -> pd.DataFrame:
    """Fetch today's probable starting pitchers for all regular season games.

    Calls statsapi.schedule for the given date and extracts home/away
    probable pitcher names. Maps team names to abbreviations via
    HISTORICAL_NAME_TO_ABBR. Not cached — starters can change day-of.

    Args:
        game_date: Date to fetch probable starters for.

    Returns:
        DataFrame with columns: home_abbr, away_abbr,
        home_starter_name, away_starter_name. One row per game.
        Empty string probable pitchers are returned as NaN.
        Returns empty DataFrame (with correct columns) if no games found.

    Raises:
        RuntimeError: If the statsapi.schedule call fails.
    """
    from mlb_edge_finder.rolling_stats import HISTORICAL_NAME_TO_ABBR

    try:
        games = statsapi.schedule(
            start_date=str(game_date),
            end_date=str(game_date),
            sportId=1,
        )
    except Exception as exc:
        raise RuntimeError(
            f"statsapi.schedule failed for {game_date}: {exc}"
        ) from exc

    rows = []
    for g in games:
        if g.get("game_type") != "R":
            continue
        home_abbr = HISTORICAL_NAME_TO_ABBR.get(g.get("home_name", ""))
        away_abbr = HISTORICAL_NAME_TO_ABBR.get(g.get("away_name", ""))
        if home_abbr is None or away_abbr is None:
            logger.warning(
                "fetch_probable_starters: unmapped team in game %s", g.get("game_id")
            )
            continue
        home_starter = g.get("home_probable_pitcher") or None
        away_starter = g.get("away_probable_pitcher") or None
        rows.append({
            "home_abbr": home_abbr,
            "away_abbr": away_abbr,
            "home_starter_name": home_starter,
            "away_starter_name": away_starter,
        })

    return pd.DataFrame(
        rows,
        columns=["home_abbr", "away_abbr", "home_starter_name", "away_starter_name"],
    )
