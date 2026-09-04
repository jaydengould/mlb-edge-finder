"""Fetch and cache historical MLB regular season game results."""
import logging
import time
from datetime import date

import pandas as pd
import statsapi

from mlb_win_probability import config

logger = logging.getLogger(__name__)

_SEASON_START = "03-20"
_SEASON_END = "09-30"
_KEEP_COLS = [
    "game_date", "home_name", "away_name", "home_score", "away_score", "home_win",
    "home_probable_pitcher", "away_probable_pitcher",
]
_HISTORICAL_SEASONS = [2019, 2021, 2022, 2023, 2024, 2025]
_RETRY_DELAYS = [2, 4, 8]


def fetch_historical(season: int, force: bool = False) -> pd.DataFrame:
    """Fetch completed regular season games for a given year via the MLB Stats API.

    Pulls all games between March 20 and September 30 of the given season,
    filters to regular season finals, and derives home_win. Writes to
    DATA_RAW_DIR/historical_YYYY.csv. Cache-first unless force=True.

    Args:
        season: The season year (e.g. 2024).
        force: If True, re-fetch even if a cache file exists.

    Returns:
        DataFrame with columns: game_date, home_name, away_name,
        home_score, away_score, home_win. One row per completed game.

    Raises:
        RuntimeError: If the API call fails or returns no completed games.
    """
    cache_path = config.DATA_RAW_DIR / f"historical_{season}.csv"
    if cache_path.exists() and not force:
        logger.debug("Cache hit for %d, loading from disk", season)
        return load_cached_historical(season)

    start_date = f"{season}-{_SEASON_START}"
    end_date = f"{season}-{_SEASON_END}"

    last_exc: Exception | None = None
    games = None
    total = len(_RETRY_DELAYS)
    for attempt, delay in enumerate(_RETRY_DELAYS, start=1):
        try:
            games = statsapi.schedule(start_date=start_date, end_date=end_date, sportId=1)
            break
        except Exception as exc:
            last_exc = exc
            if attempt < total:
                logger.warning(
                    "statsapi.schedule attempt %d/%d failed for season %d: %s — retrying in %ds",
                    attempt, total, season, exc, delay,
                )
                time.sleep(delay)
    if games is None:
        if cache_path.exists():
            logger.warning(
                "statsapi.schedule failed for season %d after %d attempts (%s) — using stale cache",
                season, total, last_exc,
            )
            return load_cached_historical(season)
        raise RuntimeError(
            f"statsapi.schedule failed for season {season}: {last_exc}"
        ) from last_exc

    if not games:
        raise RuntimeError(f"statsapi.schedule returned no data for season {season}")

    df = pd.DataFrame(games)
    df = df[(df["game_type"] == "R") & (df["status"] == "Final")].copy()

    if df.empty:
        raise RuntimeError(f"No completed regular season games found for season {season}")

    df["home_win"] = (df["home_score"] > df["away_score"]).astype(int)
    df = df.reindex(columns=_KEEP_COLS)
    df = df.rename(columns={
        "home_probable_pitcher": "home_starter_name",
        "away_probable_pitcher": "away_starter_name",
    })
    for col in ("home_starter_name", "away_starter_name"):
        df[col] = df[col].replace("", None)
    df = df.reset_index(drop=True)

    config.DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path, index=False)
    logger.info("Wrote %d games to %s", len(df), cache_path)

    return df


def load_cached_historical(season: int) -> pd.DataFrame:
    """Load previously fetched game results from DATA_RAW_DIR/historical_YYYY.csv.

    Args:
        season: The season year whose cached CSV to load.

    Returns:
        DataFrame with the same schema as fetch_historical().

    Raises:
        FileNotFoundError: If no cached file exists for the given season.
    """
    cache_path = config.DATA_RAW_DIR / f"historical_{season}.csv"
    if not cache_path.exists():
        raise FileNotFoundError(f"No cached historical data for {season}: {cache_path}")
    return pd.read_csv(cache_path)


def fetch_all_historical(force: bool = False) -> pd.DataFrame:
    """Fetch and concatenate historical game results for all training seasons.

    Calls fetch_historical() for each season in _HISTORICAL_SEASONS
    (2023, 2024, 2025) and returns a single concatenated DataFrame.

    Args:
        force: Passed through to fetch_historical() for each season.

    Returns:
        Concatenated DataFrame with the same schema as fetch_historical(),
        covering all training seasons.
    """
    frames = [fetch_historical(s, force=force) for s in _HISTORICAL_SEASONS]
    df = pd.concat(frames, ignore_index=True)
    logger.info(
        "fetch_all_historical: %d total games across seasons %s",
        len(df), _HISTORICAL_SEASONS,
    )
    return df
