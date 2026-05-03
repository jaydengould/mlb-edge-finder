"""Merge odds and stats into a model-ready feature DataFrame."""
import logging
from datetime import date

import pandas as pd

from mlb_edge_finder import config
from mlb_edge_finder.historical_ingestion import fetch_historical
from mlb_edge_finder.odds_ingestion import load_cached_odds
from mlb_edge_finder.pitcher_ingestion import fetch_probable_starters, load_cached_pitcher_stats
from mlb_edge_finder.rolling_stats import latest_rolling_stats
from mlb_edge_finder.stats_ingestion import ODDS_NAME_TO_ABBR, load_cached_stats

logger = logging.getLogger(__name__)


def build_features(game_date: date) -> pd.DataFrame:
    """Join odds, stats, and rolling stats into one row per game.

    Loads cached odds and stats for game_date, fetches the current season's
    completed games (cache-first) to compute rolling team form stats, maps
    Odds API full team names to abbreviations, then joins all three data sources
    with home_/away_ prefixes.

    Args:
        game_date: Date whose cached odds and stats CSVs to load.

    Returns:
        DataFrame with one row per game. Columns include all odds fields plus
        home_<stat>/away_<stat> for every stat column and
        home_rolling_*/away_rolling_* for rolling stats.

    Raises:
        RuntimeError: If the cached odds or stats file for game_date is absent,
            or if the historical data fetch fails.
    """
    try:
        odds_df = load_cached_odds(game_date)
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"No cached odds for {game_date} — run fetch_odds() first"
        ) from exc

    try:
        stats_df = load_cached_stats(game_date)
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"No cached stats for {game_date} — run fetch_stats() first"
        ) from exc

    # Fetch current-season completed games for rolling stats (cache-first)
    try:
        hist_df = fetch_historical(game_date.year)
    except RuntimeError as exc:
        raise RuntimeError(
            f"Could not fetch historical data for {game_date.year} — {exc}"
        ) from exc

    # Map full team names → abbreviations
    odds_df = odds_df.copy()
    odds_df["home_abbr"] = odds_df["home_team"].map(ODDS_NAME_TO_ABBR)
    odds_df["away_abbr"] = odds_df["away_team"].map(ODDS_NAME_TO_ABBR)

    unmapped = pd.concat([
        odds_df.loc[odds_df["home_abbr"].isna(), "home_team"],
        odds_df.loc[odds_df["away_abbr"].isna(), "away_team"],
    ]).unique()
    if len(unmapped):
        logger.warning("Unmapped team names dropped from features: %s", list(unmapped))
    odds_df = odds_df.dropna(subset=["home_abbr", "away_abbr"])

    # Drop data_source; it varies by run and is not a model feature
    stats = stats_df.drop(columns=["data_source"], errors="ignore")

    # Build home/away stat frames with prefixes
    stat_cols = [c for c in stats.columns if c != "team_abbr"]
    home_stats = stats.rename(columns={"team_abbr": "home_abbr"} | {c: f"home_{c}" for c in stat_cols})
    away_stats = stats.rename(columns={"team_abbr": "away_abbr"} | {c: f"away_{c}" for c in stat_cols})

    df = odds_df.merge(home_stats, on="home_abbr", how="inner")
    df = df.merge(away_stats, on="away_abbr", how="inner")

    # Join rolling stats by team_abbr only (today's games haven't been played yet)
    rolling_df = latest_rolling_stats(hist_df)
    rolling_cols = [c for c in rolling_df.columns if c != "team_abbr"]
    home_rolling = rolling_df.rename(
        columns={"team_abbr": "home_abbr"} | {c: f"home_{c}" for c in rolling_cols}
    )
    away_rolling = rolling_df.rename(
        columns={"team_abbr": "away_abbr"} | {c: f"away_{c}" for c in rolling_cols}
    )
    df = df.merge(home_rolling[["home_abbr"] + [f"home_{c}" for c in rolling_cols]], on="home_abbr", how="left")
    df = df.merge(away_rolling[["away_abbr"] + [f"away_{c}" for c in rolling_cols]], on="away_abbr", how="left")

    # Join probable starting pitcher names onto the game rows
    probable_df = fetch_probable_starters(game_date)
    df = df.merge(probable_df, on=["home_abbr", "away_abbr"], how="left")

    # Load pitcher stats and double-join with home_sp_*/away_sp_* prefix
    try:
        pitcher_stats = load_cached_pitcher_stats(game_date)
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"No cached pitcher stats for {game_date} — run fetch_pitcher_stats() first"
        ) from exc

    sp_cols = [c for c in pitcher_stats.columns if c not in ("pitcher_name", "pitcher_id")]
    home_pitcher = pitcher_stats.rename(columns={
        "pitcher_name": "home_starter_name",
        "pitcher_id": "home_pitcher_id",
        **{c: f"home_sp_{c}" for c in sp_cols},
    })
    away_pitcher = pitcher_stats.rename(columns={
        "pitcher_name": "away_starter_name",
        "pitcher_id": "away_pitcher_id",
        **{c: f"away_sp_{c}" for c in sp_cols},
    })
    home_pitcher_cols = ["home_starter_name", "home_pitcher_id"] + [f"home_sp_{c}" for c in sp_cols]
    away_pitcher_cols = ["away_starter_name", "away_pitcher_id"] + [f"away_sp_{c}" for c in sp_cols]
    df = df.merge(home_pitcher[home_pitcher_cols], on="home_starter_name", how="left")
    df = df.merge(away_pitcher[away_pitcher_cols], on="away_starter_name", how="left")

    logger.debug(
        "Built features: %d game(s), %d columns (home stat cols: %d), pitcher join %d/%d matched",
        len(df), len(df.columns), len(stat_cols), df["home_pitcher_id"].notna().sum(), len(df),
    )

    config.DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.DATA_PROCESSED_DIR / f"features_{game_date}.csv"
    df.to_csv(out_path, index=False)
    logger.info("Wrote %d rows to %s", len(df), out_path)

    return df


def load_features(game_date: date) -> pd.DataFrame:
    """Load a previously built feature DataFrame from DATA_PROCESSED_DIR.

    Args:
        game_date: The date whose features CSV to load.

    Returns:
        DataFrame with the same schema as build_features().

    Raises:
        FileNotFoundError: If no features file exists for the given date.
    """
    path = config.DATA_PROCESSED_DIR / f"features_{game_date}.csv"
    if not path.exists():
        raise FileNotFoundError(f"No cached features for {game_date}: {path}")
    return pd.read_csv(path)
