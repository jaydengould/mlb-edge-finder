"""Merge odds and stats into a model-ready feature DataFrame."""
import logging
from datetime import date

import pandas as pd

from mlb_edge_finder import config
from mlb_edge_finder.odds_ingestion import load_cached_odds
from mlb_edge_finder.stats_ingestion import ODDS_NAME_TO_ABBR, load_cached_stats

logger = logging.getLogger(__name__)


def build_features(game_date: date) -> pd.DataFrame:
    """Join odds and stats into one row per game with home_ and away_ stat columns.

    Loads cached odds and stats for game_date, maps Odds API full team names
    to abbreviations via ODDS_NAME_TO_ABBR, then joins stats twice — once for
    the home team and once for the away team — prefixing all stat columns
    accordingly. FanGraphs-specific columns (w_oba, bat_wrc_plus, fip) are
    included when present; their absence is handled gracefully.

    Args:
        game_date: Date whose cached odds and stats CSVs to load.

    Returns:
        DataFrame with one row per game. Columns include all odds fields plus
        home_<stat> and away_<stat> for every stat column in the stats CSV.
        data_source is dropped before output.

    Raises:
        RuntimeError: If the cached odds or stats file for game_date is absent.
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

    # Build home stats: rename team_abbr → home_abbr, prefix stat cols with home_
    stat_cols = [c for c in stats.columns if c != "team_abbr"]
    home_stats = stats.rename(columns={"team_abbr": "home_abbr"} | {c: f"home_{c}" for c in stat_cols})
    away_stats = stats.rename(columns={"team_abbr": "away_abbr"} | {c: f"away_{c}" for c in stat_cols})

    df = odds_df.merge(home_stats, on="home_abbr", how="inner")
    df = df.merge(away_stats, on="away_abbr", how="inner")

    logger.debug(
        "Built features: %d game(s), %d columns (home cols: %d)",
        len(df), len(df.columns), len(stat_cols),
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
