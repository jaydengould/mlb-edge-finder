"""Build and cache a labeled training dataset for XGBoost model training."""
import logging
from datetime import date

import pandas as pd

from mlb_edge_finder import config
from mlb_edge_finder.historical_ingestion import load_cached_historical
from mlb_edge_finder.rolling_stats import HISTORICAL_NAME_TO_ABBR, compute_rolling_stats
from mlb_edge_finder.pitcher_ingestion import fetch_pitcher_stats
from mlb_edge_finder.stats_ingestion import fetch_stats

logger = logging.getLogger(__name__)

# FanGraphs abbreviations that changed between seasons → current abbreviation.
# Applied to the stats DataFrame before joining so both join sides use current identifiers.
_LEGACY_ABBR_NORMALIZE: dict[str, str] = {
    "OAK": "ATH",   # Oakland → Sacramento Athletics
}

_SNAPSHOT_MONTH = 9
_SNAPSHOT_DAY = 28

_PITCHER_SNAPSHOT_MONTH_DAYS: list[tuple[int, int]] = [
    (4, 30), (6, 1), (7, 31), (9, 28),
]


def _select_snapshot_date(
    game_date: date, available_dates: list[date]
) -> date | None:
    """Return the latest snapshot date strictly before game_date, or None."""
    preceding = [d for d in available_dates if d < game_date]
    return max(preceding) if preceding else None


def _build_season(season: int) -> pd.DataFrame:
    """Load historical games, end-of-season stats, and rolling stats for one season."""
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

    # Double-join end-of-season stats with home_/away_ prefixes
    stat_cols = [c for c in stats.columns if c != "team_abbr"]
    home_stats = stats.rename(columns={"team_abbr": "home_abbr"} | {c: f"home_{c}" for c in stat_cols})
    away_stats = stats.rename(columns={"team_abbr": "away_abbr"} | {c: f"away_{c}" for c in stat_cols})

    df = hist.merge(home_stats, on="home_abbr", how="inner")
    df = df.merge(away_stats, on="away_abbr", how="inner")
    df["season"] = season

    # Add rolling stats — computed from game results, no new API calls.
    # shift(1) ensures each game's rolling stats use only prior games.
    # First game of season per team has NaN rolling stats (XGBoost handles NaN natively).
    rolling = compute_rolling_stats(hist)
    rolling_cols = [c for c in rolling.columns if c not in ("team_abbr", "game_date")]
    home_rolling = rolling.rename(
        columns={"team_abbr": "home_abbr"} | {c: f"home_{c}" for c in rolling_cols}
    )
    away_rolling = rolling.rename(
        columns={"team_abbr": "away_abbr"} | {c: f"away_{c}" for c in rolling_cols}
    )
    df = df.merge(
        home_rolling[["home_abbr", "game_date"] + [f"home_{c}" for c in rolling_cols]],
        on=["home_abbr", "game_date"], how="left",
    )
    df = df.merge(
        away_rolling[["away_abbr", "game_date"] + [f"away_{c}" for c in rolling_cols]],
        on=["away_abbr", "game_date"], how="left",
    )

    # Ensure starter name columns exist (may be absent in historical data without probable pitchers)
    for col in ("home_starter_name", "away_starter_name"):
        if col not in df.columns:
            df[col] = None

    # Load available pitcher snapshots for this season.
    # Snapshot files (pitcher_snapshot_YYYY-MM-DD.csv) are committed to the repo by
    # the GitHub Actions snapshot workflow. September 28 falls back to fetch_pitcher_stats
    # for seasons that predate the snapshot workflow.
    snapshot_dates = [
        date(season, m, d) for m, d in _PITCHER_SNAPSHOT_MONTH_DAYS
    ]
    pitcher_snapshots: dict[date, pd.DataFrame] = {}
    for snap_date in snapshot_dates:
        snap_path = config.DATA_RAW_DIR / f"pitcher_snapshot_{snap_date}.csv"
        if snap_path.exists():
            pitcher_snapshots[snap_date] = pd.read_csv(snap_path)
        elif snap_date == date(season, 9, 28):
            pitcher_snapshots[snap_date] = fetch_pitcher_stats(
                date(season, _SNAPSHOT_MONTH, _SNAPSHOT_DAY)
            )

    available_dates = sorted(pitcher_snapshots.keys())

    # Assign each game to its snapshot (latest snapshot strictly before game_date).
    game_dates_as_date = pd.to_datetime(df["game_date"]).dt.date
    df["_snap"] = game_dates_as_date.map(
        lambda gd: _select_snapshot_date(gd, available_dates)
    )

    # Determine pitcher stat columns from any available snapshot.
    any_snap_df = next(iter(pitcher_snapshots.values()), pd.DataFrame())
    sp_cols = [c for c in any_snap_df.columns if c not in ("pitcher_name", "pitcher_id")]
    home_sp_cols = [f"home_sp_{c}" for c in sp_cols]
    away_sp_cols = [f"away_sp_{c}" for c in sp_cols]
    all_pitcher_cols = ["home_pitcher_id", "away_pitcher_id"] + home_sp_cols + away_sp_cols

    groups: list[pd.DataFrame] = []

    # Games with no preceding snapshot → NaN pitcher stats.
    no_snap = df[df["_snap"].isna()].copy()
    if not no_snap.empty:
        for col in all_pitcher_cols:
            no_snap[col] = float("nan")
        groups.append(no_snap)

    # Games with a snapshot → join pitcher stats from the matched snapshot.
    for snap_date, pitcher_stats in pitcher_snapshots.items():
        mask = df["_snap"] == snap_date
        if not mask.any():
            continue
        grp = df[mask].copy()
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
        home_cols = ["home_starter_name", "home_pitcher_id"] + home_sp_cols
        away_cols = ["away_starter_name", "away_pitcher_id"] + away_sp_cols
        grp = grp.merge(home_pitcher[home_cols], on="home_starter_name", how="left")
        grp = grp.merge(away_pitcher[away_cols], on="away_starter_name", how="left")
        groups.append(grp)

    df = pd.concat(groups, ignore_index=True).drop(columns=["_snap"])

    logger.debug(
        "Season %d: pitcher join — %d/%d home starters matched, %d/%d away starters matched",
        season,
        df["home_pitcher_id"].notna().sum(), len(df),
        df["away_pitcher_id"].notna().sum(), len(df),
    )

    logger.debug("Season %d: %d games, %d columns", season, len(df), len(df.columns))
    return df


def build_training_set(seasons: list[int], force: bool = False) -> pd.DataFrame:
    """Build and cache a labeled training set by joining historical games with stats.

    For each season, loads historical game results and fetches end-of-season stats
    (September 28 snapshot), normalizes abbreviations, joins stats twice with home_/away_
    prefixes, adds rolling stats, and tags rows with a season column. Concatenates all seasons.

    Args:
        seasons: List of season years to include (e.g. [2023, 2024, 2025]).
        force: If True, rebuild even if a cache file exists.

    Returns:
        DataFrame with one row per game. Columns: game_date, season, home_name, away_name,
        home_abbr, away_abbr, home_win, plus home_<stat>/away_<stat> for every stat column,
        and home_rolling_*/away_rolling_* for rolling stats.

    Raises:
        RuntimeError: If historical data is missing for any season.
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
