"""Fetch and cache MLB moneyline odds from The Odds API."""
import logging
from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from mlb_win_probability import config

logger = logging.getLogger(__name__)


_EASTERN = ZoneInfo("America/New_York")


def _game_local_date(commence_time_iso: str) -> date:
    """Return the US/Eastern calendar date for a UTC ISO-8601 commence_time."""
    dt_utc = datetime.fromisoformat(commence_time_iso.replace("Z", "+00:00"))
    return dt_utc.astimezone(_EASTERN).date()


def _parse_response(games: list[dict], game_date: date, debug: bool = False) -> pd.DataFrame:
    rows = []
    if debug:
        logger.info("Raw API response: %d game(s) returned", len(games))
        for g in games:
            logger.info(
                "  game_id=%s  home=%s  away=%s  commence_time=%s  local_date=%s",
                g.get("id", "?"),
                g.get("home_team", "?"),
                g.get("away_team", "?"),
                g.get("commence_time", "?"),
                _game_local_date(g["commence_time"]) if g.get("commence_time") else "?",
            )
    _EMPTY_COLS = ["game_id", "home_team", "away_team",
                   "home_odds_american", "away_odds_american", "commence_time"]
    for game in games:
        if _game_local_date(game["commence_time"]) != game_date:
            continue
        for bookmaker in game.get("bookmakers", []):
            h2h = next(
                (m for m in bookmaker.get("markets", []) if m["key"] == "h2h"),
                None,
            )
            if h2h is None:
                logger.debug("No h2h market for game %s / bookmaker %s", game["id"], bookmaker["key"])
                continue
            outcomes = {o["name"]: o["price"] for o in h2h["outcomes"]}
            home_odds = outcomes.get(game["home_team"])
            away_odds = outcomes.get(game["away_team"])
            if home_odds is None or away_odds is None:
                logger.debug("Missing outcome for game %s bookmaker %s", game["id"], bookmaker["key"])
                continue
            rows.append({
                "game_id": game["id"],
                "home_team": game["home_team"],
                "away_team": game["away_team"],
                "home_odds_american": int(home_odds),
                "away_odds_american": int(away_odds),
                "bookmaker": bookmaker["key"],
                "commence_time": game["commence_time"],
            })
    if not rows:
        logger.warning("No games found for %s after filtering by date", game_date)
        return pd.DataFrame(columns=_EMPTY_COLS)

    df_raw = pd.DataFrame(rows)
    # One row per game: highest American odds value is always best for the bettor
    # (less negative for favorites, more positive for underdogs).
    df = (
        df_raw.groupby("game_id", as_index=False)
        .agg(
            home_team=("home_team", "first"),
            away_team=("away_team", "first"),
            home_odds_american=("home_odds_american", "max"),
            away_odds_american=("away_odds_american", "max"),
            commence_time=("commence_time", "first"),
        )
    )
    logger.debug("Collapsed %d bookmaker row(s) to %d game(s)", len(df_raw), len(df))
    return df


def fetch_odds(game_date: date, force: bool = False, debug: bool = False) -> pd.DataFrame:
    """Fetch MLB moneyline odds from The Odds API for a given date.

    Calls GET /v4/sports/{sport}/odds with market=h2h for the configured
    region. Writes the raw response to DATA_RAW_DIR/odds_YYYY-MM-DD.csv
    before returning. If a cached file already exists and force=False,
    returns the cached data without making an API call.

    Args:
        game_date: The date for which to fetch odds.
        force: If True, re-fetch from the API even if a cache file exists.
        debug: If True, log every game's commence_time from the raw API
            response before date filtering — useful when 0 rows are returned.

    Returns:
        DataFrame with columns: game_id, home_team, away_team,
        home_odds_american, away_odds_american, commence_time.
        One row per game; odds reflect the best available line across
        all bookmakers (highest American odds value = best for the bettor).
        Games that have already started are excluded (live in-game odds
        are not suitable for pre-game EV analysis).

    Raises:
        RuntimeError: If ODDS_API_KEY is not set or the API returns non-200.
    """
    cache_path = config.DATA_RAW_DIR / f"odds_{game_date}.csv"
    if cache_path.exists() and not force:
        logger.debug("Cache hit for %s, loading from disk", game_date)
        return load_cached_odds(game_date)

    if not config.ODDS_API_KEY:
        msg = "ODDS_API_KEY is not set"
        logger.error(msg)
        raise RuntimeError(msg)

    url = f"https://api.the-odds-api.com/v4/sports/{config.SPORT}/odds"
    params = {
        "apiKey": config.ODDS_API_KEY,
        "regions": config.REGION,
        "markets": config.MARKET,
        "dateFormat": "iso",
        "oddsFormat": "american",
    }

    response = requests.get(url, params=params, timeout=30)
    if response.status_code != 200:
        msg = f"Odds API returned {response.status_code}: {response.text}"
        logger.error(msg)
        raise RuntimeError(msg)

    all_games = response.json()
    now_utc = datetime.now(timezone.utc)
    pregame = [
        g for g in all_games
        if datetime.fromisoformat(g["commence_time"].replace("Z", "+00:00")) > now_utc
    ]
    if len(pregame) < len(all_games):
        logger.info(
            "Excluded %d already-started game(s) — live in-game odds are not used",
            len(all_games) - len(pregame),
        )

    df = _parse_response(pregame, game_date, debug=debug)

    config.DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path, index=False)
    logger.info("Wrote %d rows to %s", len(df), cache_path)

    return df


def load_cached_odds(game_date: date) -> pd.DataFrame:
    """Load previously fetched odds from DATA_RAW_DIR/odds_YYYY-MM-DD.csv.

    Args:
        game_date: The date whose cached CSV to load.

    Returns:
        DataFrame with the same schema as fetch_odds().

    Raises:
        FileNotFoundError: If no cached file exists for the given date.
    """
    cache_path = config.DATA_RAW_DIR / f"odds_{game_date}.csv"
    if not cache_path.exists():
        raise FileNotFoundError(f"No cached odds for {game_date}: {cache_path}")
    return pd.read_csv(cache_path)
