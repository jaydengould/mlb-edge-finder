"""CLI entry point: python -m mlb_win_probability [--date YYYY-MM-DD] [--force]"""
import argparse
import logging
import sys
from datetime import date

from mlb_win_probability import config, pipeline

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="mlb_win_probability",
        description=(
            "Score a day's MLB games with the win-probability model and flag "
            "games where the model diverges from the posted moneyline."
        ),
    )
    parser.add_argument(
        "--date",
        metavar="YYYY-MM-DD",
        default=None,
        help="Game date to score (default: today)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-fetch all data, bypassing caches",
    )
    args = parser.parse_args()

    if args.date is not None:
        try:
            game_date = date.fromisoformat(args.date)
        except ValueError:
            print(f"Error: invalid date '{args.date}' — expected YYYY-MM-DD", file=sys.stderr)
            sys.exit(1)
    else:
        game_date = date.today()

    config.setup_logging()

    try:
        flagged = pipeline.run(game_date, force=args.force)
    except Exception as exc:
        logger.error("Pipeline failed: %s", exc)
        sys.exit(1)

    if flagged.empty:
        print(f"No games flagged for {game_date}.")
    else:
        print(f"{len(flagged)} game side(s) flagged for {game_date}:\n")
        print(flagged.to_string(index=False))
        print(
            "\nFlagged = the model disagrees with the posted line by more than "
            "EV_THRESHOLD.\nThis is a divergence, not a profit forecast — see "
            "models/live_grading.json."
        )


if __name__ == "__main__":
    main()
