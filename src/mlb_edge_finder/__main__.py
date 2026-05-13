"""CLI entry point: python -m mlb_edge_finder [--date YYYY-MM-DD] [--force]"""
import argparse
import logging
import sys
from datetime import date

from mlb_edge_finder import config, pipeline

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="mlb_edge_finder",
        description="Find positive-EV MLB moneyline bets for a given date.",
    )
    parser.add_argument(
        "--date",
        metavar="YYYY-MM-DD",
        default=None,
        help="Game date to analyse (default: today)",
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
        edges = pipeline.run(game_date, force=args.force)
    except Exception as exc:
        logger.error("Pipeline failed: %s", exc)
        sys.exit(1)

    if edges.empty:
        print(f"No edges found for {game_date}.")
    else:
        print(f"Found {len(edges)} edge(s) for {game_date}:\n")
        print(edges.to_string(index=False))


if __name__ == "__main__":
    main()
