"""Grade the published edge history against real results and persist the summary.

Reads every outputs/edges_*.csv, joins real final scores, and writes
models/live_grading.json so the README can cite a file instead of
hard-coded numbers.

Usage:
    python scripts/grade_live.py
"""
import json
from datetime import date

import pandas as pd

from mlb_win_probability import config
from mlb_win_probability.backtest import compute_summary, grade_live_edges

OUT_PATH = config.MODELS_DIR / "live_grading.json"


def _summarize(bets: pd.DataFrame) -> dict:
    """Summary block for one slice of graded bets."""
    s = compute_summary(bets)
    return {
        "n_bets": s["n_bets"],
        "date_range": (
            [str(bets["game_date"].min()), str(bets["game_date"].max())]
            if not bets.empty
            else None
        ),
        "win_rate": s["win_rate"],
        "roi": s["roi_pct"],
        "sharpe": s["sharpe_ratio"],
        "max_drawdown": s["max_drawdown"],
        "mean_model_prob_on_flagged_bets": (
            round(float(bets["model_prob"].mean()), 4) if not bets.empty else None
        ),
        # Same quantity as win_rate, named for the calibration comparison it
        # anchors: mean predicted probability vs. the rate actually realized.
        "realized_win_rate": s["win_rate"],
    }


def main() -> dict:
    graded = grade_live_edges()
    result = {
        "generated_on": date.today().isoformat(),
        "source": "outputs/edges_*.csv graded against final scores from data/raw/historical_*.csv",
        "all": _summarize(graded),
        "high_confidence": _summarize(graded[graded["high_confidence"]]),
    }
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(f"\nWritten to {OUT_PATH}")
    return result


if __name__ == "__main__":
    config.setup_logging()
    main()
