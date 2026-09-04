"""Temporal out-of-time model evaluation: train on prior seasons, test on holdout."""
import json
import logging
from typing import Any

import pandas as pd
from xgboost import XGBClassifier

from mlb_win_probability import config
from mlb_win_probability.backtest import compute_summary, simulate_bets, sweep_market_efficiency
from mlb_win_probability.model import NON_FEATURE_COLS, TARGET_COL, calibrate, evaluate

logger = logging.getLogger(__name__)


def _temporal_split(
    training_df: pd.DataFrame,
    holdout_season: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split training_df by season column.

    Returns (train_df, test_df) where train contains seasons strictly before
    holdout_season and test contains only holdout_season rows.
    """
    train_df = training_df[training_df["season"] < holdout_season].copy()
    test_df = training_df[training_df["season"] == holdout_season].copy()
    if train_df.empty:
        raise RuntimeError(
            f"No training data found before season {holdout_season} — "
            "check that the training set covers multiple seasons"
        )
    if test_df.empty:
        raise RuntimeError(
            f"No test data found for season {holdout_season} — "
            "check that the training set includes this season"
        )
    return train_df, test_df


def _load_training_csv(data_processed_dir: Any) -> pd.DataFrame:
    """Load the most comprehensive training CSV from data_processed_dir.

    Filenames follow the pattern training_{min}-{max}.csv. Picks the file
    with the earliest start season; breaks ties by latest end season.
    """
    from pathlib import Path

    def _season_key(p: Path) -> tuple[int, int]:
        parts = p.stem.split("_")[1].split("-")
        return (int(parts[0]), int(parts[1]))

    csvs = list(Path(data_processed_dir).glob("training_*.csv"))
    if not csvs:
        raise RuntimeError(
            "No training set found — run build_training_set() first"
        )
    best = min(csvs, key=lambda p: (_season_key(p)[0], -_season_key(p)[1]))
    return pd.read_csv(best)


def _break_even_alpha(sweep_df: pd.DataFrame) -> float | None:
    """Interpolate the alpha where ROI first crosses from >= 0 to < 0.

    Returns None if ROI stays non-negative across the whole grid.
    """
    rows = sweep_df.sort_values("alpha").reset_index(drop=True)
    for i in range(1, len(rows)):
        r0 = rows.loc[i - 1, "roi_pct"]
        r1 = rows.loc[i, "roi_pct"]
        if r0 >= 0 and r1 < 0:
            a0 = rows.loc[i - 1, "alpha"]
            a1 = rows.loc[i, "alpha"]
            alpha_star = a0 + (a1 - a0) * (r0 / (r0 - r1))
            return round(float(alpha_star), 4)
    return None


def run(holdout_season: int = 2025, force: bool = False) -> dict:
    """Train on seasons before holdout_season, evaluate on holdout_season.

    Writes a JSON artifact to MODELS_DIR/temporal_eval_{holdout_season}.json
    containing model metrics + backtest summary + per-bet P&L series.

    Args:
        holdout_season: The season to hold out for testing. Default 2025.
        force: If True, overwrite any existing artifact. Default False.

    Returns:
        The written dict.

    Raises:
        RuntimeError: If no training set CSV is found, or if either the
            train or test slice is empty.
    """
    from sklearn.model_selection import train_test_split

    out_path = config.MODELS_DIR / f"temporal_eval_{holdout_season}.json"
    if out_path.exists() and not force:
        logger.info(
            "Temporal eval already at %s — skipping (use force=True to rerun)", out_path
        )
        return json.loads(out_path.read_text())

    training_df = _load_training_csv(config.DATA_PROCESSED_DIR)
    logger.info("Loaded training set (%d rows)", len(training_df))

    train_df, test_df = _temporal_split(training_df, holdout_season)
    train_seasons = sorted(int(s) for s in train_df["season"].unique())
    logger.info(
        "Temporal split: %d train rows (seasons %s), %d test rows (season %d)",
        len(train_df), train_seasons, len(test_df), holdout_season,
    )

    non_feature = [c for c in NON_FEATURE_COLS if c in train_df.columns]
    X_train_full = train_df.drop(columns=non_feature)
    y_train_full = train_df[TARGET_COL]

    X_test = test_df.drop(columns=non_feature).reindex(columns=X_train_full.columns)
    y_test = test_df[TARGET_COL]

    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.25, random_state=42, stratify=y_train_full
    )

    clf = XGBClassifier(
        n_estimators=config.XGB_N_ESTIMATORS,
        max_depth=config.XGB_MAX_DEPTH,
        eval_metric="logloss",
        random_state=42,
    )
    clf.fit(X_fit, y_fit)
    cal_clf = calibrate(clf, X_val, y_val)
    logger.info("Model trained and calibrated (%d fit, %d val rows)", len(X_fit), len(X_val))

    metrics = evaluate(cal_clf, X_test, y_test)
    logger.info(
        "Holdout metrics: accuracy=%.3f roc_auc=%.3f",
        metrics["accuracy"], metrics["roc_auc"],
    )

    meta_df = test_df[["game_date", "home_name", "away_name"]]
    backtest_df = simulate_bets(cal_clf, X_test, y_test, meta_df)
    summary = compute_summary(backtest_df)

    sweep_df = sweep_market_efficiency(cal_clf, X_test, y_test, meta_df)
    break_even = _break_even_alpha(sweep_df)
    market_efficiency_sweep = [
        {
            "alpha": round(float(r["alpha"]), 4),
            "roi_pct": float(r["roi_pct"]),
            "n_bets": int(r["n_bets"]),
        }
        for _, r in sweep_df.iterrows()
    ]
    logger.info(
        "Market-efficiency sweep: break-even alpha=%s",
        f"{break_even:.3f}" if break_even is not None else "none",
    )

    result = {
        "holdout_season": holdout_season,
        "train_seasons": train_seasons,
        "n_train": len(train_df),
        "n_test": len(test_df),
        "accuracy": round(metrics["accuracy"], 4),
        "roc_auc": round(metrics["roc_auc"], 4),
        "log_loss": round(metrics["log_loss"], 4),
        "brier_score": round(metrics["brier_score"], 4),
        "n_bets": summary["n_bets"],
        "win_rate": summary["win_rate"],
        "roi_pct": summary["roi_pct"],
        "sharpe_ratio": summary["sharpe_ratio"],
        "total_pnl": summary["total_pnl"],
        "avg_ev": summary["avg_ev"],
        "max_drawdown": summary["max_drawdown"],
        "break_even_alpha": break_even,
        "market_efficiency_sweep": market_efficiency_sweep,
    }

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    logger.info(
        "Temporal eval written to %s (ROC-AUC=%.3f, ROI=%.1f%%)",
        out_path, result["roc_auc"], result["roi_pct"],
    )
    return result


if __name__ == "__main__":
    import argparse

    from mlb_win_probability.config import setup_logging

    setup_logging()
    parser = argparse.ArgumentParser(description="Run temporal out-of-time evaluation")
    parser.add_argument("--holdout-season", type=int, default=2025)
    parser.add_argument("--force", action="store_true", help="Overwrite existing artifact")
    args = parser.parse_args()

    r = run(holdout_season=args.holdout_season, force=args.force)
    print(f"\nTemporal Eval — Holdout Season: {r['holdout_season']}")
    print(f"  Train seasons : {r['train_seasons']}")
    print(f"  Train rows    : {r['n_train']:,}")
    print(f"  Test rows     : {r['n_test']:,}")
    print(f"  ROC-AUC       : {r['roc_auc']:.3f}")
    print(f"  Accuracy      : {r['accuracy']:.3f}")
    print(f"  Bets          : {r['n_bets']:,}")
    print(f"  Win Rate      : {r['win_rate'] * 100:.1f}%")
    print(f"  ROI           : {r['roi_pct']:+.1f}%")
    print(f"  Sharpe        : {r['sharpe_ratio']:.3f}")
    be = r["break_even_alpha"]
    print(f"  Break-even α  : {be:.3f}" if be is not None else "  Break-even α  : none (ROI stays positive)")
    print(f"\nArtifact: models/temporal_eval_{r['holdout_season']}.json")
