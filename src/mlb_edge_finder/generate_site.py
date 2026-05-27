"""Generate the static GitHub Pages dashboard at docs/index.html."""
import json
import logging
from datetime import date
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[2]
OUTPUTS_DIR: Path = _ROOT / "outputs"
DOCS_DIR: Path = _ROOT / "docs"
PNL_PATH: Path = _ROOT / "data" / "backtest_pnl.json"


def _load_edges_data(outputs_dir: Path) -> tuple[list[dict], list[dict]]:
    """Load today's edges and per-day edge counts from outputs/ CSVs.

    Returns:
        (today_rows, history) where history is a list of {date, count} dicts
        covering the last 30 available days sorted oldest-first.
    """
    today = date.today().isoformat()
    csv_files = sorted(outputs_dir.glob("edges_*.csv"))[-30:]

    history: list[dict] = []
    today_rows: list[dict] = []

    for csv_path in csv_files:
        file_date = csv_path.stem[len("edges_"):]
        df = pd.read_csv(csv_path)
        history.append({"date": file_date, "count": len(df)})
        if file_date == today:
            today_rows = df.to_dict(orient="records")

    return today_rows, history


def _load_metrics(metrics_path: "Path | None") -> "dict | None":
    """Return parsed metrics JSON or None if path is missing."""
    if metrics_path is None:
        return None
    p = Path(metrics_path)
    if not p.exists():
        return None
    return json.loads(p.read_text())


def _load_pnl(pnl_path: "Path | None") -> "dict | None":
    """Return parsed backtest P&L JSON or None if path is missing."""
    if pnl_path is None:
        return None
    p = Path(pnl_path)
    if not p.exists():
        return None
    return json.loads(p.read_text())
