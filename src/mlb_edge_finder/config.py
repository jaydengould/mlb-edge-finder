import logging
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# --- Env-sourced ---
ODDS_API_KEY: str = os.getenv("ODDS_API_KEY", "")
SPORT: str = os.getenv("SPORT", "baseball_mlb")
REGION: str = os.getenv("REGION", "us")
MARKET: str = os.getenv("MARKET", "h2h")

# --- Paths ---
_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW_DIR: Path = _ROOT / "data" / "raw"
DATA_PROCESSED_DIR: Path = _ROOT / "data" / "processed"
MODELS_DIR: Path = _ROOT / "models"
LOGS_DIR: Path = _ROOT / "logs"

# --- Model hyperparameters ---
XGB_N_ESTIMATORS: int = 100
XGB_MAX_DEPTH: int = 4

# --- Edge-finding thresholds ---
EV_THRESHOLD: float = 0.05
MIN_AMERICAN_ODDS: int = -300


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logger with a console handler and a file handler at logs/run.log."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    handlers: list[logging.Handler] = [
        logging.StreamHandler(),
        logging.FileHandler(LOGS_DIR / "run.log"),
    ]
    logging.basicConfig(level=level, format=fmt, handlers=handlers, force=True)
