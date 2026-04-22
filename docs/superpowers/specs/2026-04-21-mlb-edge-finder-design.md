# MLB Edge Finder — Design Spec
_Date: 2026-04-21_

## Overview

A portfolio project that finds positive expected-value (EV) opportunities in MLB moneyline betting markets. Compares model-predicted win probabilities against bookmaker-implied probabilities and flags games where the edge exceeds a configurable threshold.

**Tech stack:** Python, pandas, pybaseball, The Odds API, scikit-learn, XGBoost, python-dotenv.

**Execution model:** Jupyter-first (interactive exploration), growing toward a CLI entry point, then scheduled automation.

---

## Project Structure

```
mlb-edge-finder/
├── src/
│   └── mlb_edge_finder/
│       ├── __init__.py
│       ├── config.py            # constants, env loading, logging setup
│       ├── odds_ingestion.py    # fetch & cache moneyline odds (The Odds API)
│       ├── stats_ingestion.py   # fetch & cache team/pitcher stats (pybaseball)
│       ├── features.py          # merge odds + stats into model-ready DataFrame
│       ├── model.py             # train, evaluate, and persist XGBoost model
│       ├── edge_finder.py       # compute EV, filter by odds, flag edges
│       └── pipeline.py          # orchestrates all modules end-to-end
├── notebooks/
│   └── 01_exploration.ipynb
├── data/
│   ├── raw/                     # gitignored
│   └── processed/               # gitignored
├── models/                      # serialized artifacts
├── logs/                        # gitignored; run.log written here
├── tests/
│   ├── test_odds_ingestion.py
│   ├── test_stats_ingestion.py
│   ├── test_features.py
│   ├── test_model.py
│   └── test_edge_finder.py
├── docs/superpowers/specs/
├── .env                         # gitignored
├── .env.template
├── requirements.txt
├── pyproject.toml               # enables `pip install -e .`
└── README.md
```

`pyproject.toml` with a `src/` layout enables `pip install -e .` so notebooks can import `mlb_edge_finder` without `sys.path` hacks.

---

## Module Responsibilities & Data Flow

| Module | Input | Output | Persists to |
|---|---|---|---|
| `odds_ingestion` | API key, sport/date params | DataFrame of games + moneylines | `data/raw/odds_YYYY-MM-DD.csv` |
| `stats_ingestion` | Date range, team list | DataFrame of team/pitcher stats | `data/raw/stats_YYYY-MM-DD.csv` |
| `features` | Raw odds + stats CSVs | Merged feature DataFrame | `data/processed/features_YYYY-MM-DD.csv` |
| `model` | Features CSV | Trained XGBoost model + eval metrics | `models/xgb_YYYY-MM-DD.pkl` + `models/metrics_YYYY-MM-DD.json` |
| `edge_finder` | Features + loaded model | EV-scored DataFrame, flagged edges | `data/processed/edges_YYYY-MM-DD.csv` |
| `pipeline` | Config only | Runs all stages in order | — |

### Kelly Criterion Seam

`edge_finder` exposes `compute_ev(prob: float, american_odds: int) -> float`. Adding Kelly later means adding `compute_kelly(ev: float, bankroll: float) -> float` alongside it — no structural changes required.

---

## Config, Environment & Logging

### `config.py` constants

```python
# Sourced from .env
ODDS_API_KEY: str

# Data paths
DATA_RAW_DIR: Path
DATA_PROCESSED_DIR: Path
MODELS_DIR: Path

# Model hyperparameters
XGB_N_ESTIMATORS: int = 100
XGB_MAX_DEPTH: int = 4

# Edge-finding thresholds
EV_THRESHOLD: float = 0.05        # flag bets with EV > 5%
MIN_AMERICAN_ODDS: int = -300     # filter out heavy favorites
```

### `.env.template` keys

```
ODDS_API_KEY=
SPORT=baseball_mlb
REGION=us
MARKET=h2h
```

### Logging

`config.setup_logging(level=logging.INFO)` configures a root logger with a `StreamHandler` (always on) and an optional `FileHandler` at `logs/run.log`. Every module uses `logger = logging.getLogger(__name__)`. Notebooks call `config.setup_logging()` in the first cell.

---

## Error Handling

| Module | Failure scenario | Behavior |
|---|---|---|
| `odds_ingestion` | API request fails | Raise `RuntimeError` with logged message |
| `stats_ingestion` | API/network failure | Raise `RuntimeError` with logged message |
| `model` | Features CSV missing | Raise `FileNotFoundError` |
| `edge_finder` | No edges clear threshold/filter | Log warning, return empty DataFrame |

No silent fallbacks. No retry logic at scaffold stage.

---

## Testing Strategy

`tests/` contains one placeholder file per module. Each file has a single smoke-test stub with a docstring describing what it should eventually verify. Tests are structural at this stage — no external API calls, no mocking infrastructure yet.

---

## Future Extensions

- **Kelly criterion:** Add `compute_kelly()` to `edge_finder` alongside `compute_ev()`.
- **CLI entry point:** Add `__main__.py` calling `pipeline.run()`.
- **Scheduling:** Wrap `pipeline.run()` with APScheduler or a cron job.
- **Alerting:** Add an `alerts.py` module to push edges via email or Slack.
