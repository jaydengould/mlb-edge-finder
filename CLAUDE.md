# CLAUDE.md — MLB Edge Finder

## Project Overview

Portfolio project that finds positive expected-value (EV) opportunities in MLB moneyline betting markets. Compares XGBoost-predicted win probabilities against bookmaker-implied probabilities and flags bets where EV exceeds a configurable threshold.

## Current Phase

**Phase 2 — Odds ingestion complete.** `odds_ingestion.fetch_odds()` and `load_cached_odds()` are fully implemented with caching (`force=False` skips re-fetch), date filtering, and error handling. `_parse_response()` collapses multi-bookmaker JSON to one row per game, keeping the best (highest American odds) line across all bookmakers. Live in-game odds are excluded before parsing. `commence_time` UTC timestamps are converted to US/Eastern before date filtering.

**Next:** Implement `stats_ingestion.fetch_stats()` and `load_cached_stats()` — those two complete data ingestion and unlock `features.build_features()`.

## Tech Stack

| Layer | Library |
|---|---|
| Stats data | pybaseball |
| Odds data | The Odds API (REST, v4) |
| Feature engineering | pandas |
| Model | XGBoost (`XGBClassifier`) |
| Config / secrets | python-dotenv |
| Tests | pytest |
| Notebooks | Jupyter |

Python 3.10+ required (`date | None` union syntax used in pipeline).

## Architecture

Flat `src/mlb_edge_finder/` package, installed in editable mode (`pip install -e .`). No sub-packages. One module per pipeline stage.

```
odds_ingestion → stats_ingestion → features → model → edge_finder
                                                           ↑
                                                       pipeline.run()
```

Each stage persists its output as a dated CSV or artifact so stages can be run independently in notebooks.

## Module Responsibilities

| Module | Job | Persists to |
|---|---|---|
| `config.py` | Env loading, path constants, `setup_logging()` | — |
| `odds_ingestion.py` | Fetch/cache moneyline odds (The Odds API) | `data/raw/odds_YYYY-MM-DD.csv` |
| `stats_ingestion.py` | Fetch/cache team + pitcher stats (pybaseball) | `data/raw/stats_YYYY-MM-DD.csv` |
| `features.py` | Merge odds + stats, engineer features | `data/processed/features_YYYY-MM-DD.csv` |
| `model.py` | Train, evaluate, persist XGBoost model | `models/xgb_YYYY-MM-DD.pkl` + `models/metrics_YYYY-MM-DD.json` |
| `edge_finder.py` | Compute EV, filter odds, flag edges | `data/processed/edges_YYYY-MM-DD.csv` |
| `pipeline.py` | Orchestrate all stages end-to-end | — |

## Config Constants (`config.py`)

All paths and thresholds live here — never hardcode them in other modules.

```python
ODDS_API_KEY        # from .env
SPORT               # default: "baseball_mlb"
REGION              # default: "us"
MARKET              # default: "h2h"
DATA_RAW_DIR        # <root>/data/raw/
DATA_PROCESSED_DIR  # <root>/data/processed/
MODELS_DIR          # <root>/models/
LOGS_DIR            # <root>/logs/
XGB_N_ESTIMATORS    # 100
XGB_MAX_DEPTH       # 4
EV_THRESHOLD        # 0.05  (flag EV > 5%)
MIN_AMERICAN_ODDS   # -300  (skip heavy favorites)
```

## Key Conventions

- **Logging:** Every module uses `logger = logging.getLogger(__name__)`. Never configure handlers inside a module — only `config.setup_logging()` does that. Call it once per entry point (notebook first cell, pipeline entry).
- **Dates:** All file naming and function arguments use `datetime.date`. Files are named `*_YYYY-MM-DD.*`. API `commence_time` values are UTC ISO-8601 — always convert to US/Eastern (`America/New_York`) before comparing against a calendar date.
- **Live game exclusion:** `fetch_odds()` drops any game whose `commence_time` is in the past (UTC) before parsing — live in-game odds are not valid for pre-game EV analysis.
- **Odds deduplication:** `_parse_response()` produces one row per `game_id`, not one per bookmaker. The best line across all bookmakers is kept (highest American odds value = best for the bettor). The `bookmaker` column does not appear in the output CSV.
- **Model persistence:** Always two files — `.pkl` for the object, `.json` for metrics/hyperparameters. Never combine them.
- **Error handling:** `odds_ingestion` and `stats_ingestion` raise `RuntimeError` on API failure. `model` raises `FileNotFoundError` if inputs are missing. `edge_finder` logs a warning and returns an empty DataFrame when no edges are found — it never raises on empty results.
- **No silent fallbacks:** If data isn't available, raise. Don't return empty DataFrames as a substitute for real data in ingestion modules.
- **Kelly seam:** `edge_finder.compute_ev(prob, american_odds)` is the EV primitive. When adding Kelly sizing, add `compute_kelly(ev, bankroll)` alongside it — don't modify `compute_ev`.

## EV Formula

```python
# Favorite (negative odds)
EV = prob * (100 / abs(odds)) - (1 - prob)

# Underdog (positive odds)
EV = prob * (odds / 100) - (1 - prob)
```

Implemented in `edge_finder.compute_ev()`. This is the only fully implemented business logic function so far.

## Data Flow (end-to-end)

```
The Odds API ──► fetch_odds()  ──► odds_YYYY-MM-DD.csv ──┐
                                                           ├─► build_features() ──► features_YYYY-MM-DD.csv
pybaseball   ──► fetch_stats() ──► stats_YYYY-MM-DD.csv ──┘         │
                                                                      ▼
                                                               train() / load_model()
                                                                      │
                                                                      ▼
                                                               find_edges() ──► edges_YYYY-MM-DD.csv
```

## Execution Model

1. **Jupyter-first (current):** `notebooks/01_exploration.ipynb` — run stages interactively, inspect DataFrames at each step.
2. **CLI (next):** `pipeline.run()` wired to `__main__.py`.
3. **Scheduled (future):** Wrap `pipeline.run()` with APScheduler or cron.

## Gitignored Paths

```
.env              # secrets
data/raw/         # cached API responses
data/processed/   # engineered features + edge outputs
logs/             # run.log
```

`models/` is NOT gitignored — commit model artifacts for portfolio visibility.

## Running Tests

```bash
pytest tests/ -v
```

17 smoke tests. All pass. Tests verify module imports and public function signatures only — no external API calls.

## Roadmap

- [x] Implement `odds_ingestion.fetch_odds()` and `load_cached_odds()`
- [ ] Implement `stats_ingestion.fetch_stats()` and `load_cached_stats()`
- [ ] Implement `features.build_features()` and `load_features()`
- [ ] Implement `model.train()`, `evaluate()`, `save_model()`, `load_model()`
- [ ] Implement `edge_finder.find_edges()`
- [ ] Implement `pipeline.run()`
- [ ] Add `compute_kelly()` to `edge_finder`
- [ ] Add `__main__.py` CLI entry point
- [ ] Add APScheduler for daily runs
