# CLAUDE.md — MLB Edge Finder

## Project Overview

Portfolio project that finds positive expected-value (EV) opportunities in MLB moneyline betting markets. Compares XGBoost-predicted win probabilities against bookmaker-implied probabilities and flags bets where EV exceeds a configurable threshold.

## Current Phase

**Phase 4 — Model training in progress.** Phases 1–3 and 4a–4b are complete.

- `odds_ingestion.fetch_odds()` and `load_cached_odds()` — cache-first, date filtering, live game exclusion, best line across bookmakers.
- `stats_ingestion.fetch_stats()` and `load_cached_stats()` — FanGraphs primary (pybaseball, 3-attempt retry with 2s/4s/8s backoff), MLB Stats API fallback (statsapi package). Output schema varies by source — see stats schema section below.
- `features.build_features(game_date)` and `load_features(game_date)` — loads cached odds and stats, maps Odds API team names to abbreviations via `ODDS_NAME_TO_ABBR`, double-joins stats with `home_`/`away_` prefixes, writes to `data/processed/features_YYYY-MM-DD.csv`.
- **4a complete:** `historical_ingestion.fetch_historical(season)`, `load_cached_historical(season)`, `fetch_all_historical()` — `statsapi.schedule` for full seasons, filter to `game_type="R"` and `status="Final"`, derive `home_win`.

**4b complete:** `training_data.build_training_set(seasons)`, `load_training_set(seasons)` — join end-of-season team stats (one snapshot per season year) to each game row. No date-accurate rolling stats; this is a known simplification.

**Next:** Phase 4c:
- **4c:** `model.train()`, `evaluate()`, `save_model()`, `load_model()`.

No starting pitcher features in this phase — deferred to future roadmap.

## Tech Stack

| Layer | Library |
|---|---|
| Stats data — primary | pybaseball (FanGraphs) |
| Stats data — fallback | statsapi (MLB Stats API) |
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
| `stats_ingestion.py` | Fetch/cache team batting + pitching stats | `data/raw/stats_YYYY-MM-DD.csv` |
| `historical_ingestion.py` | Fetch/cache historical game results per season | `data/raw/historical_YYYY.csv` |
| `training_data.py` | Join end-of-season stats to game results for model training | `data/processed/training_YYYY-YYYY.csv` |
| `features.py` | Merge odds + stats, engineer features | `data/processed/features_YYYY-MM-DD.csv` |
| `model.py` | Train, evaluate, persist XGBoost model | `models/xgb_YYYY-MM-DD.pkl` + `models/metrics_YYYY-MM-DD.json` |
| `edge_finder.py` | Compute EV, filter odds, flag edges | `data/processed/edges_YYYY-MM-DD.csv` |
| `pipeline.py` | Orchestrate all stages end-to-end | — |

## Stats Ingestion — Source and Schema

`fetch_stats(game_date, force=False)` tries FanGraphs first, falls back to the MLB Stats API. The `data_source` column in the output CSV records which was used (`"fangraphs"` or `"mlb_api"`).

**Always-present columns (both sources):**

| Column | Description |
|---|---|
| `team_abbr` | FanGraphs/MLB abbreviation (e.g. `NYY`) |
| `bat_avg` | Batting average |
| `obp` | On-base percentage |
| `slg` | Slugging percentage |
| `ops` | OBP + SLG |
| `runs_per_game` | R / G — comparable across teams mid-season |
| `era` | Earned run average |
| `whip` | Walks + hits per inning pitched |
| `k_per_9` | Strikeouts per 9 innings |
| `bb_per_9` | Walks per 9 innings |
| `data_source` | `"fangraphs"` or `"mlb_api"` |

**FanGraphs-only columns** (present when `data_source == "fangraphs"`, absent otherwise):

| Column | Description |
|---|---|
| `w_oba` | Weighted on-base average (park/context-neutral) |
| `bat_wrc_plus` | wRC+ — park/league-adjusted run creation, 100 = average |
| `fip` | Fielding-independent pitching (real FanGraphs value) |

**MLB Stats API-only columns** (present when `data_source == "mlb_api"`, absent otherwise):

| Column | Description |
|---|---|
| `fip_computed` | Computed FIP: `(13*HR + 3*BB - 2*K) / IP + 3.15` |

**Important for `features.py`:** Check column presence with `col in df.columns` before using FanGraphs-specific stats. Do not assume they exist.

## Stats Ingestion — Team Name Mapping

`ODDS_NAME_TO_ABBR` in `stats_ingestion.py` maps all 30 Odds API full team names to abbreviations used in the stats CSV. `features.py` imports this dict to translate `home_team`/`away_team` from the odds DataFrame before joining on `team_abbr`.

FanGraphs uses a few non-standard abbreviations that are normalized before output: `WSN→WSH`, `KCR→KC`, `TBR→TB`. The MLB Stats API abbreviations match the standard set directly.

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
- **Conditional stats columns:** FanGraphs-specific columns (`w_oba`, `bat_wrc_plus`, `fip`) are only present when `data_source == "fangraphs"`. Code that consumes stats must guard with `col in df.columns` — do not assume they exist.
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
(or statsapi)                                                         ▼
                                                               train() / load_model()
                                                                      │
                                                                      ▼
                                                               find_edges() ──► edges_YYYY-MM-DD.csv
```

## Execution Model

1. **Jupyter-first (current):** `notebooks/01_exploration.ipynb` — run stages interactively, inspect DataFrames at each step. Phase 4 is split into sections 4a/4b/4c so each subphase can be tested independently.
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

## Features Module

`build_features(game_date)` owns its own data loading — it calls `load_cached_odds` and `load_cached_stats` internally and raises `RuntimeError` (not `FileNotFoundError`) if either cache is absent, with a "run fetch_X() first" message.

The join flow:
1. Map `home_team`/`away_team` full names → abbreviations via `ODDS_NAME_TO_ABBR`
2. Log a warning and drop any games with unmapped team names
3. Drop `data_source` from stats — it's not a model feature
4. Join stats twice: once for home (prefix `home_`), once for away (prefix `away_`)
5. `home_abbr` and `away_abbr` are kept in the output for debugging

FanGraphs-specific stat columns (`w_oba`, `bat_wrc_plus`, `fip`) appear in the features CSV only when the underlying stats CSV contains them. `features.py` does not check for them explicitly — the prefix-rename loop carries whatever is present. Downstream code must guard with `col in df.columns`.

## Running Tests

```bash
pytest tests/ -v
```

41 smoke + integration tests. All pass.

## Roadmap

- [x] Implement `odds_ingestion.fetch_odds()` and `load_cached_odds()`
- [x] Implement `stats_ingestion.fetch_stats()` and `load_cached_stats()`
- [x] Implement `features.build_features()` and `load_features()`
- [x] **4a** — `historical_ingestion.fetch_historical(season)`, `load_cached_historical(season)`, `fetch_all_historical()` via `statsapi`
- [x] **4b** — `training_data.build_training_set(seasons)` joining end-of-season stats to game results
- [ ] **4c** — `model.train()`, `evaluate()`, `save_model()`, `load_model()`
- [ ] Implement `edge_finder.find_edges()`
- [ ] Implement `pipeline.run()`
- [ ] Add `compute_kelly()` to `edge_finder`
- [ ] Add `__main__.py` CLI entry point
- [ ] Add APScheduler for daily runs
