# CLAUDE.md — MLB Edge Finder

## Project Overview

Portfolio project that finds positive expected-value (EV) opportunities in MLB moneyline betting markets. Compares XGBoost-predicted win probabilities against bookmaker-implied probabilities and flags bets where EV exceeds a configurable threshold.

## Current Phase

**Temporal evaluation complete.** Phases 1–3, 4a–4c, 5, 6, 7, and 8 done. `compute_kelly()`, `__main__.py` CLI, probability calibration, `high_confidence` badge (replaces `prob_flag`), GitHub Actions daily workflow, historical backtest, threshold rebalance (EV_THRESHOLD=0.20, MIN_PROB_EDGE removed), `fetch_historical` resilience, feedback loop, time-matched pitcher snapshots, dashboard, and temporal out-of-time evaluation done.

- `odds_ingestion.fetch_odds()` and `load_cached_odds()` — cache-first, date filtering, live game exclusion, best line across bookmakers.
- `stats_ingestion.fetch_stats()` and `load_cached_stats()` — FanGraphs primary (pybaseball, 3-attempt retry with 2s/4s/8s backoff), MLB Stats API fallback (statsapi package). Output schema varies by source — see stats schema section below.
- `features.build_features(game_date)` and `load_features(game_date)` — loads cached odds and stats, calls `fetch_historical(game_date.year)` for rolling stats, fetches probable starters, maps Odds API team names to abbreviations via `ODDS_NAME_TO_ABBR`, double-joins stats + rolling stats + pitcher stats with `home_`/`away_`/`home_sp_`/`away_sp_` prefixes, writes to `data/processed/features_YYYY-MM-DD.csv`.
- **4a complete:** `historical_ingestion.fetch_historical(season)`, `load_cached_historical(season)`, `fetch_all_historical()` — `statsapi.schedule` for full seasons, filter to `game_type="R"` and `status="Final"`, derive `home_win`. Now also captures `home_starter_name`/`away_starter_name` from the schedule response. `fetch_historical()` retries the MLB Stats API 3× with 2s/4s/8s backoff (`_RETRY_DELAYS`). If all retries fail and a stale cache exists, returns cached data with a warning instead of raising (graceful degradation for CI). Only raises when no cache is present.
- **4b complete:** `training_data.build_training_set(seasons)`, `load_training_set(seasons)` — join end-of-season team stats + rolling stats + pitcher stats to each game row.
- **4c complete:** `model.train()`, `model.train_baseline()`, `model.evaluate()`, `model.save_model()`, `model.load_model()`.
- **5 complete:** `edge_finder.find_edges(features_df, clf, game_date)` — uses `clf.feature_names_in_` to select inference features, runs sequential home/away EV passes, filters by `EV_THRESHOLD` and `MIN_AMERICAN_ODDS`, writes `data/processed/edges_YYYY-MM-DD.csv`. `pipeline.run(game_date)` — orchestrates all five stages end-to-end; auto-discovers latest model by globbing `MODELS_DIR` for `xgb_*.pkl` sorted by filename date.
- **6 complete:** `rolling_stats.py` — `compute_rolling_stats(historical_df, window=15)` (shift-1, for training) and `latest_rolling_stats(historical_df, window=15)` (no shift, for inference). `HISTORICAL_NAME_TO_ABBR` moved here from `training_data.py` (re-exported for backwards compatibility). Adds 8 rolling columns to both training set and daily features: `home_/away_rolling_runs_scored`, `rolling_runs_allowed`, `rolling_win_pct`, `rolling_run_diff`.
- **7 complete:** `pitcher_ingestion.py` — `fetch_pitcher_stats(game_date)` (statsapi-only, playerPool=All, cache at `data/raw/pitcher_stats_YYYY-MM-DD.csv`), `load_cached_pitcher_stats(game_date)`, `fetch_probable_starters(game_date)` (live call, not cached). `historical_ingestion.fetch_historical()` enriched with `home_starter_name`/`away_starter_name`. `training_data._build_season()` and `features.build_features()` both join pitcher stats with `home_sp_*`/`away_sp_*` prefix to avoid collision with team-level stats. `model.NON_FEATURE_COLS` updated with `home_starter_name`, `away_starter_name`, `home_pitcher_id`, `away_pitcher_id`. `pipeline.run()` calls `fetch_pitcher_stats` before `build_features`. Existing cached historical CSVs must be re-fetched with `force=True` to pick up starter name columns.
- **8 complete:** `_HISTORICAL_SEASONS` in `historical_ingestion.py` expanded to `[2019, 2021, 2022, 2023, 2024, 2025]` (2020 excluded — 60-game anomaly). Training set grew from ~2,500 to 15,050 rows. `HISTORICAL_NAME_TO_ABBR` already covered all legacy names (Cleveland Indians, Florida Marlins). `_LEGACY_ABBR_NORMALIZE` (`OAK→ATH`) already handled pre-2025 FanGraphs stats. Notebook `seasons` list updated to match. Model retrained: accuracy 58.8%, ROC-AUC 0.633 on 3,010 test samples.
- **compute_kelly() complete:** `compute_kelly(prob, american_odds) -> float` added to `edge_finder.py`. Half-Kelly formula: `f* = (EV / payout) / 2`, clamped to `[0.0, 1.0]`, returns 0.0 for zero/negative EV. `find_edges()` output gains `kelly_fraction` column. `find_edges()` also guards against empty `features_df` (0 rows when no games today) — returns empty DataFrame instead of raising. `train_baseline()` wrapped in a `sklearn.pipeline.Pipeline` with `SimpleImputer(strategy="median")` to handle NaN values from rolling/pitcher stats (LogisticRegression doesn't support NaN natively).
- **CLI complete:** `src/mlb_edge_finder/__main__.py` — `python -m mlb_edge_finder [--date YYYY-MM-DD] [--force]`. `--date` validated with `date.fromisoformat()`, defaults to today. `--force` bypasses all caches (passed through to `fetch_odds`, `fetch_stats`, `fetch_pitcher_stats`). Prints a formatted table of edges or "No edges found". Exits 0 on success (including no edges), exits 1 on bad date or pipeline exception. `pipeline.run()` gained `force: bool = False` parameter.
- **Probability calibration complete:** `model.calibrate(clf, X_val, y_val)` — wraps a fitted `XGBClassifier` in `CalibratedClassifierCV(FrozenEstimator(clf), method="isotonic")` fit on the held-out validation set. `train()` now does a 60/20/20 split (was 80/20) and returns `(clf, X_val, X_test, y_val, y_test)` — the 20% val split is the calibration input. The saved model (`.pkl`) is the calibrated wrapper, not the raw XGBoost. Uses `FrozenEstimator` (sklearn 1.6+ API) to avoid deprecated `cv="prefit"`.
- **High-confidence badge complete:** `prob_flag` replaced by `high_confidence` column in `find_edges()` output (bool). `True` when `ev > HIGH_CONFIDENCE_EV (0.40)` and `model_prob − market_implied_prob > HIGH_CONFIDENCE_PROB_EDGE (0.15)`. Dashboard renders a ★ badge on high-confidence rows instead of the old ⚠ warning flag — positive signal rather than a caution.
- **GitHub Actions daily workflow complete:** `.github/workflows/daily.yml` — cron `30 13 * * *` UTC (9:30 AM EDT). Checks out repo, installs `requirements.txt` + editable package, runs `python -m mlb_edge_finder` with `ODDS_API_KEY` from repo secrets, promotes `data/processed/edges_YYYY-MM-DD.csv` → `outputs/edges_YYYY-MM-DD.csv` (header-only file written when no edges found), commits and pushes as `github-actions[bot]`. Also writes a markdown edges table to the GitHub Actions job summary via `$GITHUB_STEP_SUMMARY`. `MLB-StatsAPI>=1.7` added to `requirements.txt` (was missing, would have broken CI). `outputs/` directory committed to repo (not gitignored); `data/processed/` remains gitignored.
- **Historical backtest complete:** `backtest.py` — `simulate_market_odds(home_market_prob, vig)` generates synthetic American odds (default -110/-110, 50/50 market, 4.76% vig). `run_backtest(clf, training_df, ...)` replicates the exact 80/20 test split from `model._three_way_split()` (no leakage), runs EV/Kelly filters against synthetic odds, returns per-bet DataFrame with `game_date`, `home_name`, `away_name`, `bet_side`, `american_odds`, `model_prob`, `ev`, `kelly_fraction`, `actual_home_win`, `won`, `pnl`, `cumulative_pnl`. `compute_summary(backtest_df, unit)` returns `n_bets`, `n_wins`, `win_rate`, `total_pnl`, `roi_pct`, `avg_ev`, `max_drawdown`, `sharpe_ratio`. `notebooks/02_backtest.ipynb` is the portfolio artifact — loads saved model + training data, runs backtest, prints summary, plots two-panel cumulative P&L + bet distribution chart. Results on held-out 20% (3,010 games, ~2,370 bets): 60.3% win rate, +15.1% ROI vs synthetic -110/-110 market. High bet rate (~79%) is expected given the naive 50/50 market baseline. 18 new tests (158 total passing).
- **Threshold sweep complete:** `market_implied_prob(american_odds)` added to `edge_finder.py` — converts bookmaker odds to raw implied probability (vig-included). `sweep_thresholds(clf, training_df, ...)` in `backtest.py` — 1D sweep over `ev_threshold` only; `MIN_PROB_EDGE` removed (EV threshold already implies sufficient probability gap). Returns Sharpe-sorted DataFrame with columns `ev_threshold, n_bets, win_rate, roi_pct, sharpe_ratio, avg_bets_per_day`. `run_backtest()` gains `ev_threshold` parameter (defaults to `config.EV_THRESHOLD`). `EV_THRESHOLD=0.20` is the permissive production threshold; `HIGH_CONFIDENCE_EV=0.40` and `HIGH_CONFIDENCE_PROB_EDGE=0.15` gate the ★ badge on the dashboard. Notebook `02_backtest.ipynb` updated with 1D bar-chart sweep display. 15 new tests (173 total passing).
- **Historical ingestion resilience complete:** `fetch_historical()` now retries `statsapi.schedule` up to 3 times with exponential backoff (`_RETRY_DELAYS = [2, 4, 8]`). If all retries fail and a stale local cache (`historical_YYYY.csv`) exists, it logs a warning and returns the cached data instead of raising — prevents the GitHub Actions workflow from failing on transient MLB Stats API outages (e.g. 503 timeouts). Raises `RuntimeError` only when no cache is available. 2 new tests (175 total passing).
- **Current season feedback loop complete:** `feedback.py` — `refresh_historical(season)` force-fetches `historical_YYYY.csv`, `games_since_last_train(historical_df, last_train_date)` counts new games, `run_feedback_loop(season)` orchestrates: refresh → check count → retrain if `new_games >= RETRAIN_THRESHOLD`. Retrains with `_TRAINING_SEASONS = [2019, 2021, 2022, 2023, 2024, 2025, 2026]` using `build_training_set(force=True)`. `config.RETRAIN_THRESHOLD = 15`. `.gitignore` updated: `data/raw/*` + `!data/raw/historical_*.csv` so historical CSVs are committed. Daily workflow gains a `Run feedback loop` step (`continue-on-error: true`) and commits `historical_2026.csv` + new model files alongside edges. 8 new tests (183 total passing).
- **Time-matched pitcher snapshots complete:** `config.MIN_PITCHER_IP = 30` — pitchers below this threshold are excluded from all joins (training and inference). `pitcher_ingestion.fetch_pitcher_snapshot(snapshot_date, force)` — uses the MLB Stats API `byDateRange` stat type to fetch stats through a specific date; falls back to full-season stats for historical seasons where the API no longer supports date-range queries; writes to `data/raw/pitcher_snapshot_YYYY-MM-DD.csv` (distinct from ephemeral `pitcher_stats_*.csv`). `_parse_pitcher_splits` extracted as a shared helper. `fetch_pitcher_stats` gains the 30-IP floor. `training_data._build_season()` replaces the single `fetch_pitcher_stats(date(season, 9, 28))` call with a group join across all available snapshots: loads `pitcher_snapshot_YYYY-MM-DD.csv` files for the four snapshot dates per season, assigns each game to the latest preceding snapshot via `_select_snapshot_date`, and fills NaN for pre-snapshot games. September 28 falls back to `fetch_pitcher_stats` if no snapshot file exists (backward compatibility for seasons predating the workflow). `.gitignore` gains `!data/raw/pitcher_snapshot_*.csv`. `.github/workflows/snapshot.yml` — cron on April 30 / June 1 / July 31 at 14:30 UTC, plus `workflow_dispatch` with optional `snapshot_date` input for backfilling. 25 snapshot files committed for 2019, 2021–2026. Model retrained: accuracy 57.2%, ROC-AUC 0.601 on 3,168 test samples (15,837 training rows). 13 new tests (196 total passing).
- **Dashboard complete:** `generate_site.py` — `generate(outputs_dir, metrics_path, pnl_path, out_path)` reads all `outputs/edges_*.csv` + latest `models/metrics_*.json` + `data/backtest_pnl.json`, writes self-contained `docs/index.html` with Chart.js (CDN). SF Giants color scheme (#27251F/#FD5A1E) — personal touch. GitHub Pages serves from `docs/` on `main`. Daily workflow gains "Generate dashboard" step (`continue-on-error: true`) that commits `docs/index.html` alongside the edges CSV. `backtest.export_pnl_json(backtest_df, summary, path)` writes the static P&L artifact. 16 new tests (212 total passing).
- **Dashboard chart bug fixed:** Both Chart.js IIFEs in `_render_html()` had one extra `}}` pair in the Python f-string, producing an unbalanced `}` in the rendered JavaScript. This caused a silent syntax error that prevented both the edge history bar chart and the backtest P&L line chart from executing. Fixed by removing one `}}` pair from the closing sequence of each `new Chart(...)` call (IIFE1: 10→8 `}` chars; IIFE2: 14→12 `}` chars). Charts confirmed working locally and on the live GitHub Pages site.
- **Temporal out-of-time evaluation complete:** `temporal_eval.py` — `_temporal_split(training_df, holdout_season)` splits by `season` column (train: `season < holdout`, test: `season == holdout`). `_load_training_csv()` globs `training_*.csv` and picks the file with the earliest start season then latest end season (parsed from the filename). `run(holdout_season=2025, force=False)` trains a fresh XGBClassifier on 2019–2024 (75/25 fit/val split for calibration), calibrates with isotonic regression, evaluates on 2025 holdout, runs `simulate_bets()` on holdout, writes `models/temporal_eval_{holdout_season}.json` with model metrics + backtest summary + per-bet P&L series. `backtest.simulate_bets(clf, X_test, y_test, meta_df, ...)` extracted from `run_backtest()` so the bet loop is callable on pre-split data. `generate_site.py` updated: `_load_metrics` / `_load_pnl` replaced by `_load_temporal_eval(models_dir)` (globs `temporal_eval_*.json`); `generate()` signature changed to `(outputs_dir, models_dir, out_path)`; `_render_stats_html` and JS P&L chart read from temporal eval JSON; stats card subtitle shows "Trained XXXX–XXXX · YYYY holdout". Dashboard now shows a single credible evaluation story: temporal holdout rather than random-split. Results on 2025 holdout: 12,606 train rows (2019, 2021–2024), 2,444 test rows, ROC-AUC 0.555, 205 bets, 62.0% win rate, +18.3% ROI, Sharpe 0.197. Run via `python -m mlb_edge_finder.temporal_eval [--holdout-season 2025] [--force]`. `.github/workflows/daily.yml` `prob_flag` bug fixed (header-only CSV now uses `high_confidence`). 8 new tests (220 total passing).
- **ROI reframe + market-efficiency sweep complete:** `backtest._prob_to_american` promoted to module level; `backtest._run_bet_loop(clf, X_test, y_test, meta_df, home_odds, away_odds, ...)` extracted from `simulate_bets` to accept per-game odds; `simulate_bets` delegates to it (behavior unchanged). `backtest.sweep_market_efficiency(clf, X_test, y_test, meta_df, alpha_grid=None, vig=0.0476, ...)` sweeps market efficiency α∈[0,1] — each game's market-implied home prob set to `0.5*(1-α) + model_prob*α`, vig added, converted to per-game American odds, bet loop run — returning `alpha, roi_pct, n_bets, win_rate`. `temporal_eval._break_even_alpha(sweep_df)` linearly interpolates the α where ROI crosses 0 (None if never). `temporal_eval.run()` writes `market_efficiency_sweep` + `break_even_alpha` to the JSON and drops `pnl_series`. `generate_site.py`: `_render_pnl_html`→`_render_efficiency_html` (canvas `efficiency-chart` reading `market_efficiency_sweep`); stats card renamed "Holdout Evaluation", leads with ROC-AUC + break-even α, tags ROI/win-rate as "(naive market)". `model.calibrate()` docstring corrected (FrozenEstimator). README gains a "Limitations & What I'd Do Next" section. Dashboard now tells an honest story: weak positive signal (ROC-AUC 0.563) whose naive-market ROI (+22.3%) collapses as the market gets informed (break-even α≈0.43). Regenerated artifact: 12,599 train rows (2019, 2021–2024), 2,444 test rows. 10 new tests (230 total passing).

**Always update this file at the end of each working session** to reflect completed phases, new conventions, and any changes to the roadmap.

**Always check `notebooks/01_exploration.ipynb` after any implementation** — update section headers, function calls, and comments to match new signatures or behaviour. Specifically: if a function gains a new required argument, add `force=True` where caches need rebuilding, or add mocks where the notebook calls newly added dependencies.

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
| `historical_ingestion.py` | Fetch/cache historical game results per season (incl. probable starters) | `data/raw/historical_YYYY.csv` |
| `pitcher_ingestion.py` | Fetch/cache individual pitcher season stats; fetch time-matched snapshots; fetch probable starters for a date | `data/raw/pitcher_stats_YYYY-MM-DD.csv`, `data/raw/pitcher_snapshot_YYYY-MM-DD.csv` |
| `rolling_stats.py` | Compute rolling per-team stats from historical game results; owns `HISTORICAL_NAME_TO_ABBR` | — |
| `training_data.py` | Join end-of-season stats + rolling stats + time-matched pitcher snapshots to game results for model training | `data/processed/training_YYYY-YYYY.csv` |
| `features.py` | Merge odds + stats + rolling stats, engineer features | `data/processed/features_YYYY-MM-DD.csv` |
| `model.py` | Train, evaluate, persist XGBoost model | `models/xgb_YYYY-MM-DD.pkl` + `models/metrics_YYYY-MM-DD.json` |
| `edge_finder.py` | Compute EV, filter odds, flag edges | `data/processed/edges_YYYY-MM-DD.csv` |
| `pipeline.py` | Orchestrate all stages end-to-end | — |
| `backtest.py` | Simulate historical performance on held-out test split with synthetic odds | — |
| `temporal_eval.py` | Train on prior seasons, evaluate on holdout season, write JSON artifact | `models/temporal_eval_YYYY.json` |

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
EV_THRESHOLD            # 0.20  (permissive filter; EV threshold already implies sufficient prob gap)
MIN_AMERICAN_ODDS       # -300  (skip heavy favorites)
HIGH_CONFIDENCE_EV      # 0.40  (badge threshold — EV bar for high_confidence=True)
HIGH_CONFIDENCE_PROB_EDGE # 0.15 (badge threshold — prob gap bar for high_confidence=True)
RETRAIN_THRESHOLD       # 15   (retrain after this many new games since last model date)
MIN_PITCHER_IP          # 30   (exclude pitchers below this IP from all joins)
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
- **Notebook hygiene:** After every implementation phase, review `notebooks/01_exploration.ipynb` — update function calls to match new signatures, add `force=True` where caches need rebuilding, and note any new NaN behaviour or expected output changes in cell comments.

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
data/raw/         # cached API responses (pitcher_stats_*.csv, stats_*.csv, odds_*.csv)
data/processed/   # engineered features + edge outputs
logs/             # run.log
```

Committed to the repo (not gitignored):
- `models/` — model artifacts (`.pkl` + `.json`) for portfolio visibility
- `data/raw/historical_*.csv` — game results per season, needed by feedback loop
- `data/raw/pitcher_snapshot_*.csv` — time-matched pitcher snapshots, training artifacts committed by `snapshot.yml`

## Training Data Module

`build_training_set(seasons, force=False)` owns its own data loading — it calls `load_cached_historical`, `fetch_stats(date(season, 9, 28))`, and reads pitcher snapshot CSV files from `DATA_RAW_DIR` directly. Raises `RuntimeError` if historical data is missing (with "run fetch_historical(season) first"), and lets `fetch_stats` RuntimeErrors propagate if the API fails.

The join flow (per season):
1. Load `historical_YYYY.csv` via `load_cached_historical(season)`
2. Fetch end-of-season stats via `fetch_stats(date(season, 9, 28))` — cache-first
3. Apply `_LEGACY_ABBR_NORMALIZE` to stats `team_abbr` (e.g. `OAK→ATH`)
4. Map `home_name`/`away_name` → abbreviations via `HISTORICAL_NAME_TO_ABBR`; warn + drop unmatched rows
5. Drop `data_source` — not a model feature
6. Join stats twice: once for home (prefix `home_`), once for away (prefix `away_`)
7. Tag each row with `season` column
8. Join rolling stats (shift-1) with `home_`/`away_` prefix
9. Ensure `home_starter_name`/`away_starter_name` columns exist (NaN if absent from historical CSV)
10. Load all available `pitcher_snapshot_YYYY-MM-DD.csv` files for `_PITCHER_SNAPSHOT_MONTH_DAYS = [(4,30),(6,1),(7,31),(9,28)]`. For Sept 28, falls back to `fetch_pitcher_stats(date(season, 9, 28))` if no file exists.
11. Assign each game to its snapshot via `_select_snapshot_date` (latest snapshot strictly before `game_date`). Pre-snapshot games get NaN pitcher stats.
12. Join pitcher stats per snapshot group with `home_sp_*`/`away_sp_*` prefix (left join on pitcher name, NaN if no starter or below 30 IP)
12. Concatenate all seasons, write to `data/processed/training_{min}-{max}.csv`

`HISTORICAL_NAME_TO_ABBR` always uses **current** franchise abbreviations (e.g. "Oakland Athletics" → "ATH", not "OAK") for consistency with inference-time features. Covers current names (2022+) and legacy names (Cleveland Indians, Florida Marlins, Tampa Bay Devil Rays, Montreal Expos).

`_LEGACY_ABBR_NORMALIZE` maps FanGraphs abbreviations that changed between seasons to current equivalents. Currently: `{"OAK": "ATH"}`.

## Model Module

`model.py` trains an XGBoost classifier and a logistic regression baseline to predict home-team win probability.

**Constants:**
- `TARGET_COL = "home_win"` — binary label column
- `NON_FEATURE_COLS` — metadata columns dropped before training (`game_date`, `home_name`, `away_name`, `home_score`, `away_score`, `home_abbr`, `away_abbr`, `season`, `home_win`, `home_starter_name`, `away_starter_name`, `home_pitcher_id`, `away_pitcher_id`). Any column not in this list is treated as a feature.

**Key design decisions:**
- `_split(features_df)` — private helper; 80/20 stratified split, `random_state=42`. Used by `train_baseline()` only.
- `_three_way_split(features_df)` — private helper; 60/20/20 stratified split. Used by `train()`.
- `train(features_df)` → `(XGBClassifier, X_val, X_test, y_val, y_test)` — 60% fits XGBoost, 20% returned as calibration val set, 20% held for evaluation. Uses `config.XGB_N_ESTIMATORS` and `config.XGB_MAX_DEPTH`.
- `calibrate(clf, X_val, y_val)` → `CalibratedClassifierCV` — wraps a fitted clf with isotonic regression via `FrozenEstimator` (sklearn 1.6+, replaces deprecated `cv="prefit"`). Fit on the held-out val set only; the underlying model is not retrained.
- `train_baseline(features_df)` → `(Pipeline, X_test, y_test)` — uses `_split()` (80/20); diagnostic only, never persisted.
- `evaluate(clf, X_test, y_test)` → dict — duck-typed, works for XGBClassifier and CalibratedClassifierCV. Keys: `accuracy`, `roc_auc`, `log_loss`, `brier_score`, `n_test_samples`, `xgb_n_estimators`, `xgb_max_depth`. XGBoost-specific keys are `None` for other classifiers.
- `save_model(clf, metrics, game_date)` — writes `xgb_YYYY-MM-DD.pkl` and `metrics_YYYY-MM-DD.json` to `MODELS_DIR`. The persisted model is typically the calibrated wrapper.
- `load_model(game_date)` → classifier — raises `FileNotFoundError` if missing.

Only the calibrated XGBoost model is persisted. The logistic regression baseline is used at training time for comparison only.

**Observed baseline performance (2023–2025, end-of-season stats):** Logistic regression slightly outperforms XGBoost on static season-average features — expected, as aggregate stats lack temporal signal. Rolling window features (future roadmap) should reverse this.

## Features Module

`build_features(game_date)` owns its own data loading — it calls `load_cached_odds` and `load_cached_stats` internally and raises `RuntimeError` (not `FileNotFoundError`) if either cache is absent, with a "run fetch_X() first" message.

The join flow:
1. Map `home_team`/`away_team` full names → abbreviations via `ODDS_NAME_TO_ABBR`
2. Log a warning and drop any games with unmapped team names
3. Drop `data_source` from stats — it's not a model feature
4. Join stats twice: once for home (prefix `home_`), once for away (prefix `away_`)
5. Join rolling stats by `team_abbr` (left join, NaN for teams with no history)
6. Call `fetch_probable_starters(game_date)` — merge on `[home_abbr, away_abbr]` to add `home_starter_name`/`away_starter_name`
7. Join pitcher stats twice with `home_sp_*`/`away_sp_*` prefix (left join on pitcher name, NaN if no probable starter)
8. `home_abbr`, `away_abbr`, `home_starter_name`, `away_starter_name` are kept for debugging

FanGraphs-specific stat columns (`w_oba`, `bat_wrc_plus`, `fip`) appear in the features CSV only when the underlying stats CSV contains them. `features.py` does not check for them explicitly — the prefix-rename loop carries whatever is present. Downstream code must guard with `col in df.columns`.

## Running Tests

```bash
pytest tests/ -v
```

230 smoke + integration tests. All pass.

## Roadmap

- [x] Implement `odds_ingestion.fetch_odds()` and `load_cached_odds()`
- [x] Implement `stats_ingestion.fetch_stats()` and `load_cached_stats()`
- [x] Implement `features.build_features()` and `load_features()`
- [x] **4a** — `historical_ingestion.fetch_historical(season)`, `load_cached_historical(season)`, `fetch_all_historical()` via `statsapi`
- [x] **4b** — `training_data.build_training_set(seasons)` joining end-of-season stats to game results
- [x] **4c** — `model.train()`, `train_baseline()`, `evaluate()`, `save_model()`, `load_model()`
- [x] Implement `edge_finder.find_edges()`
- [x] Implement `pipeline.run()`
- [x] **6 — Rolling window team stats** — `rolling_stats.py` computes 4 rolling features (runs_scored, runs_allowed, win_pct, run_diff) from cached historical game results. Joined into both training set and daily features. Window=15, season-only lookback, XGBoost handles NaN for early-season games.
- [x] **7 — Starting pitcher features** — `pitcher_ingestion.py` with statsapi-only fetch. Individual pitcher `era`, `whip`, `k_per_9`, `bb_per_9`, `ip`, `fip_computed`. `home_sp_*`/`away_sp_*` prefix in both training and inference. Probable starters from `statsapi.schedule`. Historical CSVs enriched with `home_starter_name`/`away_starter_name`.
- [x] **8 — Expand training seasons** — `_HISTORICAL_SEASONS = [2019, 2021, 2022, 2023, 2024, 2025]`. Training set: 15,050 rows. Model retrained (accuracy 58.8%, ROC-AUC 0.633).
- [x] Add `compute_kelly()` to `edge_finder` — half-Kelly sizing, `kelly_fraction` column in `find_edges()` output.
- [x] Add `__main__.py` CLI entry point — `python -m mlb_edge_finder [--date YYYY-MM-DD] [--force]`.
- [x] Add probability calibration — `model.calibrate(clf, X_val, y_val)` wraps XGBoost with isotonic `CalibratedClassifierCV` (FrozenEstimator, sklearn 1.6+). `train()` returns 5-tuple with dedicated val split.
- [x] Add `high_confidence` column to `find_edges()` output — `True` when `ev > HIGH_CONFIDENCE_EV (0.40)` and `model_prob − market_implied_prob > HIGH_CONFIDENCE_PROB_EDGE (0.15)`. Replaces old `prob_flag` warning; dashboard renders ★ badge.
- [x] GitHub Actions daily workflow — `.github/workflows/daily.yml`, cron 9:30 AM EDT, commits edges to `outputs/`, job summary table via `$GITHUB_STEP_SUMMARY`.
- [x] Historical backtest — `backtest.py` with `simulate_market_odds`, `run_backtest`, `compute_summary`. `notebooks/02_backtest.ipynb` with cumulative P&L curve. Results: 60.3% win rate, +15.1% ROI on held-out 20% test split vs -110/-110 synthetic market.
- [x] Threshold rebalance — `market_implied_prob()` in `edge_finder.py`. `MIN_PROB_EDGE` removed; `EV_THRESHOLD` lowered to 0.20 (permissive). `sweep_thresholds()` now 1D (ev_threshold only), returns Sharpe-sorted DataFrame. `HIGH_CONFIDENCE_EV=0.40` and `HIGH_CONFIDENCE_PROB_EDGE=0.15` gate the ★ badge. Notebook updated with bar-chart sweep display.
- [x] Historical ingestion resilience — `fetch_historical()` retries `statsapi.schedule` 3× (2s/4s/8s backoff). On total failure, falls back to stale cache if present; raises only when no cache exists. Prevents GitHub Actions pipeline failures on transient MLB Stats API outages.
- [x] Current season feedback loop — `feedback.py` with `refresh_historical()`, `games_since_last_train()`, `run_feedback_loop()`. Retrains every 15 new games. Daily workflow commits `historical_2026.csv` and new model files. `RETRAIN_THRESHOLD=15` in config.
- [x] Time-matched pitcher snapshots — `MIN_PITCHER_IP=30` in config; `fetch_pitcher_snapshot(snapshot_date, force)` with `byDateRange` API and full-season fallback; `_parse_pitcher_splits` extracted as shared helper; `_build_season` multi-snapshot group join with `_select_snapshot_date`; `snapshot.yml` workflow cron on April 30 / June 1 / July 31; 25 snapshot files backfilled for 2019, 2021–2026. Model retrained: accuracy 57.2%, ROC-AUC 0.601 (15,837 training rows, 196 tests passing).
- [x] Dashboard / web UI — `generate_site.py` writes self-contained `docs/index.html` with Chart.js; GitHub Pages serves from `docs/` on `main`; daily workflow commits `docs/index.html` alongside edges CSV; `backtest.export_pnl_json()` writes static P&L artifact. 212 tests passing.

## Future Work (priority order)

1. **Real-odds backtest** — replace synthetic −110/−110 market odds with actual historical bookmaker lines from The Odds API historical endpoint (`GET /v4/sports/baseball_mlb/odds-history?date=...`). Requires a paid Odds API plan (free tier returns 401). Produces a far more credible P&L curve. Cache fetched odds locally like all other raw data.

2. **Out-of-time model evaluation** — retrain the model excluding 2025 entirely, then evaluate on 2025 as a true temporal holdout. Avoids the random-split leakage concern in the current backtest and better simulates forward performance.

3. **Time-matched team stats** — apply the same snapshot approach to team batting/pitching stats (`stats_ingestion`). Currently all training games for a season use September 28 end-of-season team stats (look-ahead bias). The fix mirrors the pitcher snapshot pattern: capture team stats at the same four dates per season and join each game to its nearest preceding snapshot.

4. **Additional features** — rest days (days since last game), travel distance, ballpark factors (park-adjusted run environment), weather (temperature, wind, dome vs outdoor). Each likely adds a small but real signal that compounds with existing features.
