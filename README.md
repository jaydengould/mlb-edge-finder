# MLB Edge Finder

A portfolio project that identifies positive expected-value (EV) opportunities in MLB moneyline betting markets. Compares XGBoost-predicted win probabilities against bookmaker-implied probabilities and flags bets where the model's edge exceeds a configurable threshold.

## Dashboard

**Live:** https://jaydengould.github.io/mlb-edge-finder/

Updated daily by GitHub Actions at 9:30 AM EDT. Shows today's recommended edges (★ marks high-confidence picks), a 30-day edge history, and the model's validated backtest performance on held-out test data.

*The color scheme uses the SF Giants' official black (#27251F) and orange (#FD5A1E) — a small personal touch from a lifelong Giants fan.*

## How It Works

1. **Odds ingestion** — fetches today's MLB moneyline odds from [The Odds API](https://the-odds-api.com), deduplicates across bookmakers, and keeps the best line per game.
2. **Stats ingestion** — pulls current-season team batting and pitching stats from FanGraphs (via pybaseball), falling back to the MLB Stats API if FanGraphs is unavailable.
3. **Pitcher ingestion** — fetches individual pitcher season stats and today's probable starters via the MLB Stats API. Pitchers with fewer than 30 IP are excluded from all joins to prevent meaningless ERA/WHIP from small samples.
4. **Feature engineering** — joins odds, team stats, rolling form stats (15-game window), and starting pitcher stats into one row per game.
5. **Inference** — an XGBoost classifier predicts home-team win probability. Probabilities are post-hoc calibrated with isotonic regression on a held-out validation set. EV is computed per side; bets clearing the EV threshold are flagged with a half-Kelly bet size.
6. **Output** — flagged edges are written to `data/processed/edges_YYYY-MM-DD.csv` and printed to the terminal. Edges with EV > 0.40 and a model probability gap over the market of more than 15 percentage points are marked `high_confidence=True` and shown with a ★ on the dashboard.
7. **Automation** — a GitHub Actions workflow runs the full pipeline every morning at 9:30 AM ET and commits the results to `outputs/edges_YYYY-MM-DD.csv` in the repo, with a formatted table in the Actions job summary. A separate snapshot workflow captures time-matched pitcher stats at key season dates (April 30, June 1, July 31) for use in model retraining.
8. **Backtest** — `notebooks/02_backtest.ipynb` simulates historical performance on the held-out 20% test split using synthetic −110/−110 market odds, producing a cumulative P&L curve and summary statistics.

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

Python 3.10+ required.

## Setup

```bash
# 1. Clone and install in editable mode
pip install -e .

# 2. Configure secrets
cp .env.template .env
# Edit .env — add your Odds API key:
# ODDS_API_KEY=your_key_here

# 3. Fetch historical data and train the model (one-time)
jupyter notebook notebooks/01_exploration.ipynb

# 4. Run the pipeline
python -m mlb_edge_finder
```

## CLI

```bash
# Run for today (uses cached data if available)
python -m mlb_edge_finder

# Run for a specific date
python -m mlb_edge_finder --date 2026-05-12

# Re-fetch all data, bypassing caches
python -m mlb_edge_finder --force

# Both
python -m mlb_edge_finder --date 2026-05-12 --force
```

**Exit codes:** 0 on success (including "no edges found"), 1 on invalid date or pipeline failure.

**Example output:**
```
Found 2 edge(s) for 2026-05-12:

  home_team         away_team        bet_side  american_odds  model_prob    ev  kelly_fraction  high_confidence
  New York Yankees  Boston Red Sox   home            +130       0.682   0.107           0.041        False
  Houston Astros    Texas Rangers    away            +115       0.621   0.064           0.028        False
```

`high_confidence=True` marks the strongest edges (EV > 0.40 and model probability gap over market > 15pp). These appear with a ★ prefix on the dashboard.

## Project Structure

```
src/mlb_edge_finder/
├── __main__.py           # CLI entry point (python -m mlb_edge_finder)
├── config.py             # env loading, path constants, thresholds
├── odds_ingestion.py     # fetch/cache moneyline odds (The Odds API)
├── stats_ingestion.py    # fetch/cache team batting + pitching stats
├── historical_ingestion.py  # fetch/cache historical game results per season
├── pitcher_ingestion.py  # fetch/cache individual pitcher stats + probable starters
├── rolling_stats.py      # compute 15-game rolling form stats per team
├── features.py           # merge all data sources into per-game feature rows
├── training_data.py      # build labeled training set from historical data
├── model.py              # train, calibrate, evaluate, persist XGBoost model
├── edge_finder.py        # compute EV + Kelly fraction, flag positive-EV bets
├── pipeline.py           # end-to-end orchestration
└── backtest.py           # simulate historical performance on held-out test split

notebooks/
├── 01_exploration.ipynb  # interactive pipeline exploration and model training
└── 02_backtest.ipynb     # historical backtest with cumulative P&L curve

.github/workflows/
├── daily.yml             # GitHub Actions cron — runs pipeline at 9:30 AM ET daily
└── snapshot.yml          # captures pitcher stat snapshots on April 30, June 1, July 31

outputs/                  # committed edge CSVs written by the daily workflow
```

## Automation

A GitHub Actions workflow runs the full pipeline every morning at **9:30 AM ET** (during the regular season) without the MacBook needing to be on.

- **Schedule:** `30 13 * * *` UTC — fires daily using GitHub's hosted runners
- **Output:** commits `outputs/edges_YYYY-MM-DD.csv` back to the repo — browse the full history on GitHub, which renders CSV files as formatted tables
- **Job summary:** each run prints the edges table (or "No edges found today.") directly in the [Actions tab](https://github.com/jaydengould/mlb-edge-finder/actions) — no need to open any file
- **No edges days:** still produce a header-only CSV so every run leaves a visible commit
- **Secret required:** `ODDS_API_KEY` set in repo Settings → Secrets → Actions

To trigger a manual run at any time: Actions tab → **Daily MLB Edge Finder** → **Run workflow**.

## Model

Trained on 15,837 regular-season games (2019, 2021–2026; 2020 excluded — 60-game anomaly).

**Training split:** 60% fit / 20% calibration validation / 20% test (all stratified).

**Features (40 total):**
- Team batting: `bat_avg`, `obp`, `slg`, `ops`, `runs_per_game`
- Team pitching: `era`, `whip`, `k_per_9`, `bb_per_9`
- Rolling form (15-game window, shift-1 for training): `rolling_runs_scored`, `rolling_runs_allowed`, `rolling_win_pct`, `rolling_run_diff`
- Starting pitcher: `era`, `whip`, `k_per_9`, `bb_per_9`, `ip`, `fip_computed` (home/away prefixed; only pitchers with ≥ 30 IP)

**Time-matched pitcher snapshots:** Training pitcher stats are joined from the most recent snapshot *strictly before* each game's date — not end-of-season stats. Four snapshot dates per season (April 30 / June 1 / July 31 / September 28) are captured automatically by the snapshot workflow and committed to the repo. Games before the first snapshot, and probable starters below 30 IP, receive NaN pitcher stats (consistent with ~73% of training rows which already have no matched starter). This eliminates the training/inference distribution mismatch that previously caused the model to produce extreme probabilities (>80%) on edges that the market priced near 50/50.

**Probability calibration:** After training, the raw XGBoost model is wrapped with `CalibratedClassifierCV` (isotonic regression, `FrozenEstimator`) fit on the held-out 20% calibration set. This corrects the model's tendency to produce overconfident probabilities, which is critical for EV estimates to be meaningful.

**Performance (2019–2026 test set, n=3,168):**
| Metric | XGBoost |
|---|---|
| Accuracy | 57.2% |
| ROC-AUC | 0.601 |
| Log Loss | 0.687 |

The slight reduction in headline metrics vs the previous model reflects more realistic training data — early-season games now correctly have NaN pitcher features rather than borrowing end-of-season stats. The practical benefit is that inference probabilities stay in a credible range (55–75%), keeping EV estimates meaningful.

## Edge Definition

A bet is flagged when both conditions hold:

```
EV > EV_THRESHOLD          # default 20% — configurable in config.py
american_odds >= MIN_AMERICAN_ODDS   # default -300 — skips heavy favorites
```

`EV_THRESHOLD` was set by a Sharpe-optimal 1D sweep over the held-out test split (see Backtest section). At standard MLB moneyline odds, a meaningful EV gap already implies the model disagrees significantly with the market — an explicit `MIN_PROB_EDGE` filter is not required.

**High-confidence badge:** edges with EV > `HIGH_CONFIDENCE_EV` (0.40) *and* `model_prob − market_implied_prob > HIGH_CONFIDENCE_PROB_EDGE` (0.15) receive `high_confidence=True` and are shown with ★ on the dashboard. These are the strongest signals, not a separate bet filter.

**EV formula:**
```python
# Favorite (negative odds)
EV = prob * (100 / abs(odds)) - (1 - prob)

# Underdog (positive odds)
EV = prob * (odds / 100) - (1 - prob)
```

**Market-implied probability** (used for high-confidence badge and display):
```python
# Negative odds (favourite): -110 → 110/210 = 52.4%
market_implied_prob = abs(odds) / (abs(odds) + 100)

# Positive odds (underdog): +130 → 100/230 = 43.5%
market_implied_prob = 100 / (odds + 100)
```

**Kelly bet sizing (half-Kelly):**
```python
kelly_fraction = (EV / payout) / 2   # half of full Kelly, clamped to [0.0, 1.0]
```

## Backtest

`notebooks/02_backtest.ipynb` validates the model against the held-out 20% test split (3,010 games never seen during training or calibration). It includes a 1D threshold sweep over `EV_THRESHOLD` to find the Sharpe-optimal value.

**Method:** synthetic market odds of −110/−110 (50/50 even market, 4.76% vig). EV is computed against these synthetic lines for each game in the test set. Bets are flagged when `EV > EV_THRESHOLD`.

**Threshold sweep:** `sweep_thresholds()` evaluates `EV_THRESHOLD` across a range of values and returns a DataFrame with columns `ev_threshold`, `n_bets`, `win_rate`, `roi_pct`, `sharpe_ratio`, `avg_bets_per_day`, sorted by Sharpe ratio. The Sharpe-optimal value sets the default `EV_THRESHOLD=0.20` in `config.py`.

**Caveats:** Synthetic −110/−110 odds are a naive baseline; real bookmakers price each game individually, so actual edge frequency against live lines will differ. End-of-season team stats are used for all games in each season (a remaining source of look-ahead bias), which likely overstates performance. Pitcher stats are now time-matched (resolved), eliminating the largest source of distribution mismatch.

## Running Tests

```bash
pytest tests/ -v
```

208 smoke + integration tests. All pass.

## Roadmap

- [x] Odds ingestion — cache-first, best line across bookmakers, live game exclusion
- [x] Stats ingestion — FanGraphs primary, MLB Stats API fallback
- [x] Feature engineering — odds + stats join with home/away prefixes
- [x] Historical ingestion — full regular-season game results via MLB Stats API
- [x] Training data — end-of-season stats + rolling stats + pitcher stats joined to game results
- [x] Model — XGBoost classifier + logistic regression baseline, persist with metrics JSON
- [x] Edge finder — EV computation, threshold filtering, edges CSV output
- [x] Pipeline — end-to-end orchestration with auto model discovery
- [x] Rolling window team stats — 15-game trailing form (runs, win %, run diff)
- [x] Starting pitcher features — individual ERA/WHIP/K9/BB9/IP/FIP for probable starters
- [x] Expand training seasons — 2019, 2021–2025 (15,050 games)
- [x] Kelly criterion bet sizing — half-Kelly `kelly_fraction` column in edge output
- [x] CLI — `python -m mlb_edge_finder [--date YYYY-MM-DD] [--force]`
- [x] Probability calibration — isotonic regression via `CalibratedClassifierCV` fit on held-out val set; `model.calibrate(clf, X_val, y_val)`
- [x] High-confidence badge — `high_confidence=True` in edge output when EV > 0.40 and model prob gap over market > 15pp; shown as ★ on the dashboard
- [x] GitHub Actions daily automation — cron 9:30 AM ET, commits edges to `outputs/`, job summary table in Actions UI
- [x] Historical backtest — `backtest.py` + `notebooks/02_backtest.ipynb`; backtest P&L on held-out 20% test split vs synthetic −110/−110 market
- [x] Threshold sweep — `sweep_thresholds()` 1D sweep over `ev_threshold`; Sharpe-optimal value sets `EV_THRESHOLD=0.20` in config; returns `ev_threshold, n_bets, win_rate, roi_pct, sharpe_ratio, avg_bets_per_day`
- [x] Historical ingestion resilience — `fetch_historical()` retries the MLB Stats API 3× (2s/4s/8s backoff) before failing; if all retries fail and a stale cache exists, returns cached data with a warning instead of crashing the pipeline
- [x] Current season feedback loop — `feedback.py` refreshes `historical_2026.csv` daily and retrains the model every 15 new games; workflow commits historical data and new model files alongside edges
- [x] Time-matched pitcher snapshots — `fetch_pitcher_snapshot()` captures stats through a specific date via the MLB Stats API `byDateRange` endpoint; `_build_season` joins each training game to the latest preceding snapshot (April 30 / June 1 / July 31 / September 28) instead of end-of-season stats; `MIN_PITCHER_IP=30` floor applied at all join points; `snapshot.yml` workflow auto-commits snapshot files on schedule
- [x] Dashboard — self-contained `docs/index.html` generated by `generate_site.py`; served via GitHub Pages; updated daily by the Actions workflow alongside the edges CSV; shows today's edges (★ on high-confidence rows), 30-day history bar chart, and backtest P&L line chart with SF Giants color scheme
- [x] Dashboard chart fix — repaired unbalanced `}}` pairs in the Chart.js f-string template that produced a silent JS syntax error, preventing both canvases from rendering
