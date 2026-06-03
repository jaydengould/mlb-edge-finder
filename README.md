# MLB Edge Finder

[![CI](https://github.com/jaydengould/mlb-edge-finder/actions/workflows/ci.yml/badge.svg)](https://github.com/jaydengould/mlb-edge-finder/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A portfolio project that identifies positive expected-value (EV) opportunities in MLB moneyline betting markets. Compares XGBoost-predicted win probabilities against bookmaker-implied probabilities and flags bets where the model's edge exceeds a configurable threshold.

## Dashboard

**Live:** https://jaydengould.github.io/mlb-edge-finder/

Updated daily by GitHub Actions at 9:30 AM EDT. Shows today's recommended edges (★ marks high-confidence picks), a 30-day edge history, and an honest evaluation of the model on a **true temporal holdout** (trained on 2019–2024, tested blind on the full 2025 season). The headline chart is a **market-efficiency stress test**: it shows how quickly the model's apparent betting edge disappears as the synthetic market is made more informed — because an edge against a naive market is not a real-world edge. See [Limitations](#limitations--what-id-do-next).

*The color scheme uses the SF Giants' official black (#27251F) and orange (#FD5A1E) — a small personal touch from a lifelong Giants fan.*

## How It Works

1. **Odds ingestion** — fetches today's MLB moneyline odds from [The Odds API](https://the-odds-api.com), deduplicates across bookmakers, and keeps the best line per game.
2. **Stats ingestion** — pulls current-season team batting and pitching stats from FanGraphs (via pybaseball), falling back to the MLB Stats API if FanGraphs is unavailable.
3. **Pitcher ingestion** — fetches individual pitcher season stats and today's probable starters via the MLB Stats API. Pitchers with fewer than 30 IP are excluded from all joins to prevent meaningless ERA/WHIP from small samples.
4. **Feature engineering** — joins odds, team stats, rolling form stats (15-game window), and starting pitcher stats into one row per game.
5. **Inference** — an XGBoost classifier predicts home-team win probability. Probabilities are post-hoc calibrated with isotonic regression on a held-out validation set. EV is computed per side; bets clearing the EV threshold are flagged with a half-Kelly bet size.
6. **Output** — flagged edges are written to `data/processed/edges_YYYY-MM-DD.csv` and printed to the terminal. Edges with EV > 0.40 and a model probability gap over the market of more than 15 percentage points are marked `high_confidence=True` and shown with a ★ on the dashboard.
7. **Automation** — a GitHub Actions workflow runs the full pipeline every morning at 9:30 AM ET and commits the results to `outputs/edges_YYYY-MM-DD.csv` in the repo, with a formatted table in the Actions job summary. A separate snapshot workflow captures time-matched pitcher stats at key season dates (April 30, June 1, July 31) for use in model retraining.
8. **Temporal evaluation** — `temporal_eval.py` trains a fresh model on 2019–2024 and evaluates it blind on the full 2025 season — a true out-of-time holdout. Holdout metrics plus a **market-efficiency sweep** (`sweep_market_efficiency`) are written to `models/temporal_eval_2025.json` and shown on the dashboard. `notebooks/02_backtest.ipynb` contains the original random-split backtest for comparison.

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
# 1. Clone and install in editable mode (pulls all runtime dependencies)
pip install -e .
# For the notebooks + test suite, install the dev extras instead:
# pip install -e ".[dev]"

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
├── backtest.py           # simulate historical performance on held-out test split
└── temporal_eval.py      # out-of-time evaluation: train 2019-2024, test on 2025

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

**Production model performance (2019–2026 test set, random 20% split, n=3,168):**
| Metric | XGBoost |
|---|---|
| Accuracy | 57.2% |
| ROC-AUC | 0.601 |
| Log Loss | 0.687 |

**Temporal holdout performance (trained 2019–2024, tested on full 2025 season, n=2,444):**
| Metric | Value |
|---|---|
| ROC-AUC | 0.563 |
| Accuracy | 55.9% |
| Brier score | 0.245 |
| Win rate — vs naive synthetic market (illustrative) | 64.0% |
| ROI — vs naive synthetic market (illustrative) | +22.3% |

The temporal holdout is the credible evaluation — the model never saw 2025 during training. **The ROC-AUC of 0.563 is the honest headline: a weak-but-positive ranking signal, not a profitable system.** The win rate and ROI are computed against a *naive synthetic 50/50 market* and are illustrative only — a real sportsbook prices the favorite, which erases most of that apparent edge. The market-efficiency sweep on the dashboard quantifies how fragile it is: the edge breaks even once the synthetic market is only **α ≈ 0.43** of the way to being as informed as the model itself — and a real book is *more* informed than this weak model, so in practice the edge is gone. See [Limitations](#limitations--what-id-do-next).

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

**Caveats:** Synthetic −110/−110 odds are a naive baseline; real bookmakers price each game individually, so actual edge frequency against live lines will differ. End-of-season team stats are used for all games in each season (a remaining source of look-ahead bias). Pitcher stats are time-matched (resolved), eliminating the largest source of distribution mismatch.

The dashboard shows the **temporal holdout** results (trained 2019–2024, tested on 2025) and a **market-efficiency sweep** rather than a raw P&L curve. The sweep interpolates each game's synthetic market probability from naive (50/50) toward the model's own prediction and plots betting ROI at each step; the break-even point is the headline. The notebook retains the original random-split P&L results for comparison.

## Limitations & What I'd Do Next

This is a portfolio project on a genuinely hard problem; the value is the end-to-end system and a rigorous, self-critical evaluation — not a claim of beating the market. Known limitations:

- **Synthetic odds, not real lines.** The backtest prices every game against a synthetic market, not real historical bookmaker odds (which require a paid Odds API plan). A real book moves the line toward the favorite, so the naive-market ROI overstates any real edge.
- **The market-efficiency sweep uses the model as its own "sharp market."** Interpolating toward the model's own prediction is a principled proxy for an informed market, not a claim of equivalence to live lines — a real market could be sharper or duller. (A real book is almost certainly *sharper* than this weak model, so the true break-even is below the α ≈ 0.43 measured here.)
- **Weak signal.** ROC-AUC 0.563 is barely above chance. The model also slightly underperforms its own predictions on the bets it makes (predicts ~66%, realizes ~64%), a sign of calibration drift under the 2024→2025 temporal shift.
- **Look-ahead in team stats.** Pitcher stats are time-matched to each game; team batting/pitching stats still use end-of-season values (a remaining, smaller source of leakage).

**What I'd do next:** a real-odds backtest against historical closing lines; time-matched team-stat snapshots (mirroring the pitcher-snapshot approach); and richer features (rest days, travel, ballpark factors, weather).

## Running Tests

```bash
pytest tests/ -v
```

230 smoke + integration tests. All pass.

## Key Engineering Decisions

**Temporal holdout over random split.** The model is evaluated by training on 2019–2024 and testing blind on the full 2025 season — never touching 2025 data during training or calibration. A random 80/20 split leaks future games into the training set and overstates real-world performance; temporal separation is the only honest proxy for forward deployment.

**Time-matched pitcher snapshots.** Early versions joined all training games to end-of-season pitcher stats — a look-ahead that caused the model to see a starter's full-season ERA for games played in April. The fix mirrors how inference works: a GitHub Actions workflow commits pitcher stat snapshots at four dates per season (April 30 / June 1 / July 31 / September 28), and each training game is joined to the latest snapshot strictly before its date. This eliminated the training/inference distribution mismatch that previously produced extreme model probabilities on games the market priced near 50/50.

**Probability calibration.** Raw XGBoost outputs are overconfident — a predicted 70% win probability does not mean a team wins 70% of the time. Without calibration, EV estimates are unreliable by construction. The fix wraps the fitted model in `CalibratedClassifierCV` (isotonic regression, `FrozenEstimator`) fit on a dedicated held-out validation set, producing probabilities that track empirical win rates.

**Market-efficiency sweep instead of a P&L curve.** A backtest against synthetic −110/−110 (50/50) odds flatters any model that correctly identifies favorites — the market already prices them as favorites, so the apparent edge is partly an artifact of the baseline. Rather than headline a misleading ROI number, the dashboard shows a sweep: synthetic market probabilities are interpolated from naive (50/50) toward the model's own predictions, and ROI is plotted at each step. The break-even α (~0.43) is the honest headline — it quantifies exactly how informed the market needs to be before the edge disappears.

**Resilient ingestion with graceful degradation.** Each pipeline stage writes a dated CSV artifact so stages can be run and inspected independently. Network-facing fetches (MLB Stats API, FanGraphs) retry up to 3× with exponential backoff (2s/4s/8s). If all retries fail and a stale cache exists, the pipeline logs a warning and continues rather than crashing the daily workflow — a hard failure on a transient 503 would orphan the day's edge output.
