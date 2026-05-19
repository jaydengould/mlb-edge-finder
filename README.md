# MLB Edge Finder

A portfolio project that identifies positive expected-value (EV) opportunities in MLB moneyline betting markets. Compares XGBoost-predicted win probabilities against bookmaker-implied probabilities and flags bets where the model's edge exceeds a configurable threshold.

## How It Works

1. **Odds ingestion** — fetches today's MLB moneyline odds from [The Odds API](https://the-odds-api.com), deduplicates across bookmakers, and keeps the best line per game.
2. **Stats ingestion** — pulls current-season team batting and pitching stats from FanGraphs (via pybaseball), falling back to the MLB Stats API if FanGraphs is unavailable.
3. **Pitcher ingestion** — fetches individual pitcher season stats and today's probable starters via the MLB Stats API.
4. **Feature engineering** — joins odds, team stats, rolling form stats (15-game window), and starting pitcher stats into one row per game.
5. **Inference** — an XGBoost classifier predicts home-team win probability. Probabilities are post-hoc calibrated with isotonic regression on a held-out validation set. EV is computed per side; bets exceeding the threshold are flagged with a half-Kelly bet size.
6. **Output** — flagged edges are written to `data/processed/edges_YYYY-MM-DD.csv` and printed to the terminal. Any edge where `model_prob > 0.80` is marked with `prob_flag=True` for manual review.

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

  home_team         away_team        bet_side  american_odds  model_prob    ev  kelly_fraction  prob_flag
  New York Yankees  Boston Red Sox   home            +130       0.682   0.107           0.041      False
  Houston Astros    Texas Rangers    away            +115       0.621   0.064           0.028      False
```

`prob_flag=True` marks rows where `model_prob > 0.80` — review these manually before acting, as extreme probabilities can indicate a feature outlier rather than a genuine edge.

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
├── model.py              # train, evaluate, persist XGBoost model
├── edge_finder.py        # compute EV + Kelly fraction, flag positive-EV bets
└── pipeline.py           # end-to-end orchestration
```

## Model

Trained on 15,050 regular-season games (2019, 2021–2025; 2020 excluded — 60-game anomaly).

**Training split:** 60% fit / 20% calibration validation / 20% test (all stratified).

**Features (40 total):**
- Team batting: `bat_avg`, `obp`, `slg`, `ops`, `runs_per_game`
- Team pitching: `era`, `whip`, `k_per_9`, `bb_per_9`
- Rolling form (15-game window, shift-1 for training): `rolling_runs_scored`, `rolling_runs_allowed`, `rolling_win_pct`, `rolling_run_diff`
- Starting pitcher: `era`, `whip`, `k_per_9`, `bb_per_9`, `ip`, `fip_computed` (home/away prefixed)

**Probability calibration:** After training, the raw XGBoost model is wrapped with `CalibratedClassifierCV` (isotonic regression, `FrozenEstimator`) fit on the held-out 20% calibration set. This corrects the model's tendency to produce overconfident probabilities (e.g. 90%+ for games that are realistically 60/40), which is critical for EV estimates to be meaningful.

**Performance (2019–2025 test set, n=3,010):**
| Metric | XGBoost | Logistic Regression baseline |
|---|---|---|
| Accuracy | 58.8% | 59.2% |
| ROC-AUC | 0.633 | 0.625 |
| Log Loss | 0.669 | 0.667 |

XGBoost and the logistic regression baseline perform similarly on aggregate season-average features — expected, since these features lack temporal signal within a season. Rolling window stats improve on this; further signal could come from rest days, travel, ballpark factors, or weather.

## Edge Definition

A bet is flagged when both conditions hold:

```
EV > EV_THRESHOLD        # default 5% — configurable in config.py
american_odds >= MIN_AMERICAN_ODDS  # default -300 — skips heavy favorites
```

**EV formula:**
```python
# Favorite (negative odds)
EV = prob * (100 / abs(odds)) - (1 - prob)

# Underdog (positive odds)
EV = prob * (odds / 100) - (1 - prob)
```

**Kelly bet sizing (half-Kelly):**
```python
kelly_fraction = (EV / payout) / 2   # half of full Kelly, clamped to [0.0, 1.0]
```

## Running Tests

```bash
pytest tests/ -v
```

140 smoke + integration tests. All pass.

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
- [x] High-probability flag — `prob_flag=True` in edge output when `model_prob > 0.80`
- [ ] APScheduler for daily automated runs
