# MLB Edge Finder

A portfolio project that identifies positive expected-value (EV) opportunities in MLB moneyline betting markets by comparing model-predicted win probabilities against bookmaker-implied probabilities.

## Tech Stack

Python · pandas · pybaseball · The Odds API · scikit-learn · XGBoost · python-dotenv

## Setup

```bash
# 1. Clone and install
pip install -e .

# 2. Configure secrets
cp .env.template .env
# Edit .env and add your Odds API key

# 3. Launch the starter notebook
jupyter notebook notebooks/01_exploration.ipynb
```

## Project Structure

```
src/mlb_edge_finder/
├── config.py           # env loading, path constants, logging
├── odds_ingestion.py   # fetch moneyline odds (The Odds API)
├── stats_ingestion.py  # fetch team/pitcher stats (pybaseball)
├── features.py         # merge odds + stats into feature DataFrame
├── model.py            # train, evaluate, persist XGBoost model
├── edge_finder.py      # compute EV, flag positive-EV bets
└── pipeline.py         # end-to-end orchestration
```

## Edge Definition

A bet is flagged as an edge when:
- Model EV > 5% (configurable via `EV_THRESHOLD` in `config.py`)
- American odds >= -300 (configurable via `MIN_AMERICAN_ODDS`)

## Roadmap

- [ ] Implement all module stubs
- [ ] Kelly criterion bet sizing (`compute_kelly()` in `edge_finder.py`)
- [ ] CLI entry point (`python -m mlb_edge_finder`)
- [ ] Scheduled daily runs (APScheduler)
