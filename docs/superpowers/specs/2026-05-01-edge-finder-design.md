# Phase 5: Edge Finder Design

**Date:** 2026-05-01
**Scope:** `edge_finder.find_edges()` + `pipeline.run()`

---

## Decisions

| Question | Decision |
|---|---|
| Feature column selection | Use `clf.feature_names_in_` — model is authoritative |
| `game_date` parameter | Explicit third param on `find_edges()` |
| Model discovery in pipeline | Auto-discover latest `xgb_*.pkl` by filename date |
| Home/away transformation | Sequential home pass then away pass, concatenate |

---

## `find_edges(features_df, clf, game_date)`

### Signature

```python
def find_edges(features_df: pd.DataFrame, clf: XGBClassifier, game_date: date) -> pd.DataFrame:
```

### Steps

1. **Feature selection** — extract `clf.feature_names_in_`. Select those columns from `features_df`. Raise `ValueError` if any expected column is absent.
2. **Inference** — `clf.predict_proba(X)[:, 1]` → home win probabilities. Away probability = `1 - home_prob`.
3. **Home pass** — compute `compute_ev(home_prob, home_odds_american)` per game. Filter: `ev > config.EV_THRESHOLD` AND `home_odds_american >= config.MIN_AMERICAN_ODDS`. Build sub-DataFrame with schema below, `bet_side="home"`.
4. **Away pass** — same with away probability and `away_odds_american`, `bet_side="away"`.
5. **Concatenate** home and away edge frames. If empty, log warning and return empty DataFrame with correct column schema.
6. **Persist** to `DATA_PROCESSED_DIR/edges_{game_date}.csv`. Log edge count.
7. **Return** edges DataFrame.

### Output Schema

| Column | Description |
|---|---|
| `game_id` | Odds API game identifier |
| `home_team` | Full home team name |
| `away_team` | Full away team name |
| `bet_side` | `"home"` or `"away"` |
| `american_odds` | The relevant side's American odds |
| `model_prob` | Model-predicted win probability for that side |
| `ev` | Expected value per unit wagered |

### Error Handling

- Raises `ValueError` if `features_df` is missing any column from `clf.feature_names_in_`.
- Never raises on empty results — logs warning, returns empty DataFrame with correct columns.

---

## `pipeline.run(game_date=None)`

### Signature

```python
def run(game_date: date | None = None) -> pd.DataFrame:
```

### Steps

1. Default `game_date` to `date.today()` if `None`.
2. `odds_ingestion.fetch_odds(game_date)` — cache-first.
3. `stats_ingestion.fetch_stats(game_date)` — cache-first, FanGraphs → MLB API fallback.
4. `features.build_features(game_date)`.
5. Glob `MODELS_DIR` for `xgb_*.pkl`, sort lexicographically by filename (YYYY-MM-DD embeds cleanly), load the most recent via `model.load_model()`. Raise `FileNotFoundError` with a helpful message if no models exist.
6. `edge_finder.find_edges(features_df, clf, game_date)`.
7. Return edges DataFrame.

### Error Handling

- `FileNotFoundError` if no models exist in `MODELS_DIR`.
- All other errors propagate from stage modules (no suppression).

---

## Tests (additions to `test_edge_finder.py`)

| Test | Scenario | Assert |
|---|---|---|
| `test_find_edges_returns_edges` | One game with high home EV | One row, `bet_side="home"`, correct schema |
| `test_find_edges_filters_min_odds` | EV > threshold but odds < MIN_AMERICAN_ODDS | Empty result |
| `test_find_edges_empty_when_no_edges` | All games below EV threshold | Empty DataFrame, correct columns, no exception |
| `test_find_edges_both_sides` | Both sides pass filters | Two rows returned |
| `test_find_edges_missing_feature_column` | `features_df` missing a model column | `ValueError` raised |
