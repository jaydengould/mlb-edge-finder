# Phase 4c — model.py Design

**Date:** 2026-04-29  
**Scope:** Implement `train()`, `train_baseline()`, `evaluate()`, `save_model()`, `load_model()` in `src/mlb_edge_finder/model.py`.

---

## Overview

`model.py` trains an XGBoost classifier to predict home-team win probability, evaluates it against a logistic regression baseline, and persists the XGBoost model to disk. It is consumed by the notebook (Phase 4c section) and eventually by `pipeline.run()`.

---

## Data Preparation

### `NON_FEATURE_COLS` constant

A module-level list of columns to exclude from the feature matrix:

```python
NON_FEATURE_COLS = [
    "game_date", "home_name", "away_name",
    "home_score", "away_score", "home_abbr", "away_abbr",
    "season", TARGET_COL,
]
```

Any column not in this list becomes a feature. Optional FanGraphs columns (`w_oba`, `bat_wrc_plus`, `fip`) are picked up automatically when present — no explicit allowlist needed.

### `_split(features_df)` — private helper

```python
def _split(features_df):
    -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
```

1. Raises `ValueError` if `TARGET_COL` not in `features_df.columns`.
2. Raises `FileNotFoundError` if `features_df` is empty.
3. Drops `NON_FEATURE_COLS` (ignoring any that are absent) to produce `X`.
4. Extracts `y = features_df[TARGET_COL]`.
5. Calls `train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)`.
6. Returns `(X_train, X_test, y_train, y_test)`.

Stratifying on `y` preserves the home-win class ratio in both splits.

---

## Training Functions

### `train(features_df)` → `(XGBClassifier, X_test, y_test)`

1. Calls `_split(features_df)`.
2. Instantiates `XGBClassifier(n_estimators=config.XGB_N_ESTIMATORS, max_depth=config.XGB_MAX_DEPTH, eval_metric="logloss", random_state=42)`.
3. Fits on `(X_train, y_train)`.
4. Logs feature count and training set shape.
5. Returns `(clf, X_test, y_test)`.

### `train_baseline(features_df)` → `(LogisticRegression, X_test, y_test)`

1. Calls `_split(features_df)` — same `random_state=42`, identical test split to `train()`.
2. Instantiates `LogisticRegression(max_iter=1000, random_state=42)`.
3. Fits on `(X_train, y_train)`.
4. Logs training set shape.
5. Returns `(clf, X_test, y_test)`.

Both functions surface `ValueError` / `FileNotFoundError` from `_split()` without catching.

---

## Evaluation

### `evaluate(clf, X_test, y_test)` → `dict[str, Any]`

Works with any sklearn-compatible classifier (XGBoost or LogisticRegression).

Returned dict keys:

| Key | Source |
|---|---|
| `accuracy` | `accuracy_score(y_test, clf.predict(X_test))` |
| `roc_auc` | `roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])` |
| `log_loss` | `log_loss(y_test, clf.predict_proba(X_test)[:, 1])` |
| `brier_score` | `brier_score_loss(y_test, clf.predict_proba(X_test)[:, 1])` |
| `n_test_samples` | `len(y_test)` |
| `xgb_n_estimators` | `getattr(clf, "n_estimators", None)` |
| `xgb_max_depth` | `getattr(clf, "max_depth", None)` |

Using `getattr(..., None)` means the same function returns clean output for both model types without branching — logistic regression just gets `None` for the XGBoost-specific keys.

---

## Persistence

### `save_model(clf, metrics, game_date)`

1. Creates `config.MODELS_DIR` if it doesn't exist.
2. Writes `xgb_YYYY-MM-DD.pkl` via `pickle.dump`.
3. Writes `metrics_YYYY-MM-DD.json` via `json.dump`.
4. Logs both file paths.

### `load_model(game_date)` → `XGBClassifier`

1. Builds path: `config.MODELS_DIR / f"xgb_{game_date}.pkl"`.
2. Raises `FileNotFoundError` with a descriptive message if the file doesn't exist.
3. Unpickles and returns the classifier.

Only XGBoost models are persisted. The logistic regression baseline is a diagnostic tool used during training evaluation only.

---

## Error Handling

| Condition | Raised by | Exception |
|---|---|---|
| `TARGET_COL` missing from DataFrame | `_split()` | `ValueError` |
| Empty DataFrame passed | `_split()` | `FileNotFoundError` |
| Model file not found on disk | `load_model()` | `FileNotFoundError` |

---

## Testing

Existing `test_model.py` has four signature smoke tests (all pass). New tests to add:

- `train()` returns a fitted `XGBClassifier` and correctly shaped `X_test`/`y_test`
- `train_baseline()` returns a fitted `LogisticRegression` with the same test split shape
- `evaluate()` returns a dict with all expected keys for both model types
- `save_model()` + `load_model()` round-trip produces equivalent predictions
- `train()` raises `ValueError` when `TARGET_COL` is absent
- `train()` raises `FileNotFoundError` when DataFrame is empty

---

## What's Out of Scope

- Hyperparameter tuning / cross-validation (future roadmap)
- Starting pitcher features (deferred per CLAUDE.md)
- Persisting the baseline model
- Kelly sizing (added to `edge_finder`, not `model`)
