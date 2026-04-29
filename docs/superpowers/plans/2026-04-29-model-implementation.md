# Phase 4c — model.py Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `train()`, `train_baseline()`, `evaluate()`, `save_model()`, and `load_model()` in `model.py`, replacing all `raise NotImplementedError` stubs.

**Architecture:** A private `_split()` helper owns the 80/20 stratified train/test split with a fixed random seed. `train()` and `train_baseline()` both call `_split()` so they produce identical test sets for fair metric comparison. `evaluate()` is duck-typed and works with any sklearn-compatible classifier. Only the XGBoost model is persisted to disk.

**Tech Stack:** XGBoost 3.0, scikit-learn 1.7 (`LogisticRegression`, `train_test_split`, `accuracy_score`, `roc_auc_score`, `log_loss`, `brier_score_loss`), pandas, pickle, json.

---

## Files

- Modify: `src/mlb_edge_finder/model.py` — replace all `raise NotImplementedError` stubs; add `NON_FEATURE_COLS`, `_split()`, `train_baseline()`; update `train()` return type
- Modify: `tests/test_model.py` — add behaviour tests (existing 4 signature tests stay untouched)
- Modify: `notebooks/01_exploration.ipynb` — add Phase 4c section

---

## Task 1: Data preparation — `NON_FEATURE_COLS` and `_split()`

**Files:**
- Modify: `src/mlb_edge_finder/model.py`
- Modify: `tests/test_model.py`

- [ ] **Step 1: Add a shared test fixture to `tests/test_model.py`**

Append this below the existing imports (keep existing tests untouched):

```python
import numpy as np
import pandas as pd
import pytest


def _make_df(n=20):
    """Minimal training DataFrame with correct schema for testing."""
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "game_date": ["2024-04-01"] * n,
        "home_name": ["Team A"] * n,
        "away_name": ["Team B"] * n,
        "home_score": rng.integers(0, 10, n),
        "away_score": rng.integers(0, 10, n),
        "home_abbr": ["AAA"] * n,
        "away_abbr": ["BBB"] * n,
        "season": [2024] * n,
        "home_win": [1, 0] * (n // 2),
        "home_bat_avg": rng.uniform(0.220, 0.280, n),
        "home_obp": rng.uniform(0.300, 0.380, n),
        "home_slg": rng.uniform(0.380, 0.500, n),
        "home_ops": rng.uniform(0.680, 0.880, n),
        "home_runs_per_game": rng.uniform(3.5, 6.0, n),
        "home_era": rng.uniform(3.0, 5.5, n),
        "home_whip": rng.uniform(1.1, 1.5, n),
        "home_k_per_9": rng.uniform(7.0, 10.5, n),
        "home_bb_per_9": rng.uniform(2.5, 4.0, n),
        "home_fip_computed": rng.uniform(3.5, 5.0, n),
        "away_bat_avg": rng.uniform(0.220, 0.280, n),
        "away_obp": rng.uniform(0.300, 0.380, n),
        "away_slg": rng.uniform(0.380, 0.500, n),
        "away_ops": rng.uniform(0.680, 0.880, n),
        "away_runs_per_game": rng.uniform(3.5, 6.0, n),
        "away_era": rng.uniform(3.0, 5.5, n),
        "away_whip": rng.uniform(1.1, 1.5, n),
        "away_k_per_9": rng.uniform(7.0, 10.5, n),
        "away_bb_per_9": rng.uniform(2.5, 4.0, n),
        "away_fip_computed": rng.uniform(3.5, 5.0, n),
    })


def test_split_shapes():
    from mlb_edge_finder.model import _split
    df = _make_df(20)
    X_train, X_test, y_train, y_test = _split(df)
    assert len(X_train) == 16
    assert len(X_test) == 4
    assert len(y_train) == 16
    assert len(y_test) == 4


def test_split_no_metadata_columns():
    from mlb_edge_finder.model import _split, NON_FEATURE_COLS
    df = _make_df(20)
    X_train, X_test, y_train, y_test = _split(df)
    for col in NON_FEATURE_COLS:
        assert col not in X_train.columns
        assert col not in X_test.columns


def test_split_missing_target_raises():
    from mlb_edge_finder.model import _split
    df = _make_df(20).drop(columns=["home_win"])
    with pytest.raises(ValueError, match="home_win"):
        _split(df)


def test_split_empty_df_raises():
    from mlb_edge_finder.model import _split
    df = _make_df(20).iloc[0:0]
    with pytest.raises(FileNotFoundError):
        _split(df)
```

- [ ] **Step 2: Run the new tests to confirm they fail**

```bash
cd /Users/jaydengould/Documents/projects/mlb-edge-finder && pytest tests/test_model.py::test_split_shapes tests/test_model.py::test_split_no_metadata_columns tests/test_model.py::test_split_missing_target_raises tests/test_model.py::test_split_empty_df_raises -v
```

Expected: 4 failures — `ImportError` or `AttributeError` on `_split` / `NON_FEATURE_COLS`.

- [ ] **Step 3: Implement `NON_FEATURE_COLS` and `_split()` in `model.py`**

Replace the block after the `TARGET_COL` line and before `def train(...)` with:

```python
TARGET_COL = "home_win"

NON_FEATURE_COLS = [
    "game_date", "home_name", "away_name",
    "home_score", "away_score", "home_abbr", "away_abbr",
    "season", TARGET_COL,
]


def _split(
    features_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    from sklearn.model_selection import train_test_split

    if TARGET_COL not in features_df.columns:
        raise ValueError(f"Missing target column '{TARGET_COL}' in features_df")
    if features_df.empty:
        raise FileNotFoundError("features_df is empty — run build_training_set() first")
    X = features_df.drop(columns=[c for c in NON_FEATURE_COLS if c in features_df.columns])
    y = features_df[TARGET_COL]
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
```

- [ ] **Step 4: Run the tests and confirm they pass**

```bash
cd /Users/jaydengould/Documents/projects/mlb-edge-finder && pytest tests/test_model.py::test_split_shapes tests/test_model.py::test_split_no_metadata_columns tests/test_model.py::test_split_missing_target_raises tests/test_model.py::test_split_empty_df_raises -v
```

Expected: 4 PASSED.

- [ ] **Step 5: Run full test suite to confirm no regressions**

```bash
cd /Users/jaydengould/Documents/projects/mlb-edge-finder && pytest tests/ -v
```

Expected: all existing tests pass plus the 4 new ones.

- [ ] **Step 6: Commit**

```bash
git add src/mlb_edge_finder/model.py tests/test_model.py
git commit -m "feat: add NON_FEATURE_COLS and _split() to model.py"
```

---

## Task 2: `train()`

**Files:**
- Modify: `src/mlb_edge_finder/model.py`
- Modify: `tests/test_model.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_model.py`:

```python
def test_train_returns_classifier_and_test_split():
    from mlb_edge_finder.model import train
    from xgboost import XGBClassifier
    df = _make_df(20)
    clf, X_test, y_test = train(df)
    assert isinstance(clf, XGBClassifier)
    assert len(X_test) == 4
    assert len(y_test) == 4


def test_train_clf_can_predict_proba():
    from mlb_edge_finder.model import train
    df = _make_df(20)
    clf, X_test, y_test = train(df)
    proba = clf.predict_proba(X_test)
    assert proba.shape == (4, 2)
    assert (proba >= 0).all() and (proba <= 1).all()


def test_train_raises_on_missing_target():
    from mlb_edge_finder.model import train
    df = _make_df(20).drop(columns=["home_win"])
    with pytest.raises(ValueError):
        train(df)


def test_train_raises_on_empty_df():
    from mlb_edge_finder.model import train
    df = _make_df(20).iloc[0:0]
    with pytest.raises(FileNotFoundError):
        train(df)
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd /Users/jaydengould/Documents/projects/mlb-edge-finder && pytest tests/test_model.py::test_train_returns_classifier_and_test_split tests/test_model.py::test_train_clf_can_predict_proba tests/test_model.py::test_train_raises_on_missing_target tests/test_model.py::test_train_raises_on_empty_df -v
```

Expected: 4 failures — `NotImplementedError`.

- [ ] **Step 3: Implement `train()` in `model.py`**

Replace the existing `train()` stub (the whole function body) with:

```python
def train(
    features_df: pd.DataFrame,
) -> tuple[XGBClassifier, pd.DataFrame, pd.Series]:
    """Train an XGBoost classifier to predict home-team win probability.

    Splits features_df 80/20 (stratified, random_state=42), fits an
    XGBClassifier using config.XGB_N_ESTIMATORS and config.XGB_MAX_DEPTH,
    and returns the model plus the held-out test split.

    Args:
        features_df: Output of build_training_set() or load_training_set().
            Must contain TARGET_COL as the label column.

    Returns:
        Tuple of (fitted XGBClassifier, X_test DataFrame, y_test Series).

    Raises:
        ValueError: If TARGET_COL is missing from features_df.
        FileNotFoundError: If features_df is empty.
    """
    X_train, X_test, y_train, y_test = _split(features_df)
    clf = XGBClassifier(
        n_estimators=config.XGB_N_ESTIMATORS,
        max_depth=config.XGB_MAX_DEPTH,
        eval_metric="logloss",
        random_state=42,
    )
    clf.fit(X_train, y_train)
    logger.info(
        "Trained XGBClassifier: %d samples, %d features",
        len(X_train),
        X_train.shape[1],
    )
    return clf, X_test, y_test
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
cd /Users/jaydengould/Documents/projects/mlb-edge-finder && pytest tests/test_model.py::test_train_returns_classifier_and_test_split tests/test_model.py::test_train_clf_can_predict_proba tests/test_model.py::test_train_raises_on_missing_target tests/test_model.py::test_train_raises_on_empty_df -v
```

Expected: 4 PASSED.

- [ ] **Step 5: Run full test suite**

```bash
cd /Users/jaydengould/Documents/projects/mlb-edge-finder && pytest tests/ -v
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/mlb_edge_finder/model.py tests/test_model.py
git commit -m "feat: implement train() in model.py"
```

---

## Task 3: `train_baseline()`

**Files:**
- Modify: `src/mlb_edge_finder/model.py`
- Modify: `tests/test_model.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_model.py`:

```python
def test_train_baseline_returns_logistic_regression_and_test_split():
    from mlb_edge_finder.model import train_baseline
    from sklearn.linear_model import LogisticRegression
    df = _make_df(20)
    clf, X_test, y_test = train_baseline(df)
    assert isinstance(clf, LogisticRegression)
    assert len(X_test) == 4
    assert len(y_test) == 4


def test_train_baseline_same_test_split_as_train():
    from mlb_edge_finder.model import train, train_baseline
    df = _make_df(20)
    _, X_test_xgb, y_test_xgb = train(df)
    _, X_test_lr, y_test_lr = train_baseline(df)
    pd.testing.assert_frame_equal(X_test_xgb.reset_index(drop=True), X_test_lr.reset_index(drop=True))
    pd.testing.assert_series_equal(y_test_xgb.reset_index(drop=True), y_test_lr.reset_index(drop=True))


def test_train_baseline_can_predict_proba():
    from mlb_edge_finder.model import train_baseline
    df = _make_df(20)
    clf, X_test, y_test = train_baseline(df)
    proba = clf.predict_proba(X_test)
    assert proba.shape == (4, 2)
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd /Users/jaydengould/Documents/projects/mlb-edge-finder && pytest tests/test_model.py::test_train_baseline_returns_logistic_regression_and_test_split tests/test_model.py::test_train_baseline_same_test_split_as_train tests/test_model.py::test_train_baseline_can_predict_proba -v
```

Expected: 3 failures — `AttributeError` (no `train_baseline`).

- [ ] **Step 3: Implement `train_baseline()` in `model.py`**

Add this function after `train()` (before `evaluate()`):

```python
def train_baseline(
    features_df: pd.DataFrame,
) -> tuple[Any, pd.DataFrame, pd.Series]:
    """Train a logistic regression baseline for comparison with XGBoost.

    Uses the same 80/20 stratified split as train() for a fair comparison.
    Not persisted to disk — use evaluate() to compare metrics.

    Args:
        features_df: Same format as accepted by train().

    Returns:
        Tuple of (fitted LogisticRegression, X_test DataFrame, y_test Series).

    Raises:
        ValueError: If TARGET_COL is missing from features_df.
        FileNotFoundError: If features_df is empty.
    """
    from sklearn.linear_model import LogisticRegression

    X_train, X_test, y_train, y_test = _split(features_df)
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)
    logger.info("Trained LogisticRegression baseline: %d samples", len(X_train))
    return clf, X_test, y_test
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
cd /Users/jaydengould/Documents/projects/mlb-edge-finder && pytest tests/test_model.py::test_train_baseline_returns_logistic_regression_and_test_split tests/test_model.py::test_train_baseline_same_test_split_as_train tests/test_model.py::test_train_baseline_can_predict_proba -v
```

Expected: 3 PASSED.

- [ ] **Step 5: Run full test suite**

```bash
cd /Users/jaydengould/Documents/projects/mlb-edge-finder && pytest tests/ -v
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/mlb_edge_finder/model.py tests/test_model.py
git commit -m "feat: implement train_baseline() in model.py"
```

---

## Task 4: `evaluate()`

**Files:**
- Modify: `src/mlb_edge_finder/model.py`
- Modify: `tests/test_model.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_model.py`:

```python
EXPECTED_METRIC_KEYS = {
    "accuracy", "roc_auc", "log_loss", "brier_score",
    "n_test_samples", "xgb_n_estimators", "xgb_max_depth",
}


def test_evaluate_xgb_returns_all_keys():
    from mlb_edge_finder.model import train, evaluate
    df = _make_df(20)
    clf, X_test, y_test = train(df)
    metrics = evaluate(clf, X_test, y_test)
    assert set(metrics.keys()) == EXPECTED_METRIC_KEYS


def test_evaluate_xgb_metric_ranges():
    from mlb_edge_finder.model import train, evaluate
    df = _make_df(20)
    clf, X_test, y_test = train(df)
    metrics = evaluate(clf, X_test, y_test)
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert metrics["log_loss"] >= 0.0
    assert 0.0 <= metrics["brier_score"] <= 1.0
    assert metrics["n_test_samples"] == 4


def test_evaluate_xgb_hyperparams_populated():
    from mlb_edge_finder.model import train, evaluate
    from mlb_edge_finder import config
    df = _make_df(20)
    clf, X_test, y_test = train(df)
    metrics = evaluate(clf, X_test, y_test)
    assert metrics["xgb_n_estimators"] == config.XGB_N_ESTIMATORS
    assert metrics["xgb_max_depth"] == config.XGB_MAX_DEPTH


def test_evaluate_baseline_hyperparam_keys_are_none():
    from mlb_edge_finder.model import train_baseline, evaluate
    df = _make_df(20)
    clf, X_test, y_test = train_baseline(df)
    metrics = evaluate(clf, X_test, y_test)
    assert set(metrics.keys()) == EXPECTED_METRIC_KEYS
    assert metrics["xgb_n_estimators"] is None
    assert metrics["xgb_max_depth"] is None
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd /Users/jaydengould/Documents/projects/mlb-edge-finder && pytest tests/test_model.py::test_evaluate_xgb_returns_all_keys tests/test_model.py::test_evaluate_xgb_metric_ranges tests/test_model.py::test_evaluate_xgb_hyperparams_populated tests/test_model.py::test_evaluate_baseline_hyperparam_keys_are_none -v
```

Expected: 4 failures — `NotImplementedError`.

- [ ] **Step 3: Implement `evaluate()` in `model.py`**

Replace the existing `evaluate()` stub body with:

```python
def evaluate(clf: Any, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, Any]:
    """Compute evaluation metrics for a trained classifier.

    Works with any sklearn-compatible classifier (XGBClassifier or
    LogisticRegression). XGBoost-specific keys are None for other classifiers.

    Args:
        clf: Fitted classifier with predict() and predict_proba() methods.
        X_test: Feature matrix (rows = games, columns = feature columns).
        y_test: True binary labels (1 = home win, 0 = away win).

    Returns:
        Dict with keys: accuracy, roc_auc, log_loss, brier_score,
        n_test_samples, xgb_n_estimators, xgb_max_depth.
    """
    from sklearn.metrics import (
        accuracy_score,
        brier_score_loss,
        log_loss,
        roc_auc_score,
    )

    proba = clf.predict_proba(X_test)[:, 1]
    preds = clf.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, preds),
        "roc_auc": roc_auc_score(y_test, proba),
        "log_loss": log_loss(y_test, proba),
        "brier_score": brier_score_loss(y_test, proba),
        "n_test_samples": len(y_test),
        "xgb_n_estimators": getattr(clf, "n_estimators", None),
        "xgb_max_depth": getattr(clf, "max_depth", None),
    }
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
cd /Users/jaydengould/Documents/projects/mlb-edge-finder && pytest tests/test_model.py::test_evaluate_xgb_returns_all_keys tests/test_model.py::test_evaluate_xgb_metric_ranges tests/test_model.py::test_evaluate_xgb_hyperparams_populated tests/test_model.py::test_evaluate_baseline_hyperparam_keys_are_none -v
```

Expected: 4 PASSED.

- [ ] **Step 5: Run full test suite**

```bash
cd /Users/jaydengould/Documents/projects/mlb-edge-finder && pytest tests/ -v
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/mlb_edge_finder/model.py tests/test_model.py
git commit -m "feat: implement evaluate() in model.py"
```

---

## Task 5: `save_model()` and `load_model()`

**Files:**
- Modify: `src/mlb_edge_finder/model.py`
- Modify: `tests/test_model.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_model.py`:

```python
def test_save_and_load_model_roundtrip(tmp_path, monkeypatch):
    from mlb_edge_finder import config
    from mlb_edge_finder.model import evaluate, load_model, save_model, train
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    df = _make_df(20)
    clf, X_test, y_test = train(df)
    metrics = evaluate(clf, X_test, y_test)
    game_date = date(2024, 4, 1)
    save_model(clf, metrics, game_date)
    loaded = load_model(game_date)
    import numpy as np
    np.testing.assert_array_equal(
        clf.predict(X_test), loaded.predict(X_test)
    )


def test_save_model_writes_both_files(tmp_path, monkeypatch):
    from mlb_edge_finder import config
    from mlb_edge_finder.model import evaluate, save_model, train
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    df = _make_df(20)
    clf, X_test, y_test = train(df)
    metrics = evaluate(clf, X_test, y_test)
    game_date = date(2024, 4, 1)
    save_model(clf, metrics, game_date)
    assert (tmp_path / "xgb_2024-04-01.pkl").exists()
    assert (tmp_path / "metrics_2024-04-01.json").exists()


def test_load_model_raises_when_missing(tmp_path, monkeypatch):
    from mlb_edge_finder import config
    from mlb_edge_finder.model import load_model
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    with pytest.raises(FileNotFoundError, match="2024-04-01"):
        load_model(date(2024, 4, 1))
```

Add this import at the top of the test file (after existing imports):

```python
from datetime import date
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd /Users/jaydengould/Documents/projects/mlb-edge-finder && pytest tests/test_model.py::test_save_and_load_model_roundtrip tests/test_model.py::test_save_model_writes_both_files tests/test_model.py::test_load_model_raises_when_missing -v
```

Expected: 3 failures — `NotImplementedError`.

- [ ] **Step 3: Implement `save_model()` in `model.py`**

Replace the `save_model()` stub body with:

```python
def save_model(clf: XGBClassifier, metrics: dict[str, Any], game_date: date) -> None:
    """Persist a trained model and its metrics to disk.

    Writes two files to MODELS_DIR:
      - xgb_YYYY-MM-DD.pkl  — pickled XGBClassifier object
      - metrics_YYYY-MM-DD.json — JSON with eval metrics and hyperparameters

    Args:
        clf: Fitted XGBClassifier to persist.
        metrics: Output of evaluate().
        game_date: Used to name the output files.
    """
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    pkl_path = config.MODELS_DIR / f"xgb_{game_date}.pkl"
    json_path = config.MODELS_DIR / f"metrics_{game_date}.json"
    with open(pkl_path, "wb") as f:
        pickle.dump(clf, f)
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Saved model → %s", pkl_path)
    logger.info("Saved metrics → %s", json_path)
```

- [ ] **Step 4: Implement `load_model()` in `model.py`**

Replace the `load_model()` stub body with:

```python
def load_model(game_date: date) -> XGBClassifier:
    """Load a previously saved XGBClassifier from MODELS_DIR.

    Args:
        game_date: The date whose .pkl file to load.

    Returns:
        Fitted XGBClassifier ready for inference.

    Raises:
        FileNotFoundError: If no model file exists for the given date.
    """
    pkl_path = config.MODELS_DIR / f"xgb_{game_date}.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(
            f"No model found for {game_date} at {pkl_path} — run save_model() first"
        )
    with open(pkl_path, "rb") as f:
        return pickle.load(f)
```

- [ ] **Step 5: Run tests to confirm they pass**

```bash
cd /Users/jaydengould/Documents/projects/mlb-edge-finder && pytest tests/test_model.py::test_save_and_load_model_roundtrip tests/test_model.py::test_save_model_writes_both_files tests/test_model.py::test_load_model_raises_when_missing -v
```

Expected: 3 PASSED.

- [ ] **Step 6: Run full test suite**

```bash
cd /Users/jaydengould/Documents/projects/mlb-edge-finder && pytest tests/ -v
```

Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add src/mlb_edge_finder/model.py tests/test_model.py
git commit -m "feat: implement save_model() and load_model() in model.py"
```

---

## Task 6: Notebook — Phase 4c section

**Files:**
- Modify: `notebooks/01_exploration.ipynb`

- [ ] **Step 1: Add a Markdown header cell for Phase 4c**

Add a new Markdown cell at the end of the notebook:

```markdown
## Phase 4c — Model Training and Evaluation
```

- [ ] **Step 2: Add imports and data-loading cell**

Add a new Code cell:

```python
from datetime import date
from mlb_edge_finder import model
from mlb_edge_finder.training_data import load_training_set

training_df = load_training_set([2023, 2024, 2025])
print(f"Training set: {training_df.shape[0]} rows, {training_df.shape[1]} columns")
```

- [ ] **Step 3: Add XGBoost training cell**

Add a new Code cell:

```python
clf, X_test, y_test = model.train(training_df)
print(f"Test set size: {len(X_test)} games")
print(f"Features used: {list(X_test.columns)}")
```

- [ ] **Step 4: Add baseline training cell**

Add a new Code cell:

```python
baseline_clf, _, _ = model.train_baseline(training_df)
print("Baseline (logistic regression) trained.")
```

- [ ] **Step 5: Add evaluation and comparison cell**

Add a new Code cell:

```python
import pandas as pd

xgb_metrics = model.evaluate(clf, X_test, y_test)
lr_metrics = model.evaluate(baseline_clf, X_test, y_test)

comparison = pd.DataFrame(
    {"XGBoost": xgb_metrics, "LogisticRegression": lr_metrics},
    index=xgb_metrics.keys(),
)
print(comparison.to_string())
```

- [ ] **Step 6: Add save and reload cell**

Add a new Code cell:

```python
today = date.today()
model.save_model(clf, xgb_metrics, today)

loaded_clf = model.load_model(today)
print("Model reloaded successfully.")
print(f"Sample predictions: {loaded_clf.predict(X_test[:3])}")
```

- [ ] **Step 7: Run all Phase 4c cells top-to-bottom and confirm no errors**

Kernel → Restart & Run All. Verify:
- Training set loads with ~6787 rows
- `clf` and `baseline_clf` are trained without error
- Metrics comparison table prints with all 7 keys
- `xgb_*.pkl` and `metrics_*.json` appear in `models/`
- Loaded model produces same predictions as original

- [ ] **Step 8: Commit**

```bash
git add notebooks/01_exploration.ipynb models/
git commit -m "feat: add Phase 4c model training section to notebook"
```
