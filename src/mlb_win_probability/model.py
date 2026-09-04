"""Train, evaluate, and persist the XGBoost win-probability model."""
import json
import logging
import pickle
from datetime import date
from typing import Any

import pandas as pd
from xgboost import XGBClassifier

from mlb_win_probability import config

logger = logging.getLogger(__name__)

# Column that holds the binary win/loss target in the features DataFrame.
TARGET_COL = "home_win"

NON_FEATURE_COLS = [
    "game_date", "home_name", "away_name",
    "home_score", "away_score", "home_abbr", "away_abbr",
    "season", TARGET_COL,
    "home_starter_name", "away_starter_name",
    "home_pitcher_id", "away_pitcher_id",
]


def _split(
    features_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """80/20 stratified split — used by train_baseline() only."""
    from sklearn.model_selection import train_test_split

    if TARGET_COL not in features_df.columns:
        raise ValueError(f"Missing target column '{TARGET_COL}' in features_df")
    if features_df.empty:
        raise FileNotFoundError("features_df is empty — run build_training_set() first")
    X = features_df.drop(columns=[c for c in NON_FEATURE_COLS if c in features_df.columns])
    y = features_df[TARGET_COL]
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def _three_way_split(
    features_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """60/20/20 stratified split: fit / validation / test.

    The validation set is held out from XGBoost training so it can be used to
    fit a post-hoc probability calibrator without data leakage.
    """
    from sklearn.model_selection import train_test_split

    if TARGET_COL not in features_df.columns:
        raise ValueError(f"Missing target column '{TARGET_COL}' in features_df")
    if features_df.empty:
        raise FileNotFoundError("features_df is empty — run build_training_set() first")
    X = features_df.drop(columns=[c for c in NON_FEATURE_COLS if c in features_df.columns])
    y = features_df[TARGET_COL]
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.25, random_state=42, stratify=y_trainval
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def train(
    features_df: pd.DataFrame,
) -> tuple[XGBClassifier, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Train an XGBoost classifier to predict home-team win probability.

    Splits features_df 60/20/20 (stratified, random_state=42): 60% used to fit
    XGBoost, 20% returned as a calibration validation set (X_val/y_val), and
    20% held out for final evaluation (X_test/y_test).

    Pass X_val/y_val to calibrate() to produce a probability-calibrated model.

    Args:
        features_df: Output of build_training_set() or load_training_set().
            Must contain TARGET_COL as the label column.

    Returns:
        Tuple of (fitted XGBClassifier, X_val, X_test, y_val, y_test).

    Raises:
        ValueError: If TARGET_COL is missing from features_df.
        FileNotFoundError: If features_df is empty.
    """
    X_train, X_val, X_test, y_train, y_val, y_test = _three_way_split(features_df)
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
    return clf, X_val, X_test, y_val, y_test


def calibrate(clf: Any, X_val: pd.DataFrame, y_val: pd.Series) -> Any:
    """Wrap a fitted classifier with isotonic probability calibration.

    Wraps clf in a FrozenEstimator (sklearn 1.6+) so the underlying model is
    not retrained — only the isotonic calibration layer is fit on X_val/y_val
    via CalibratedClassifierCV. X_val must be held out from the data used to
    train clf.

    Args:
        clf: A fitted classifier with predict_proba() (e.g. from train()).
        X_val: Feature matrix for the held-out validation set.
        y_val: True binary labels for the validation set.

    Returns:
        CalibratedClassifierCV fitted on X_val/y_val, ready for inference.
    """
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.frozen import FrozenEstimator

    cal_clf = CalibratedClassifierCV(estimator=FrozenEstimator(clf), method="isotonic")
    cal_clf.fit(X_val, y_val)
    logger.info("Calibrated classifier with isotonic regression on %d samples", len(y_val))
    return cal_clf


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
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    X_train, X_test, y_train, y_test = _split(features_df)
    clf = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", LogisticRegression(max_iter=1000, random_state=42)),
    ])
    clf.fit(X_train, y_train)
    logger.info("Trained LogisticRegression baseline: %d samples", len(X_train))
    return clf, X_test, y_test


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
