"""Train, evaluate, and persist the XGBoost win-probability model."""
import json
import logging
import pickle
from datetime import date
from typing import Any

import pandas as pd
from xgboost import XGBClassifier

from mlb_edge_finder import config

logger = logging.getLogger(__name__)

# Column that holds the binary win/loss target in the features DataFrame.
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


def train(features_df: pd.DataFrame) -> XGBClassifier:
    """Train an XGBoost classifier to predict home-team win probability.

    Splits features_df into train/test sets (80/20), fits an XGBClassifier
    using config.XGB_N_ESTIMATORS and config.XGB_MAX_DEPTH, and returns
    the trained model. Does not persist — call save_model() separately.

    Args:
        features_df: Output of features.build_features() or load_features().
            Must contain TARGET_COL as the label column.

    Returns:
        Fitted XGBClassifier instance.

    Raises:
        FileNotFoundError: If features_df is empty.
        ValueError: If TARGET_COL is missing from features_df.
    """
    raise NotImplementedError


def evaluate(clf: XGBClassifier, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, Any]:
    """Compute evaluation metrics for a trained classifier.

    Args:
        clf: Fitted XGBClassifier from train().
        X_test: Feature matrix (rows = games, columns = feature columns).
        y_test: True binary labels (1 = home win, 0 = away win).

    Returns:
        Dict with keys: accuracy, roc_auc, log_loss, n_test_samples,
        xgb_n_estimators, xgb_max_depth.
    """
    raise NotImplementedError


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
    raise NotImplementedError


def load_model(game_date: date) -> XGBClassifier:
    """Load a previously saved XGBClassifier from MODELS_DIR.

    Args:
        game_date: The date whose .pkl file to load.

    Returns:
        Fitted XGBClassifier ready for inference.

    Raises:
        FileNotFoundError: If no model file exists for the given date.
    """
    raise NotImplementedError
