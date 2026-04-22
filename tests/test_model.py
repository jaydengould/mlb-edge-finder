"""Smoke tests: model exposes expected public API."""
import inspect


def test_train_signature():
    from mlb_edge_finder import model
    assert callable(model.train)
    sig = inspect.signature(model.train)
    assert "features_df" in sig.parameters


def test_evaluate_signature():
    from mlb_edge_finder import model
    assert callable(model.evaluate)
    sig = inspect.signature(model.evaluate)
    assert "clf" in sig.parameters
    assert "X_test" in sig.parameters
    assert "y_test" in sig.parameters


def test_save_model_signature():
    from mlb_edge_finder import model
    assert callable(model.save_model)
    sig = inspect.signature(model.save_model)
    assert "clf" in sig.parameters
    assert "metrics" in sig.parameters
    assert "game_date" in sig.parameters


def test_load_model_signature():
    from mlb_edge_finder import model
    assert callable(model.load_model)
    sig = inspect.signature(model.load_model)
    assert "game_date" in sig.parameters
