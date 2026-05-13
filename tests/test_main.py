"""Tests for __main__.py CLI entry point."""
from datetime import date
from unittest.mock import patch

import pandas as pd
import pytest


def _make_edges():
    return pd.DataFrame([{
        "game_id": "g1",
        "home_team": "New York Yankees",
        "away_team": "Boston Red Sox",
        "bet_side": "home",
        "american_odds": 110,
        "model_prob": 0.75,
        "ev": 0.083,
        "kelly_fraction": 0.038,
    }])


def test_main_runs_for_today():
    """No --date flag → pipeline.run called with date.today() and force=False."""
    from mlb_edge_finder.__main__ import main
    with patch("sys.argv", ["mlb_edge_finder"]), \
         patch("mlb_edge_finder.__main__.config.setup_logging"), \
         patch("mlb_edge_finder.__main__.pipeline.run", return_value=pd.DataFrame()) as mock_run:
        main()
    mock_run.assert_called_once_with(date.today(), force=False)


def test_main_date_flag():
    """--date 2026-05-12 → pipeline.run called with date(2026, 5, 12)."""
    from mlb_edge_finder.__main__ import main
    with patch("sys.argv", ["mlb_edge_finder", "--date", "2026-05-12"]), \
         patch("mlb_edge_finder.__main__.config.setup_logging"), \
         patch("mlb_edge_finder.__main__.pipeline.run", return_value=pd.DataFrame()) as mock_run:
        main()
    mock_run.assert_called_once_with(date(2026, 5, 12), force=False)


def test_main_force_flag():
    """--force → pipeline.run called with force=True."""
    from mlb_edge_finder.__main__ import main
    with patch("sys.argv", ["mlb_edge_finder", "--force"]), \
         patch("mlb_edge_finder.__main__.config.setup_logging"), \
         patch("mlb_edge_finder.__main__.pipeline.run", return_value=pd.DataFrame()) as mock_run:
        main()
    mock_run.assert_called_once_with(date.today(), force=True)


def test_main_invalid_date():
    """Invalid --date value → exits with code 1."""
    from mlb_edge_finder.__main__ import main
    with patch("sys.argv", ["mlb_edge_finder", "--date", "not-a-date"]):
        with pytest.raises(SystemExit) as exc_info:
            main()
    assert exc_info.value.code == 1


def test_main_pipeline_error_exits_nonzero():
    """Pipeline exception → exits with code 1."""
    from mlb_edge_finder.__main__ import main
    with patch("sys.argv", ["mlb_edge_finder"]), \
         patch("mlb_edge_finder.__main__.config.setup_logging"), \
         patch("mlb_edge_finder.__main__.pipeline.run",
               side_effect=FileNotFoundError("no model")):
        with pytest.raises(SystemExit) as exc_info:
            main()
    assert exc_info.value.code == 1


def test_main_no_edges_exits_zero(capsys):
    """Empty edges DataFrame → exits 0 and prints 'No edges found'."""
    from mlb_edge_finder.__main__ import main
    with patch("sys.argv", ["mlb_edge_finder"]), \
         patch("mlb_edge_finder.__main__.config.setup_logging"), \
         patch("mlb_edge_finder.__main__.pipeline.run", return_value=pd.DataFrame()):
        main()  # must not raise
    out = capsys.readouterr().out
    assert "No edges found" in out


def test_main_edges_printed_to_stdout(capsys):
    """Edges DataFrame → stdout contains team name and summary line."""
    from mlb_edge_finder.__main__ import main
    with patch("sys.argv", ["mlb_edge_finder"]), \
         patch("mlb_edge_finder.__main__.config.setup_logging"), \
         patch("mlb_edge_finder.__main__.pipeline.run", return_value=_make_edges()):
        main()
    out = capsys.readouterr().out
    assert "Found 1 edge" in out
    assert "New York Yankees" in out
