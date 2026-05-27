import csv
import json
from datetime import date
from pathlib import Path

import pytest


def _write_edges_csv(path: Path, rows: list[dict]) -> None:
    cols = ["game_id", "home_team", "away_team", "bet_side",
            "american_odds", "model_prob", "ev", "kelly_fraction", "prob_flag"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def _write_header_only_csv(path: Path) -> None:
    path.write_text(
        "game_id,home_team,away_team,bet_side,american_odds,model_prob,ev,kelly_fraction,prob_flag\n"
    )


# --- _load_edges_data ---

def test_load_edges_data_today_rows(tmp_path):
    from mlb_edge_finder.generate_site import _load_edges_data
    today = date.today().isoformat()
    _write_edges_csv(tmp_path / f"edges_{today}.csv", [
        {"game_id": "abc", "home_team": "Giants", "away_team": "Dodgers",
         "bet_side": "home", "american_odds": -110, "model_prob": 0.7,
         "ev": 0.55, "kelly_fraction": 0.25, "prob_flag": False},
    ])
    today_rows, history = _load_edges_data(tmp_path)
    assert len(today_rows) == 1
    assert today_rows[0]["home_team"] == "Giants"


def test_load_edges_data_history_count(tmp_path):
    from mlb_edge_finder.generate_site import _load_edges_data
    _write_edges_csv(tmp_path / "edges_2026-05-20.csv", [
        {"game_id": "a", "home_team": "X", "away_team": "Y", "bet_side": "home",
         "american_odds": -110, "model_prob": 0.6, "ev": 0.5, "kelly_fraction": 0.2, "prob_flag": False},
        {"game_id": "b", "home_team": "X", "away_team": "Z", "bet_side": "away",
         "american_odds": 120, "model_prob": 0.6, "ev": 0.5, "kelly_fraction": 0.2, "prob_flag": False},
    ])
    _write_header_only_csv(tmp_path / "edges_2026-05-21.csv")
    _, history = _load_edges_data(tmp_path)
    counts = {h["date"]: h["count"] for h in history}
    assert counts["2026-05-20"] == 2
    assert counts["2026-05-21"] == 0


def test_load_edges_data_empty_outputs_dir(tmp_path):
    from mlb_edge_finder.generate_site import _load_edges_data
    today_rows, history = _load_edges_data(tmp_path)
    assert today_rows == []
    assert history == []


def test_load_edges_data_caps_at_30_days(tmp_path):
    from mlb_edge_finder.generate_site import _load_edges_data
    for i in range(35):
        _write_header_only_csv(tmp_path / f"edges_2026-04-{i+1:02d}.csv")
    _, history = _load_edges_data(tmp_path)
    assert len(history) <= 30


# --- _load_metrics ---

def test_load_metrics_returns_dict(tmp_path):
    from mlb_edge_finder.generate_site import _load_metrics
    p = tmp_path / "metrics_2026-05-26.json"
    p.write_text(json.dumps({"roc_auc": 0.601, "n_test_samples": 3168}))
    result = _load_metrics(p)
    assert result["roc_auc"] == 0.601


def test_load_metrics_missing_path_returns_none():
    from mlb_edge_finder.generate_site import _load_metrics
    assert _load_metrics(None) is None


def test_load_metrics_nonexistent_file_returns_none(tmp_path):
    from mlb_edge_finder.generate_site import _load_metrics
    assert _load_metrics(tmp_path / "missing.json") is None


# --- _load_pnl ---

def test_load_pnl_returns_dict(tmp_path):
    from mlb_edge_finder.generate_site import _load_pnl
    p = tmp_path / "backtest_pnl.json"
    p.write_text(json.dumps({"cumulative_pnl": [0.0, 1.0], "summary": {"win_rate": 0.6}}))
    result = _load_pnl(p)
    assert result["summary"]["win_rate"] == 0.6


def test_load_pnl_missing_returns_none():
    from mlb_edge_finder.generate_site import _load_pnl
    assert _load_pnl(None) is None


# --- generate() integration ---

def test_generate_creates_index_html(tmp_path):
    from mlb_edge_finder.generate_site import generate
    today = date.today().isoformat()
    outputs_dir = tmp_path / "outputs"
    outputs_dir.mkdir()
    _write_edges_csv(outputs_dir / f"edges_{today}.csv", [
        {"game_id": "abc", "home_team": "Giants", "away_team": "Dodgers",
         "bet_side": "home", "american_odds": -110, "model_prob": 0.7,
         "ev": 0.55, "kelly_fraction": 0.25, "prob_flag": False},
    ])
    out = tmp_path / "docs" / "index.html"
    generate(outputs_dir=outputs_dir, metrics_path=None, pnl_path=None, out_path=out)
    assert out.exists()
    html = out.read_text()
    assert "Giants" in html
    assert "Dodgers" in html
    assert "MLB Edge Finder" in html


def test_generate_empty_state_when_no_edges_today(tmp_path):
    from mlb_edge_finder.generate_site import generate
    outputs_dir = tmp_path / "outputs"
    outputs_dir.mkdir()
    today = date.today().isoformat()
    _write_header_only_csv(outputs_dir / f"edges_{today}.csv")
    out = tmp_path / "docs" / "index.html"
    generate(outputs_dir=outputs_dir, metrics_path=None, pnl_path=None, out_path=out)
    html = out.read_text()
    assert "No edges found today" in html


def test_generate_includes_stats_when_metrics_present(tmp_path):
    from mlb_edge_finder.generate_site import generate
    outputs_dir = tmp_path / "outputs"
    outputs_dir.mkdir()
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(json.dumps({"roc_auc": 0.601, "n_test_samples": 3168}))
    pnl_path = tmp_path / "pnl.json"
    pnl_path.write_text(json.dumps({
        "cumulative_pnl": [0.0, 100.0, 200.0],
        "summary": {"n_bets": 10, "win_rate": 0.6, "roi_pct": 15.1, "sharpe_ratio": 0.74},
    }))
    out = tmp_path / "docs" / "index.html"
    generate(outputs_dir=outputs_dir, metrics_path=metrics_path, pnl_path=pnl_path, out_path=out)
    html = out.read_text()
    assert "60.0%" in html   # win rate
    assert "15.1%" in html   # roi
    assert "0.601" in html   # roc_auc


def test_generate_still_works_when_pnl_missing(tmp_path):
    from mlb_edge_finder.generate_site import generate
    outputs_dir = tmp_path / "outputs"
    outputs_dir.mkdir()
    out = tmp_path / "docs" / "index.html"
    generate(outputs_dir=outputs_dir, metrics_path=None, pnl_path=None, out_path=out)
    assert out.exists()
    html = out.read_text()
    assert "<!DOCTYPE html>" in html


def test_generate_prob_flag_shows_warning_badge(tmp_path):
    from mlb_edge_finder.generate_site import generate
    today = date.today().isoformat()
    outputs_dir = tmp_path / "outputs"
    outputs_dir.mkdir()
    _write_edges_csv(outputs_dir / f"edges_{today}.csv", [
        {"game_id": "x", "home_team": "A", "away_team": "B",
         "bet_side": "away", "american_odds": 150, "model_prob": 0.85,
         "ev": 0.6, "kelly_fraction": 0.3, "prob_flag": True},
    ])
    out = tmp_path / "docs" / "index.html"
    generate(outputs_dir=outputs_dir, metrics_path=None, pnl_path=None, out_path=out)
    html = out.read_text()
    assert "⚠" in html
