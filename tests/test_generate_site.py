import json
import csv
from datetime import date
from pathlib import Path

import pytest


def _write_edges_csv(path: Path, rows: list[dict]) -> None:
    cols = ["game_id", "home_team", "away_team", "bet_side",
            "american_odds", "model_prob", "ev", "kelly_fraction", "high_confidence"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def _write_header_only_csv(path: Path) -> None:
    path.write_text(
        "game_id,home_team,away_team,bet_side,american_odds,model_prob,ev,kelly_fraction,high_confidence\n"
    )


def _write_temporal_eval_json(path: Path, **overrides) -> dict:
    data = {
        "holdout_season": 2025,
        "train_seasons": [2019, 2021, 2022, 2023, 2024],
        "n_train": 12000,
        "n_test": 2400,
        "accuracy": 0.572,
        "roc_auc": 0.601,
        "log_loss": 0.681,
        "brier_score": 0.243,
        "n_bets": 1800,
        "win_rate": 0.603,
        "roi_pct": 15.1,
        "sharpe_ratio": 0.42,
        "total_pnl": 1800.0,
        "max_drawdown": 420.0,
        "avg_ev": 0.28,
        "break_even_alpha": 0.1,
        "market_efficiency_sweep": [
            {"alpha": 0.0, "roi_pct": 15.1, "n_bets": 1800},
            {"alpha": 0.5, "roi_pct": 4.0, "n_bets": 700},
            {"alpha": 1.0, "roi_pct": -4.8, "n_bets": 40},
        ],
    }
    data.update(overrides)
    path.write_text(json.dumps(data))
    return data


# --- _load_edges_data ---

def test_load_edges_data_today_rows(tmp_path):
    from mlb_win_probability.generate_site import _load_edges_data
    today = date.today().isoformat()
    _write_edges_csv(tmp_path / f"edges_{today}.csv", [
        {"game_id": "abc", "home_team": "Giants", "away_team": "Dodgers",
         "bet_side": "home", "american_odds": -110, "model_prob": 0.7,
         "ev": 0.55, "kelly_fraction": 0.25, "high_confidence": False},
    ])
    today_rows, history = _load_edges_data(tmp_path)
    assert len(today_rows) == 1
    assert today_rows[0]["home_team"] == "Giants"


def test_load_edges_data_history_count(tmp_path):
    from mlb_win_probability.generate_site import _load_edges_data
    _write_edges_csv(tmp_path / "edges_2026-05-20.csv", [
        {"game_id": "a", "home_team": "X", "away_team": "Y", "bet_side": "home",
         "american_odds": -110, "model_prob": 0.6, "ev": 0.5, "kelly_fraction": 0.2, "high_confidence": False},
        {"game_id": "b", "home_team": "X", "away_team": "Z", "bet_side": "away",
         "american_odds": 120, "model_prob": 0.6, "ev": 0.5, "kelly_fraction": 0.2, "high_confidence": False},
    ])
    _write_header_only_csv(tmp_path / "edges_2026-05-21.csv")
    _, history = _load_edges_data(tmp_path)
    counts = {h["date"]: h["count"] for h in history}
    assert counts["2026-05-20"] == 2
    assert counts["2026-05-21"] == 0


def test_load_edges_data_empty_outputs_dir(tmp_path):
    from mlb_win_probability.generate_site import _load_edges_data
    today_rows, history = _load_edges_data(tmp_path)
    assert today_rows == []
    assert history == []


def test_load_edges_data_caps_at_30_days(tmp_path):
    from mlb_win_probability.generate_site import _load_edges_data
    for i in range(35):
        _write_header_only_csv(tmp_path / f"edges_2026-04-{i+1:02d}.csv")
    _, history = _load_edges_data(tmp_path)
    assert len(history) <= 30


# --- _load_temporal_eval ---

def test_load_temporal_eval_returns_dict(tmp_path):
    from mlb_win_probability.generate_site import _load_temporal_eval
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    _write_temporal_eval_json(models_dir / "temporal_eval_2025.json")
    result = _load_temporal_eval(models_dir)
    assert result is not None
    assert result["roc_auc"] == 0.601


def test_load_temporal_eval_returns_none_when_missing(tmp_path):
    from mlb_win_probability.generate_site import _load_temporal_eval
    assert _load_temporal_eval(tmp_path) is None


def test_load_temporal_eval_picks_most_recent(tmp_path):
    from mlb_win_probability.generate_site import _load_temporal_eval
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    _write_temporal_eval_json(models_dir / "temporal_eval_2024.json", roc_auc=0.55)
    _write_temporal_eval_json(models_dir / "temporal_eval_2025.json", roc_auc=0.601)
    result = _load_temporal_eval(models_dir)
    assert result["roc_auc"] == 0.601


# --- generate() integration ---

def test_generate_creates_index_html(tmp_path):
    from mlb_win_probability.generate_site import generate
    today = date.today().isoformat()
    outputs_dir = tmp_path / "outputs"
    outputs_dir.mkdir()
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    _write_edges_csv(outputs_dir / f"edges_{today}.csv", [
        {"game_id": "abc", "home_team": "Giants", "away_team": "Dodgers",
         "bet_side": "home", "american_odds": -110, "model_prob": 0.7,
         "ev": 0.55, "kelly_fraction": 0.25, "high_confidence": False},
    ])
    out = tmp_path / "docs" / "index.html"
    generate(outputs_dir=outputs_dir, models_dir=models_dir, out_path=out)
    assert out.exists()
    html = out.read_text()
    assert "Giants" in html
    assert "MLB Win Probability" in html


def test_generate_empty_state_when_no_edges_today(tmp_path):
    from mlb_win_probability.generate_site import generate
    outputs_dir = tmp_path / "outputs"
    outputs_dir.mkdir()
    today = date.today().isoformat()
    _write_header_only_csv(outputs_dir / f"edges_{today}.csv")
    out = tmp_path / "docs" / "index.html"
    generate(outputs_dir=outputs_dir, models_dir=tmp_path, out_path=out)
    html = out.read_text()
    assert "No games flagged today" in html


def test_generate_includes_stats_when_temporal_eval_present(tmp_path):
    from mlb_win_probability.generate_site import generate
    outputs_dir = tmp_path / "outputs"
    outputs_dir.mkdir()
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    _write_temporal_eval_json(models_dir / "temporal_eval_2025.json")
    out = tmp_path / "docs" / "index.html"
    generate(outputs_dir=outputs_dir, models_dir=models_dir, out_path=out)
    html = out.read_text()
    assert "60.3%" in html    # win_rate
    assert "15.1%" in html    # roi_pct
    assert "0.601" in html    # roc_auc
    assert "2025 holdout" in html


def test_generate_still_works_when_temporal_eval_missing(tmp_path):
    from mlb_win_probability.generate_site import generate
    outputs_dir = tmp_path / "outputs"
    outputs_dir.mkdir()
    out = tmp_path / "docs" / "index.html"
    generate(outputs_dir=outputs_dir, models_dir=tmp_path, out_path=out)
    assert out.exists()
    html = out.read_text()
    assert "<!DOCTYPE html>" in html


def test_generate_high_confidence_shows_star_badge(tmp_path):
    from mlb_win_probability.generate_site import generate
    today = date.today().isoformat()
    outputs_dir = tmp_path / "outputs"
    outputs_dir.mkdir()
    _write_edges_csv(outputs_dir / f"edges_{today}.csv", [
        {"game_id": "x", "home_team": "A", "away_team": "B",
         "bet_side": "away", "american_odds": 150, "model_prob": 0.75,
         "ev": 0.6, "kelly_fraction": 0.3, "high_confidence": True},
    ])
    out = tmp_path / "docs" / "index.html"
    generate(outputs_dir=outputs_dir, models_dir=tmp_path, out_path=out)
    html = out.read_text()
    assert "★" in html
    assert "⚠" not in html


def test_generate_renders_efficiency_chart(tmp_path):
    from mlb_win_probability.generate_site import generate
    outputs_dir = tmp_path / "outputs"
    outputs_dir.mkdir()
    (outputs_dir / "edges_2025-01-01.csv").write_text(
        "game_id,home_team,away_team,bet_side,american_odds,model_prob,ev,kelly_fraction,high_confidence\n"
    )
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    _write_temporal_eval_json(models_dir / "temporal_eval_2025.json")
    out = tmp_path / "index.html"
    generate(outputs_dir=outputs_dir, models_dir=models_dir, out_path=out)
    html = out.read_text()
    assert "efficiency-chart" in html
    assert "market_efficiency_sweep" in html


def test_generate_no_pnl_chart(tmp_path):
    from mlb_win_probability.generate_site import generate
    outputs_dir = tmp_path / "outputs"
    outputs_dir.mkdir()
    (outputs_dir / "edges_2025-01-01.csv").write_text(
        "game_id,home_team,away_team,bet_side,american_odds,model_prob,ev,kelly_fraction,high_confidence\n"
    )
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    _write_temporal_eval_json(models_dir / "temporal_eval_2025.json")
    out = tmp_path / "index.html"
    generate(outputs_dir=outputs_dir, models_dir=models_dir, out_path=out)
    html = out.read_text()
    assert "pnl-chart" not in html
    assert "pnl_series" not in html


def test_generate_stats_card_shows_roc_and_break_even(tmp_path):
    from mlb_win_probability.generate_site import generate
    outputs_dir = tmp_path / "outputs"
    outputs_dir.mkdir()
    (outputs_dir / "edges_2025-01-01.csv").write_text(
        "game_id,home_team,away_team,bet_side,american_odds,model_prob,ev,kelly_fraction,high_confidence\n"
    )
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    _write_temporal_eval_json(models_dir / "temporal_eval_2025.json")
    out = tmp_path / "index.html"
    generate(outputs_dir=outputs_dir, models_dir=models_dir, out_path=out)
    html = out.read_text()
    assert "0.601" in html
    assert "ROC-AUC" in html
    assert "synthetic counterparty" in html.lower()
