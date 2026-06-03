"""Generate the static GitHub Pages dashboard at docs/index.html."""
import json
import logging
from datetime import date
from html import escape
from pathlib import Path

import pandas as pd

from mlb_edge_finder.stats_ingestion import ODDS_NAME_TO_ABBR

logger = logging.getLogger(__name__)


def _team_abbr(full_name: str) -> str:
    """Return the 3-letter abbreviation for a full Odds API team name."""
    return ODDS_NAME_TO_ABBR.get(full_name, full_name)


_ROOT = Path(__file__).resolve().parents[2]
OUTPUTS_DIR: Path = _ROOT / "outputs"
DOCS_DIR: Path = _ROOT / "docs"


def _load_edges_data(outputs_dir: Path) -> tuple[list[dict], list[dict]]:
    """Load today's edges and per-day edge counts from outputs/ CSVs.

    Returns:
        (today_rows, history) where history is a list of {date, count} dicts
        covering the last 30 available days sorted oldest-first.
    """
    if not outputs_dir.exists():
        return [], []
    today = date.today().isoformat()
    csv_files = sorted(outputs_dir.glob("edges_*.csv"))[-30:]

    history: list[dict] = []
    today_rows: list[dict] = []

    for csv_path in csv_files:
        file_date = csv_path.stem[len("edges_"):]
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            logger.warning("Skipping malformed CSV: %s", csv_path)
            continue
        history.append({"date": file_date, "count": len(df)})
        if file_date == today:
            today_rows = df.to_dict(orient="records")

    return today_rows, history


def _load_temporal_eval(models_dir: Path) -> dict | None:
    """Load the most recent temporal_eval_*.json from models_dir, or None."""
    files = sorted(Path(models_dir).glob("temporal_eval_*.json"))
    if not files:
        return None
    return json.loads(files[-1].read_text())


def _render_stats_html(te_data: dict | None) -> str:
    """Render the holdout-evaluation stats card, or '' if no data."""
    if not te_data:
        return ""
    rows = []
    if "roc_auc" in te_data:
        rows.append(
            f'<div class="stat-row"><span class="stat-label">ROC-AUC</span>'
            f'<span class="stat-value neutral">{te_data["roc_auc"]:.3f}</span></div>'
        )
    if te_data.get("break_even_alpha") is not None:
        rows.append(
            f'<div class="stat-row"><span class="stat-label">Break-even efficiency</span>'
            f'<span class="stat-value">&alpha; &approx; {te_data["break_even_alpha"]:.2f}</span></div>'
        )
    elif "break_even_alpha" in te_data:
        rows.append(
            '<div class="stat-row"><span class="stat-label">Break-even efficiency</span>'
            '<span class="stat-value neutral">none in range</span></div>'
        )
    if "accuracy" in te_data:
        rows.append(
            f'<div class="stat-row"><span class="stat-label">Accuracy</span>'
            f'<span class="stat-value neutral">{te_data["accuracy"] * 100:.1f}%</span></div>'
        )
    if "n_test" in te_data:
        rows.append(
            f'<div class="stat-row"><span class="stat-label">Holdout games</span>'
            f'<span class="stat-value neutral">{te_data["n_test"]:,}</span></div>'
        )
    if "win_rate" in te_data:
        rows.append(
            f'<div class="stat-row"><span class="stat-label">Win Rate (naive market)</span>'
            f'<span class="stat-value">{te_data["win_rate"] * 100:.1f}%</span></div>'
        )
    if "roi_pct" in te_data:
        roi = te_data["roi_pct"]
        roi_prefix = "+" if roi >= 0 else ""
        roi_class = "stat-value green" if roi >= 0 else "stat-value"
        rows.append(
            f'<div class="stat-row"><span class="stat-label">ROI (naive market)</span>'
            f'<span class="{roi_class}">{roi_prefix}{roi:.1f}%</span></div>'
        )
    if not rows:
        return ""
    train_seasons = te_data.get("train_seasons", [])
    holdout = te_data.get("holdout_season", "")
    subtitle = ""
    if train_seasons and holdout:
        subtitle = (
            f'<div class="card-subtitle">Trained {train_seasons[0]}&ndash;'
            f'{train_seasons[-1]} &middot; {holdout} holdout &middot; synthetic-market stress test</div>'
        )
    return (
        '<div class="card"><div class="card-title">Holdout Evaluation</div>'
        + subtitle
        + "".join(rows)
        + "</div>"
    )


def _render_efficiency_html(te_data: dict | None) -> str:
    """Render the market-efficiency sensitivity chart card, or '' if no data."""
    if te_data is None or not te_data.get("market_efficiency_sweep"):
        return ""
    return (
        '<div class="card">'
        '<div class="card-title">Edge vs Market Efficiency</div>'
        '<div class="chart-wrap-sm"><canvas id="efficiency-chart"></canvas></div>'
        '<div class="card-caption">Synthetic-market stress test &mdash; betting ROI as the '
        "market becomes as informed as the model (0 = ignores matchup, 1 = as sharp as the model).</div>"
        "</div>"
    )


def _render_edges_html(today_rows: list[dict]) -> str:
    """Render the today's edges table as static HTML."""
    if not today_rows:
        return (
            '<div class="empty-state">No edges found today &mdash; model found no '
            "+EV opportunities meeting the current thresholds.</div>"
        )
    rows_html = ""
    for r in today_rows:
        home = escape(str(r.get("home_team", "")))
        away = escape(str(r.get("away_team", "")))
        side = escape(str(r.get("bet_side", "")))
        sc = "side-home" if r.get("bet_side") == "home" else "side-away"
        raw_home = str(r.get("home_team", ""))
        raw_away = str(r.get("away_team", ""))
        bet_team = escape(_team_abbr(raw_home if r.get("bet_side") == "home" else raw_away))
        high_conf = r.get("high_confidence")
        is_high_conf = high_conf is True or str(high_conf).strip() == "True"
        star = "★ " if is_high_conf else ""
        odds_int = int(r.get("american_odds", 0) or 0)
        odds_str = f"+{odds_int}" if odds_int > 0 else str(odds_int)
        model_prob = float(r.get("model_prob", 0))
        ev = float(r.get("ev", 0))
        ev_str = f"+{ev * 100:.1f}%" if ev >= 0 else f"{ev * 100:.1f}%"
        kelly = float(r.get("kelly_fraction", 0))
        rows_html += (
            f"<tr>"
            f"<td>{home} vs {away}</td>"
            f'<td><span class="side-badge {sc}">{star}{bet_team}</span></td>'
            f"<td>{odds_str}</td>"
            f"<td>{model_prob * 100:.1f}%</td>"
            f'<td class="ev-val">{ev_str}</td>'
            f"<td>{kelly * 100:.1f}%</td>"
            f"</tr>"
        )
    return (
        "<table><thead><tr>"
        "<th>Matchup</th><th>Bet On</th><th>Odds</th>"
        "<th>Model Prob</th><th>EV</th><th>Kelly</th>"
        f"</tr></thead><tbody>{rows_html}</tbody></table>"
    )


def _render_html(
    today_rows: list[dict],
    history: list[dict],
    te_data: dict | None,
    updated: str,
) -> str:
    """Return the complete HTML page as a string."""
    edges_table_html = _render_edges_html(today_rows)
    history_json = json.dumps(history)
    te_json = json.dumps(te_data) if te_data else "null"
    stats_html = _render_stats_html(te_data)
    efficiency_chart_html = _render_efficiency_html(te_data)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>MLB Edge Finder — Daily Dashboard</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
  <style>
    *{{margin:0;padding:0;box-sizing:border-box}}
    body{{background:#1a1713;color:#d6cfc4;font-family:-apple-system,'Segoe UI',sans-serif;min-height:100vh}}
    a{{color:#FD5A1E}}
    .page{{max-width:1100px;margin:0 auto;padding:24px 20px}}
    .header{{border-bottom:1px solid #3d3930;padding-bottom:16px;margin-bottom:24px;display:flex;align-items:flex-start;justify-content:space-between;flex-wrap:wrap;gap:8px}}
    .header-title{{font-size:22px;font-weight:800;letter-spacing:-0.02em;color:#fff}}
    .header-sub{{font-size:13px;color:#8a8070;margin-top:4px}}
    .badge{{font-size:11px;font-weight:700;background:#FD5A1E22;color:#FD5A1E;border:1px solid #FD5A1E44;padding:3px 10px;border-radius:12px;white-space:nowrap;margin-top:4px}}
    .main-layout{{display:flex;gap:24px;align-items:flex-start}}
    .col-main{{flex:1;min-width:0}}
    .col-sidebar{{width:240px;flex-shrink:0;display:flex;flex-direction:column;gap:16px}}
    .section{{margin-bottom:28px}}
    .section-title{{font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;color:#8a8070;margin-bottom:12px}}
    .table-wrap{{overflow-x:auto;border-radius:8px;border:1px solid #3d3930}}
    table{{width:100%;border-collapse:collapse;font-size:13px}}
    thead tr{{background:#27251F}}
    th{{padding:10px 12px;text-align:left;font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:0.07em;color:#8a8070;white-space:nowrap}}
    td{{padding:10px 12px;border-bottom:1px solid #2e2b25;color:#d6cfc4;white-space:nowrap}}
    tbody tr:last-child td{{border-bottom:none}}
    tbody tr:hover td{{background:#27251F55}}
    .ev-val{{color:#FD5A1E;font-weight:700}}
    .side-badge{{display:inline-block;font-size:11px;font-weight:600;padding:2px 8px;border-radius:4px;white-space:nowrap}}
    .side-home{{background:#FD5A1E22;color:#FD5A1E;border:1px solid #FD5A1E44}}
    .side-away{{background:#ffffff11;color:#8a8070;border:1px solid #3d3930}}
    .empty-state{{text-align:center;padding:32px;color:#8a8070;font-size:14px;line-height:1.6}}
    .card{{background:#27251F;border:1px solid #3d3930;border-radius:8px;padding:16px}}
    .card-title{{font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;color:#8a8070;margin-bottom:4px}}
    .card-subtitle{{font-size:10px;color:#8a8070;margin-bottom:10px;opacity:0.75}}
    .card-caption{{font-size:10px;color:#8a8070;margin-top:8px;line-height:1.4;opacity:0.8}}
    .stat-row{{display:flex;justify-content:space-between;align-items:baseline;margin-bottom:8px}}
    .stat-row:last-child{{margin-bottom:0}}
    .stat-label{{font-size:12px;color:#8a8070}}
    .stat-value{{font-size:14px;font-weight:700;color:#FD5A1E}}
    .stat-value.green{{color:#4ade80}}
    .stat-value.neutral{{color:#d6cfc4}}
    .chart-wrap{{position:relative;height:120px}}
    .chart-wrap-sm{{position:relative;height:100px}}
    .updated{{font-size:11px;color:#8a8070;text-align:right}}
    @media(max-width:700px){{.main-layout{{flex-direction:column}}.col-sidebar{{width:100%}}}}
  </style>
</head>
<body>
<div class="page">
  <div class="header">
    <div>
      <div class="header-title">&#9918; MLB Edge Finder</div>
      <div class="header-sub">XGBoost model identifying positive expected-value MLB moneyline bets</div>
    </div>
    <span class="badge">Updated {updated}</span>
  </div>
  <div class="main-layout">
    <div class="col-main">
      <div class="section">
        <div class="section-title">Today&#39;s Edges</div>
        <div class="table-wrap" id="edges-table">{edges_table_html}</div>
      </div>
      <div class="section">
        <div class="section-title">Edge History &mdash; Last 30 Days</div>
        <div class="chart-wrap"><canvas id="history-chart"></canvas></div>
      </div>
    </div>
    <div class="col-sidebar">
      <div class="updated">Updated {updated}</div>
      {stats_html}
      {efficiency_chart_html}
    </div>
  </div>
</div>
<script>
const HISTORY={history_json};
const TE={te_json};
</script>
<script>
(function(){{
  if(!HISTORY||HISTORY.length===0)return;
  var ctx=document.getElementById('history-chart').getContext('2d');
  new Chart(ctx,{{type:'bar',data:{{labels:HISTORY.map(function(d){{return d.date.slice(5)}}),datasets:[{{data:HISTORY.map(function(d){{return d.count}}),backgroundColor:'#FD5A1E99',borderColor:'#FD5A1E',borderWidth:1,borderRadius:2}}]}},options:{{responsive:true,maintainAspectRatio:false,plugins:{{legend:{{display:false}},tooltip:{{callbacks:{{title:function(t){{return HISTORY[t[0].dataIndex].date}},label:function(t){{return t.raw+' edge'+(t.raw!==1?'s':'')}}}}}}}},scales:{{x:{{grid:{{color:'#3d393044'}},ticks:{{color:'#8a8070',font:{{size:10}}}}}},y:{{grid:{{color:'#3d393044'}},ticks:{{color:'#8a8070',font:{{size:10}},stepSize:1}},beginAtZero:true}}}}}}}});
}})();
(function(){{
  var el=document.getElementById('efficiency-chart');
  if(!el||!TE||!TE.market_efficiency_sweep||TE.market_efficiency_sweep.length===0)return;
  var S=TE.market_efficiency_sweep;
  var ctx=el.getContext('2d');
  new Chart(ctx,{{type:'line',data:{{labels:S.map(function(d){{return d.alpha}}),datasets:[{{data:S.map(function(d){{return d.roi_pct}}),borderColor:'#FD5A1E',borderWidth:2,pointRadius:0,tension:0.1,fill:false}}]}},options:{{responsive:true,maintainAspectRatio:false,plugins:{{legend:{{display:false}},tooltip:{{callbacks:{{title:function(t){{return'α = '+S[t[0].dataIndex].alpha}},label:function(t){{return'ROI '+t.raw.toFixed(1)+'%'}}}}}}}},scales:{{x:{{grid:{{color:'#3d393044'}},ticks:{{color:'#8a8070',font:{{size:9}}}}}},y:{{grid:{{color:'#3d393044'}},ticks:{{color:'#8a8070',font:{{size:10}},callback:function(v){{return v+'%'}}}}}}}}}}}});
}})();
</script>
</body>
</html>"""


def generate(
    outputs_dir: Path,
    models_dir: Path,
    out_path: Path,
) -> None:
    """Generate docs/index.html from outputs CSVs and temporal eval artifact.

    Never raises — degrades gracefully if the temporal eval file is missing.

    Args:
        outputs_dir: Directory containing edges_YYYY-MM-DD.csv files.
        models_dir: Directory containing temporal_eval_*.json files.
        out_path: Destination for the generated index.html.
    """
    today_rows, history = _load_edges_data(outputs_dir)
    te_data = _load_temporal_eval(models_dir)

    updated = date.today().strftime("%B %d, %Y").replace(" 0", " ")
    html = _render_html(today_rows, history, te_data, updated)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    logger.info(
        "Dashboard written to %s (%d edges today, %d history days)",
        out_path, len(today_rows), len(history),
    )


if __name__ == "__main__":
    from mlb_edge_finder import config as _config

    generate(
        outputs_dir=_ROOT / "outputs",
        models_dir=_config.MODELS_DIR,
        out_path=DOCS_DIR / "index.html",
    )
