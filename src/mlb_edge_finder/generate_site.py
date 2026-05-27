"""Generate the static GitHub Pages dashboard at docs/index.html."""
import json
import logging
from datetime import date
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[2]
OUTPUTS_DIR: Path = _ROOT / "outputs"
DOCS_DIR: Path = _ROOT / "docs"
PNL_PATH: Path = _ROOT / "data" / "backtest_pnl.json"


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
        df = pd.read_csv(csv_path)
        history.append({"date": file_date, "count": len(df)})
        if file_date == today:
            today_rows = df.to_dict(orient="records")

    return today_rows, history


def _load_metrics(metrics_path: Path | None) -> dict | None:
    """Return parsed metrics JSON or None if path is missing."""
    if metrics_path is None:
        return None
    p = Path(metrics_path)
    if not p.exists():
        return None
    return json.loads(p.read_text())


def _load_pnl(pnl_path: Path | None) -> dict | None:
    """Return parsed backtest P&L JSON or None if path is missing."""
    if pnl_path is None:
        return None
    p = Path(pnl_path)
    if not p.exists():
        return None
    return json.loads(p.read_text())


def _render_stats_html(metrics: dict | None, pnl_data: dict | None) -> str:
    """Render the backtest performance stats card HTML, or '' if no data."""
    summary = (pnl_data or {}).get("summary", {})
    rows = []
    if "win_rate" in summary:
        rows.append(
            f'<div class="stat-row"><span class="stat-label">Win Rate</span>'
            f'<span class="stat-value">{summary["win_rate"] * 100:.1f}%</span></div>'
        )
    if "roi_pct" in summary:
        rows.append(
            f'<div class="stat-row"><span class="stat-label">Backtest ROI</span>'
            f'<span class="stat-value green">+{summary["roi_pct"]:.1f}%</span></div>'
        )
    if "sharpe_ratio" in summary:
        rows.append(
            f'<div class="stat-row"><span class="stat-label">Sharpe</span>'
            f'<span class="stat-value">{summary["sharpe_ratio"]:.3f}</span></div>'
        )
    if metrics and "roc_auc" in metrics:
        rows.append(
            f'<div class="stat-row"><span class="stat-label">ROC-AUC</span>'
            f'<span class="stat-value neutral">{metrics["roc_auc"]:.3f}</span></div>'
        )
    if metrics and "n_test_samples" in metrics:
        rows.append(
            f'<div class="stat-row"><span class="stat-label">Test games</span>'
            f'<span class="stat-value neutral">{metrics["n_test_samples"]:,}</span></div>'
        )
    if not rows:
        return ""
    return (
        '<div class="card"><div class="card-title">Backtest Performance</div>'
        + "".join(rows)
        + "</div>"
    )


def _render_pnl_html(pnl_data: dict | None) -> str:
    """Render the P&L chart card HTML, or '' if no data."""
    if pnl_data is None:
        return ""
    return (
        '<div class="card">'
        '<div class="card-title">Backtest P&amp;L Curve</div>'
        '<div class="chart-wrap-sm"><canvas id="pnl-chart"></canvas></div>'
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
        sc = "side-home" if r.get("bet_side") == "home" else "side-away"
        prob_flag = r.get("prob_flag")
        flagged = prob_flag is True or str(prob_flag).strip() == "True"
        flag = '<span title="Model probability >80% — review manually">⚠</span>' if flagged else ""
        odds_val = r.get("american_odds", 0)
        odds_str = f"+{odds_val}" if int(odds_val) > 0 else str(odds_val)
        model_prob = float(r.get("model_prob", 0))
        ev = float(r.get("ev", 0))
        kelly = float(r.get("kelly_fraction", 0))
        bet_side = r.get("bet_side", "")
        rows_html += (
            f"<tr>"
            f"<td>{r.get('home_team', '')} vs {r.get('away_team', '')}</td>"
            f'<td><span class="side-badge {sc}">{bet_side}</span></td>'
            f"<td>{odds_str}</td>"
            f"<td>{model_prob * 100:.1f}%</td>"
            f'<td class="ev-val">+{ev * 100:.1f}%</td>'
            f"<td>{kelly * 100:.1f}%</td>"
            f"<td>{flag}</td>"
            f"</tr>"
        )
    return (
        "<table><thead><tr>"
        "<th>Matchup</th><th>Side</th><th>Odds</th>"
        "<th>Model Prob</th><th>EV</th><th>Kelly</th><th>Flag</th>"
        f"</tr></thead><tbody>{rows_html}</tbody></table>"
    )


def _render_html(
    today_rows: list[dict],
    history: list[dict],
    metrics: dict | None,
    pnl_data: dict | None,
    updated: str,
) -> str:
    """Return the complete HTML page as a string."""
    edges_table_html = _render_edges_html(today_rows)
    history_json = json.dumps(history)
    pnl_json = json.dumps(pnl_data) if pnl_data else "null"
    metrics_json = json.dumps(metrics) if metrics else "null"
    stats_html = _render_stats_html(metrics, pnl_data)
    pnl_chart_html = _render_pnl_html(pnl_data)

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
    .side-badge{{display:inline-block;font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:0.06em;padding:2px 7px;border-radius:4px}}
    .side-home{{background:#FD5A1E22;color:#FD5A1E;border:1px solid #FD5A1E44}}
    .side-away{{background:#ffffff11;color:#8a8070;border:1px solid #3d3930}}
    .empty-state{{text-align:center;padding:32px;color:#8a8070;font-size:14px;line-height:1.6}}
    .card{{background:#27251F;border:1px solid #3d3930;border-radius:8px;padding:16px}}
    .card-title{{font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;color:#8a8070;margin-bottom:12px}}
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
      {pnl_chart_html}
    </div>
  </div>
</div>
<script>
const HISTORY={history_json};
const PNL={pnl_json};
const METRICS={metrics_json};
</script>
<script>
(function(){{
  if(!HISTORY||HISTORY.length===0)return;
  var ctx=document.getElementById('history-chart').getContext('2d');
  new Chart(ctx,{{type:'bar',data:{{labels:HISTORY.map(function(d){{return d.date.slice(5)}}),datasets:[{{data:HISTORY.map(function(d){{return d.count}}),backgroundColor:'#FD5A1E99',borderColor:'#FD5A1E',borderWidth:1,borderRadius:2}}]}},options:{{responsive:true,maintainAspectRatio:false,plugins:{{legend:{{display:false}},tooltip:{{callbacks:{{title:function(t){{return HISTORY[t[0].dataIndex].date}},label:function(t){{return t.raw+' edge'+(t.raw!==1?'s':'')}}}}}}}},scales:{{x:{{grid:{{color:'#3d393044'}},ticks:{{color:'#8a8070',font:{{size:10}}}}}},y:{{grid:{{color:'#3d393044'}},ticks:{{color:'#8a8070',font:{{size:10}},stepSize:1}},beginAtZero:true}}}}}}}}}});
}})();
(function(){{
  var el=document.getElementById('pnl-chart');
  if(!el||!PNL)return;
  var ctx=el.getContext('2d');
  new Chart(ctx,{{type:'line',data:{{labels:PNL.cumulative_pnl.map(function(_,i){{return i}}),datasets:[{{data:PNL.cumulative_pnl,borderColor:'#FD5A1E',borderWidth:2,pointRadius:0,tension:0.1,fill:false}}]}},options:{{responsive:true,maintainAspectRatio:false,plugins:{{legend:{{display:false}},tooltip:{{callbacks:{{label:function(t){{return'$'+t.raw.toFixed(0)}}}}}}}},scales:{{x:{{display:false}},y:{{grid:{{color:'#3d393044'}},ticks:{{color:'#8a8070',font:{{size:10}},callback:function(v){{return'$'+v}}}}}}}}}}}}}});
}})();
</script>
</body>
</html>"""


def generate(
    outputs_dir: Path,
    metrics_path: Path | None,
    pnl_path: Path | None,
    out_path: Path,
) -> None:
    """Generate docs/index.html from outputs CSVs and model artifacts.

    Never raises — degrades gracefully if metrics or PnL files are missing.

    Args:
        outputs_dir: Directory containing edges_YYYY-MM-DD.csv files.
        metrics_path: Path to a metrics_YYYY-MM-DD.json file, or None.
        pnl_path: Path to data/backtest_pnl.json, or None.
        out_path: Destination for the generated index.html.
    """
    today_rows, history = _load_edges_data(outputs_dir)
    metrics = _load_metrics(metrics_path)
    pnl_data = _load_pnl(pnl_path)

    updated = date.today().strftime("%B %-d, %Y")
    html = _render_html(today_rows, history, metrics, pnl_data, updated)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    logger.info(
        "Dashboard written to %s (%d edges today, %d history days)",
        out_path, len(today_rows), len(history),
    )


if __name__ == "__main__":
    import glob as _glob
    from mlb_edge_finder import config as _config

    _metrics_files = sorted(_glob.glob(str(_config.MODELS_DIR / "metrics_*.json")))
    _metrics_path = Path(_metrics_files[-1]) if _metrics_files else None

    generate(
        outputs_dir=_ROOT / "outputs",
        metrics_path=_metrics_path,
        pnl_path=PNL_PATH,
        out_path=DOCS_DIR / "index.html",
    )
