# GitHub Actions Daily Workflow — Design Spec

**Date:** 2026-05-19  
**Status:** Approved

## Goal

Automate daily execution of `python -m mlb_edge_finder` on GitHub's servers at 9:30 AM ET each morning. Results are committed back to the repo as `outputs/edges_YYYY-MM-DD.csv`, creating a visible, browsable output history without requiring the MacBook to be on.

## Trigger

Cron schedule: `30 13 * * *` UTC (9:30 AM ET during EDT / 2:30 PM UTC during EST in winter).

Note: GitHub Actions cron can be delayed 5–30 minutes during high-traffic periods — acceptable for this use case since no automated betting occurs.

## Workflow Steps

1. **Checkout repo** — `actions/checkout@v4` with write permissions so the workflow can commit back
2. **Set up Python 3.12** — `actions/setup-python@v5`
3. **Install dependencies** — `pip install -e .`
4. **Run pipeline** — `python -m mlb_edge_finder`, using `ODDS_API_KEY` injected from GitHub Actions secrets. Exits 0 on success (including no edges), exits 1 on failure.
5. **Move output** — copy `data/processed/edges_YYYY-MM-DD.csv` → `outputs/edges_YYYY-MM-DD.csv`
6. **Commit and push** — only if the edges file is new (git diff check). Commit message: `chore: edges YYYY-MM-DD`. Uses `github-actions[bot]` as the committer.

## Error Handling

- If `python -m mlb_edge_finder` exits 1 (API down, no starters, pipeline failure), the workflow step fails and GitHub marks the run as failed. No notification is sent — failure is visible in the Actions tab. No partial commits are made.
- The `continue-on-error` flag is NOT set — a failed run is surfaced, not silently swallowed.

## Repo Changes

| Path | Change |
|---|---|
| `.github/workflows/daily.yml` | New — workflow definition |
| `outputs/` | New unignored directory for committed edge files |
| `outputs/.gitkeep` | New — keeps directory tracked in git when no edges files exist yet |

`data/processed/` remains gitignored. Only the final edges CSV is promoted to `outputs/`.

## Secrets

One secret required: `ODDS_API_KEY` — added to the repo under Settings → Secrets and variables → Actions. The workflow reads it as `${{ secrets.ODDS_API_KEY }}` and passes it as an environment variable.

## What Does Not Change

- Local CLI (`python -m mlb_edge_finder`) works exactly as before — the workflow and CLI are independent
- The model is loaded from the committed `.pkl` file in `models/` — no retraining occurs in the workflow
- All other pipeline behavior (cache-first data fetching, EV filtering, Kelly sizing, `prob_flag`) is unchanged

## Out of Scope

- Email/Slack notifications on success or failure
- Daily model retraining
- Appending current-season results to training data
- Multiple workflow runs per day
