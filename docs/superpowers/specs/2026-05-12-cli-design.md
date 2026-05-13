# CLI Entry Point Design Spec

**Date:** 2026-05-12  
**Status:** Approved

## Goal

Add `src/mlb_edge_finder/__main__.py` so the package can be run as:

```bash
python -m mlb_edge_finder [--date YYYY-MM-DD] [--force]
```

Also add `force: bool = False` to `pipeline.run()` so caches can be bypassed end-to-end.

## Invocation

| Command | Behaviour |
|---|---|
| `python -m mlb_edge_finder` | Run for today |
| `python -m mlb_edge_finder --date 2026-05-12` | Run for a specific date |
| `python -m mlb_edge_finder --force` | Re-fetch all data, bypass caches |
| `python -m mlb_edge_finder --date 2026-05-12 --force` | Both |

## `__main__.py`

**Arguments (argparse, stdlib only):**

- `--date` — optional, ISO format (`YYYY-MM-DD`), validated with `date.fromisoformat()`. Defaults to `date.today()`. Invalid format → print error, `sys.exit(1)`.
- `--force` — boolean flag (`store_true`). Defaults to `False`.

**Flow:**
1. Parse args
2. Validate `--date` (catch `ValueError` from `date.fromisoformat`)
3. Call `config.setup_logging()`
4. Call `pipeline.run(game_date, force=force)`
5. On success: print summary + formatted table (see Output section)
6. On any exception: log error message, `sys.exit(1)`

**Exit codes:**
- `0` — success (including "no edges found" — not an error)
- `1` — bad `--date` argument or pipeline exception

## Output Format

When edges are found:

```
Found 2 edge(s) for 2026-05-12:

  home_team         away_team        bet_side  american_odds  model_prob    ev  kelly_fraction
  New York Yankees  Boston Red Sox   home            +110       0.712   0.083           0.038
```

When no edges are found:

```
No edges found for 2026-05-12.
```

Uses `DataFrame.to_string(index=False)`. No new dependencies.

## `pipeline.run()` update

Add `force: bool = False` parameter. Pass it through to all four cache-aware calls:

```python
odds_ingestion.fetch_odds(game_date, force=force)
stats_ingestion.fetch_stats(game_date, force=force)
pitcher_ingestion.fetch_pitcher_stats(game_date, force=force)
features.build_features(game_date, force=force)
```

All four functions already accept `force`. No other changes to `pipeline.py`.

## No Changes Needed

- `fetch_odds`, `fetch_stats`, `fetch_pitcher_stats`, `build_features` — all already have `force` parameters
- `pyproject.toml` — no console script entry point needed (`python -m` is sufficient)
- No new dependencies

## Tests

New file: `tests/test_main.py`

| Test | What it verifies |
|---|---|
| `test_main_runs_for_today` | No `--date` → `pipeline.run` called with `date.today()`, `force=False` |
| `test_main_date_flag` | `--date 2026-05-12` → `pipeline.run` called with `date(2026, 5, 12)`, `force=False` |
| `test_main_force_flag` | `--force` → `pipeline.run` called with `force=True` |
| `test_main_invalid_date` | `--date not-a-date` → exits with code 1 |
| `test_main_pipeline_error_exits_nonzero` | `pipeline.run` raises `FileNotFoundError` → exits with code 1 |
| `test_main_no_edges_exits_zero` | `pipeline.run` returns empty DataFrame → exits with code 0 |
| `test_main_edges_printed_to_stdout` | Edges DataFrame → output contains team name |

Updated file: `tests/test_pipeline.py`

| Test | What it verifies |
|---|---|
| `test_pipeline_run_accepts_force_param` | `pipeline.run` signature has `force` parameter |
| `test_pipeline_run_passes_force_to_stages` | `force=True` is forwarded to `fetch_odds`, `fetch_stats`, `fetch_pitcher_stats`, `build_features` |
