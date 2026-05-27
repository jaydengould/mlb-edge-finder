# Historical API Fallback Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** When `fetch_historical()` exhausts all retries and the MLB Stats API is unavailable, fall back to the stale cache instead of raising — so the daily pipeline never fails due to a transient API outage.

**Architecture:** Add a cache-fallback branch after the retry loop in `fetch_historical()`. If all retries fail and a cache file exists, log a warning and return the cached DataFrame. If no cache exists, raise as before. `time.sleep` is patched in tests to keep them fast.

**Tech Stack:** Python stdlib (`time`), pandas, pytest, unittest.mock

---

### Task 1: Add failing tests for the new fallback behavior

**Files:**
- Modify: `tests/test_historical_ingestion.py`

- [ ] **Step 1: Add the two new tests**

Open `tests/test_historical_ingestion.py` and append these two tests at the bottom (before the `historical_ingestion_module` helper):

```python
def test_fetch_historical_falls_back_to_cache_when_api_fails(tmp_path):
    """All retries fail but cache exists — should return cached data with a warning."""
    from mlb_edge_finder import historical_ingestion

    cached = pd.DataFrame([{
        "game_date": "2026-04-01", "home_name": "Yankees", "away_name": "Red Sox",
        "home_score": 5, "away_score": 3, "home_win": 1,
        "home_starter_name": None, "away_starter_name": None,
    }])
    cache_file = tmp_path / "historical_2026.csv"
    cached.to_csv(cache_file, index=False)

    with patch("mlb_edge_finder.historical_ingestion.statsapi.schedule", side_effect=Exception("503")), \
         patch("mlb_edge_finder.historical_ingestion.config.DATA_RAW_DIR", tmp_path), \
         patch("mlb_edge_finder.historical_ingestion.time.sleep"):
        df = historical_ingestion.fetch_historical(2026, force=True)

    assert len(df) == 1
    assert df.iloc[0]["home_name"] == "Yankees"


def test_fetch_historical_raises_when_api_fails_and_no_cache(tmp_path):
    """All retries fail and no cache exists — should raise RuntimeError."""
    from mlb_edge_finder import historical_ingestion

    with patch("mlb_edge_finder.historical_ingestion.statsapi.schedule", side_effect=Exception("503")), \
         patch("mlb_edge_finder.historical_ingestion.config.DATA_RAW_DIR", tmp_path), \
         patch("mlb_edge_finder.historical_ingestion.time.sleep"):
        with pytest.raises(RuntimeError, match="statsapi.schedule failed"):
            historical_ingestion.fetch_historical(2026, force=True)
```

- [ ] **Step 2: Also patch `time.sleep` in the existing failure test** so it stays fast

Find `test_fetch_historical_raises_on_api_failure` and add the sleep patch:

```python
def test_fetch_historical_raises_on_api_failure(tmp_path):
    from mlb_edge_finder import historical_ingestion
    with patch("mlb_edge_finder.historical_ingestion.statsapi.schedule", side_effect=Exception("timeout")), \
         patch("mlb_edge_finder.historical_ingestion.config.DATA_RAW_DIR", tmp_path), \
         patch("mlb_edge_finder.historical_ingestion.time.sleep"):
        with pytest.raises(RuntimeError, match="statsapi.schedule failed"):
            historical_ingestion.fetch_historical(2024, force=True)
```

- [ ] **Step 3: Run the new tests to confirm they fail**

```bash
pytest tests/test_historical_ingestion.py::test_fetch_historical_falls_back_to_cache_when_api_fails tests/test_historical_ingestion.py::test_fetch_historical_raises_when_api_fails_and_no_cache -v
```

Expected: both tests FAIL (fallback behavior not yet implemented).

---

### Task 2: Implement the cache fallback in `fetch_historical`

**Files:**
- Modify: `src/mlb_edge_finder/historical_ingestion.py:49-65`

- [ ] **Step 1: Replace the raise-on-failure block with the fallback**

Find this block (after the retry loop):

```python
    if games is None:
        raise RuntimeError(f"statsapi.schedule failed for season {season}: {last_exc}") from last_exc
```

Replace it with:

```python
    if games is None:
        if cache_path.exists():
            logger.warning(
                "statsapi.schedule failed for season %d after %d attempts (%s) — using stale cache",
                season, total, last_exc,
            )
            return load_cached_historical(season)
        raise RuntimeError(
            f"statsapi.schedule failed for season {season}: {last_exc}"
        ) from last_exc
```

- [ ] **Step 2: Run the new tests to confirm they pass**

```bash
pytest tests/test_historical_ingestion.py::test_fetch_historical_falls_back_to_cache_when_api_fails tests/test_historical_ingestion.py::test_fetch_historical_raises_when_api_fails_and_no_cache -v
```

Expected: both tests PASS.

- [ ] **Step 3: Run the full test suite to confirm no regressions**

```bash
pytest tests/ -v
```

Expected: all 173 tests PASS (plus the 2 new ones = 175 total).

- [ ] **Step 4: Commit**

```bash
git add src/mlb_edge_finder/historical_ingestion.py tests/test_historical_ingestion.py
git commit -m "feat: fall back to stale cache when statsapi is unavailable in fetch_historical

If all 3 retry attempts fail and a local cache exists, log a warning and
return the cached data rather than raising. Raises only when no cache is
present. Prevents the daily workflow from failing on transient MLB Stats
API outages (e.g. 503 timeouts).

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```
