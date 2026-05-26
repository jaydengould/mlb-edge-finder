# Time-Matched Pitcher Snapshots Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the single end-of-season pitcher stats join with a time-matched multi-snapshot approach (April 30 / June 1 / July 31 / September 28), add a 30-IP minimum floor to both training and inference, and add a GitHub Actions cron workflow to automatically capture and commit snapshots each season.

**Architecture:** `pitcher_ingestion.py` gains a shared `_parse_pitcher_splits` helper, a `fetch_pitcher_snapshot(snapshot_date, force)` function (using the MLB Stats API `byDateRange` stat type), and an IP floor applied in both `fetch_pitcher_stats` and `fetch_pitcher_snapshot`. `training_data.py` gains a `_select_snapshot_date` helper and replaces the single end-of-season pitcher join in `_build_season` with a group-by-snapshot join. A new `snapshot.yml` workflow commits `pitcher_snapshot_YYYY-MM-DD.csv` files on schedule and on demand.

**Tech Stack:** Python 3.12, statsapi, pandas, pytest, GitHub Actions

---

## File Map

| Action | File | What changes |
|---|---|---|
| Modify | `src/mlb_edge_finder/config.py` | Add `MIN_PITCHER_IP = 30` |
| Modify | `src/mlb_edge_finder/pitcher_ingestion.py` | Extract `_parse_pitcher_splits`, add `fetch_pitcher_snapshot`, apply IP floor in `fetch_pitcher_stats` |
| Modify | `src/mlb_edge_finder/training_data.py` | Add `_select_snapshot_date`, rewrite pitcher join in `_build_season` |
| Modify | `tests/test_pitcher_ingestion.py` | New tests for `fetch_pitcher_snapshot` and IP floor |
| Modify | `tests/test_training_data.py` | Update broken pitcher join tests, add snapshot join tests |
| Modify | `.gitignore` | Add `!data/raw/pitcher_snapshot_*.csv` |
| Create | `.github/workflows/snapshot.yml` | Cron + manual-trigger snapshot workflow |

---

## Task 1: Add `MIN_PITCHER_IP` to config

**Files:**
- Modify: `src/mlb_edge_finder/config.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write the failing test**

Open `tests/test_config.py` and add:

```python
def test_min_pitcher_ip_constant():
    from mlb_edge_finder import config
    assert config.MIN_PITCHER_IP == 30
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_config.py::test_min_pitcher_ip_constant -v
```

Expected: `FAILED — AttributeError: module 'mlb_edge_finder.config' has no attribute 'MIN_PITCHER_IP'`

- [ ] **Step 3: Add the constant to config.py**

In `src/mlb_edge_finder/config.py`, after `MIN_AMERICAN_ODDS`:

```python
MIN_PITCHER_IP: int = 30         # exclude pitchers below this threshold from all joins
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_config.py::test_min_pitcher_ip_constant -v
```

Expected: `PASSED`

- [ ] **Step 5: Commit**

```bash
git add src/mlb_edge_finder/config.py tests/test_config.py
git commit -m "feat: add MIN_PITCHER_IP=30 threshold to config"
```

---

## Task 2: Extract `_parse_pitcher_splits` helper in `pitcher_ingestion.py`

This refactor extracts the row-building loop shared by both `fetch_pitcher_stats` and the new `fetch_pitcher_snapshot`. No behavior change — existing tests must still pass.

**Files:**
- Modify: `src/mlb_edge_finder/pitcher_ingestion.py`

- [ ] **Step 1: Extract the helper**

In `src/mlb_edge_finder/pitcher_ingestion.py`, add above `fetch_pitcher_stats`:

```python
def _parse_pitcher_splits(splits: list) -> list[dict]:
    """Parse raw statsapi splits into pitcher row dicts. Skips pitchers with ip == 0."""
    rows = []
    for s in splits:
        player = s.get("player", {})
        st = s.get("stat", {})
        ip_str = st.get("inningsPitched", "0") or "0"
        ip = float(ip_str)
        if ip == 0:
            continue
        hr = int(st.get("homeRuns", 0) or 0)
        bb = int(st.get("baseOnBalls", 0) or 0)
        k_out = int(st.get("strikeOuts", 0) or 0)
        fip = (13 * hr + 3 * bb - 2 * k_out) / ip + _FIP_CONSTANT
        rows.append({
            "pitcher_id": player.get("id"),
            "pitcher_name": player.get("fullName"),
            "era": float(st.get("era", 0) or 0),
            "whip": float(st.get("whip", 0) or 0),
            "k_per_9": float(st.get("strikeoutsPer9Inn", 0) or 0),
            "bb_per_9": float(st.get("walksPer9Inn", 0) or 0),
            "ip": ip,
            "fip_computed": fip,
        })
    return rows
```

Then replace the inline loop inside `fetch_pitcher_stats` with a call to `_parse_pitcher_splits`. The existing function body from `rows = []` through `df = pd.DataFrame(rows)` becomes:

```python
    rows = _parse_pitcher_splits(splits)
    df = pd.DataFrame(rows)
    config.DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path, index=False)
    logger.info("Wrote %d pitchers to %s", len(df), cache_path)
    return df
```

- [ ] **Step 2: Run all pitcher tests to verify no regression**

```bash
pytest tests/test_pitcher_ingestion.py -v
```

Expected: all existing tests pass (behavior unchanged).

- [ ] **Step 3: Commit**

```bash
git add src/mlb_edge_finder/pitcher_ingestion.py
git commit -m "refactor: extract _parse_pitcher_splits helper in pitcher_ingestion"
```

---

## Task 3: Apply IP floor in `fetch_pitcher_stats`

**Files:**
- Modify: `src/mlb_edge_finder/pitcher_ingestion.py`
- Test: `tests/test_pitcher_ingestion.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_pitcher_ingestion.py`:

```python
def test_fetch_pitcher_stats_excludes_low_ip_pitchers(tmp_path):
    """Pitchers with ip < MIN_PITCHER_IP are excluded from fetch_pitcher_stats output."""
    from mlb_edge_finder import pitcher_ingestion, config
    response = {
        "stats": [{
            "splits": [
                {
                    "player": {"id": 1, "fullName": "Low IP Pitcher"},
                    "stat": {
                        "inningsPitched": str(config.MIN_PITCHER_IP - 1),
                        "era": "2.50", "whip": "1.10",
                        "strikeoutsPer9Inn": "9.0", "walksPer9Inn": "2.0",
                        "homeRuns": 1, "baseOnBalls": 5, "strikeOuts": 25,
                    },
                },
                {
                    "player": {"id": 2, "fullName": "Qualified Pitcher"},
                    "stat": {
                        "inningsPitched": str(config.MIN_PITCHER_IP),
                        "era": "3.50", "whip": "1.20",
                        "strikeoutsPer9Inn": "9.0", "walksPer9Inn": "3.0",
                        "homeRuns": 3, "baseOnBalls": 10, "strikeOuts": 50,
                    },
                },
            ]
        }]
    }
    with patch("mlb_edge_finder.pitcher_ingestion.statsapi.get", return_value=response), \
         patch("mlb_edge_finder.pitcher_ingestion.config.DATA_RAW_DIR", tmp_path):
        df = pitcher_ingestion.fetch_pitcher_stats(date(2026, 5, 26), force=True)
    assert len(df) == 1
    assert df.iloc[0]["pitcher_name"] == "Qualified Pitcher"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_pitcher_ingestion.py::test_fetch_pitcher_stats_excludes_low_ip_pitchers -v
```

Expected: `FAILED — AssertionError: assert 2 == 1` (both pitchers currently included)

- [ ] **Step 3: Add IP floor to `fetch_pitcher_stats`**

In `fetch_pitcher_stats`, replace the current block from `df = pd.DataFrame(rows)` through `return df` with:

```python
    df = pd.DataFrame(rows)
    df = df[df["ip"] >= config.MIN_PITCHER_IP].reset_index(drop=True)
    config.DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path, index=False)
    logger.info("Wrote %d pitchers to %s", len(df), cache_path)
    return df
```

- [ ] **Step 4: Run all pitcher tests**

```bash
pytest tests/test_pitcher_ingestion.py -v
```

Expected: all pass. Note: `test_fetch_pitcher_stats_skips_zero_ip` uses ip=50.1 (>= 30) so it still passes.

- [ ] **Step 5: Commit**

```bash
git add src/mlb_edge_finder/pitcher_ingestion.py tests/test_pitcher_ingestion.py
git commit -m "feat: apply MIN_PITCHER_IP floor in fetch_pitcher_stats"
```

---

## Task 4: Add `fetch_pitcher_snapshot` to `pitcher_ingestion.py`

**Files:**
- Modify: `src/mlb_edge_finder/pitcher_ingestion.py`
- Test: `tests/test_pitcher_ingestion.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_pitcher_ingestion.py`:

```python
def test_fetch_pitcher_snapshot_signature():
    from mlb_edge_finder import pitcher_ingestion
    import inspect
    assert callable(pitcher_ingestion.fetch_pitcher_snapshot)
    sig = inspect.signature(pitcher_ingestion.fetch_pitcher_snapshot)
    assert "snapshot_date" in sig.parameters
    assert "force" in sig.parameters


def test_fetch_pitcher_snapshot_writes_to_snapshot_path(tmp_path):
    """fetch_pitcher_snapshot writes to pitcher_snapshot_*.csv, not pitcher_stats_*.csv."""
    from mlb_edge_finder import pitcher_ingestion
    with patch("mlb_edge_finder.pitcher_ingestion.statsapi.get",
               return_value=_make_stats_response(ip="150.0")), \
         patch("mlb_edge_finder.pitcher_ingestion.config.DATA_RAW_DIR", tmp_path):
        pitcher_ingestion.fetch_pitcher_snapshot(date(2026, 4, 30), force=True)
    assert (tmp_path / "pitcher_snapshot_2026-04-30.csv").exists()
    assert not (tmp_path / "pitcher_stats_2026-04-30.csv").exists()


def test_fetch_pitcher_snapshot_excludes_low_ip(tmp_path):
    """fetch_pitcher_snapshot excludes pitchers below MIN_PITCHER_IP."""
    from mlb_edge_finder import pitcher_ingestion, config
    response = {
        "stats": [{
            "splits": [
                {
                    "player": {"id": 1, "fullName": "Low IP"},
                    "stat": {
                        "inningsPitched": str(config.MIN_PITCHER_IP - 1),
                        "era": "2.00", "whip": "1.00",
                        "strikeoutsPer9Inn": "10.0", "walksPer9Inn": "2.0",
                        "homeRuns": 0, "baseOnBalls": 3, "strikeOuts": 20,
                    },
                },
                {
                    "player": {"id": 2, "fullName": "Qualified"},
                    "stat": {
                        "inningsPitched": str(config.MIN_PITCHER_IP + 10),
                        "era": "3.50", "whip": "1.20",
                        "strikeoutsPer9Inn": "9.0", "walksPer9Inn": "3.0",
                        "homeRuns": 3, "baseOnBalls": 12, "strikeOuts": 45,
                    },
                },
            ]
        }]
    }
    with patch("mlb_edge_finder.pitcher_ingestion.statsapi.get", return_value=response), \
         patch("mlb_edge_finder.pitcher_ingestion.config.DATA_RAW_DIR", tmp_path):
        df = pitcher_ingestion.fetch_pitcher_snapshot(date(2026, 4, 30), force=True)
    assert len(df) == 1
    assert df.iloc[0]["pitcher_name"] == "Qualified"


def test_fetch_pitcher_snapshot_cache_first(tmp_path):
    """fetch_pitcher_snapshot returns cached file without calling statsapi."""
    from mlb_edge_finder import pitcher_ingestion
    cached = pd.DataFrame([{
        "pitcher_id": 99, "pitcher_name": "Cached Ace",
        "era": 2.5, "whip": 1.0, "k_per_9": 11.0, "bb_per_9": 2.0,
        "ip": 50.0, "fip_computed": 2.8,
    }])
    (tmp_path / "pitcher_snapshot_2026-04-30.csv").write_text(cached.to_csv(index=False))
    with patch("mlb_edge_finder.pitcher_ingestion.statsapi.get") as mock_get, \
         patch("mlb_edge_finder.pitcher_ingestion.config.DATA_RAW_DIR", tmp_path):
        df = pitcher_ingestion.fetch_pitcher_snapshot(date(2026, 4, 30))
    mock_get.assert_not_called()
    assert df.iloc[0]["pitcher_name"] == "Cached Ace"


def test_fetch_pitcher_snapshot_falls_back_to_season_stats_when_no_splits(tmp_path):
    """When byDateRange returns empty splits, falls back to full-season stats."""
    from mlb_edge_finder import pitcher_ingestion
    empty_response = {"stats": [{"splits": []}]}
    full_season_response = _make_stats_response(ip="150.0")
    responses = [empty_response, full_season_response]
    with patch("mlb_edge_finder.pitcher_ingestion.statsapi.get", side_effect=responses), \
         patch("mlb_edge_finder.pitcher_ingestion.config.DATA_RAW_DIR", tmp_path):
        df = pitcher_ingestion.fetch_pitcher_snapshot(date(2024, 4, 30), force=True)
    assert len(df) == 1
    assert df.iloc[0]["pitcher_name"] == "Gerrit Cole"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_pitcher_ingestion.py::test_fetch_pitcher_snapshot_signature tests/test_pitcher_ingestion.py::test_fetch_pitcher_snapshot_writes_to_snapshot_path -v
```

Expected: `FAILED — AttributeError: module has no attribute 'fetch_pitcher_snapshot'`

- [ ] **Step 3: Implement `fetch_pitcher_snapshot`**

Add after `load_cached_pitcher_stats` in `src/mlb_edge_finder/pitcher_ingestion.py`:

```python
def fetch_pitcher_snapshot(snapshot_date: date, force: bool = False) -> pd.DataFrame:
    """Fetch season-to-date pitching stats as of snapshot_date, filtered to >= MIN_PITCHER_IP.

    Uses the MLB Stats API byDateRange stat type to get cumulative stats from
    season start (March 1) through snapshot_date. Falls back to full-season
    stats if byDateRange returns no splits (e.g. for completed historical seasons
    where the API no longer supports date-range queries).

    Writes to DATA_RAW_DIR/pitcher_snapshot_YYYY-MM-DD.csv, which is committed
    to the repo via the snapshot GitHub Actions workflow.

    Args:
        snapshot_date: Date to snapshot through. Season derived from snapshot_date.year.
        force: If True, re-fetch even if a cache file exists.

    Returns:
        DataFrame with columns: pitcher_id, pitcher_name, era, whip,
        k_per_9, bb_per_9, ip, fip_computed. Only pitchers with ip >= MIN_PITCHER_IP.

    Raises:
        RuntimeError: If all statsapi calls fail.
    """
    cache_path = config.DATA_RAW_DIR / f"pitcher_snapshot_{snapshot_date}.csv"
    if cache_path.exists() and not force:
        logger.debug("Cache hit for pitcher_snapshot %s, loading from disk", snapshot_date)
        return pd.read_csv(cache_path)

    season = snapshot_date.year
    season_start = f"{season}-03-01"
    end_date = snapshot_date.strftime("%Y-%m-%d")

    try:
        data = statsapi.get("stats", {
            "stats": "byDateRange",
            "group": "pitching",
            "sportId": 1,
            "season": season,
            "startDate": season_start,
            "endDate": end_date,
            "playerPool": "All",
            "limit": 5000,
        })
    except Exception as exc:
        raise RuntimeError(
            f"statsapi failed fetching pitcher snapshot for {snapshot_date}: {exc}"
        ) from exc

    splits = data.get("stats", [{}])[0].get("splits", [])
    if not splits:
        logger.warning(
            "byDateRange returned no data for %s — falling back to full-season stats",
            snapshot_date,
        )
        return _fetch_snapshot_from_full_season(season, snapshot_date)

    df = pd.DataFrame(_parse_pitcher_splits(splits))
    df = df[df["ip"] >= config.MIN_PITCHER_IP].reset_index(drop=True)
    config.DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path, index=False)
    logger.info("Wrote %d pitchers to pitcher_snapshot_%s.csv", len(df), snapshot_date)
    return df


def _fetch_snapshot_from_full_season(season: int, snapshot_date: date) -> pd.DataFrame:
    """Fallback: build a snapshot using full-season stats when byDateRange is unavailable."""
    try:
        data = statsapi.get("stats", {
            "stats": "season",
            "group": "pitching",
            "sportId": 1,
            "season": season,
            "playerPool": "All",
            "limit": 5000,
        })
    except Exception as exc:
        raise RuntimeError(
            f"statsapi fallback failed for pitcher snapshot {snapshot_date}: {exc}"
        ) from exc

    splits = data.get("stats", [{}])[0].get("splits", [])
    df = pd.DataFrame(_parse_pitcher_splits(splits))
    df = df[df["ip"] >= config.MIN_PITCHER_IP].reset_index(drop=True)
    cache_path = config.DATA_RAW_DIR / f"pitcher_snapshot_{snapshot_date}.csv"
    config.DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path, index=False)
    logger.info(
        "Wrote %d pitchers (full-season fallback) to pitcher_snapshot_%s.csv",
        len(df), snapshot_date,
    )
    return df
```

- [ ] **Step 4: Run new snapshot tests**

```bash
pytest tests/test_pitcher_ingestion.py -v -k "snapshot"
```

Expected: all 5 new snapshot tests pass.

- [ ] **Step 5: Run full pitcher test suite**

```bash
pytest tests/test_pitcher_ingestion.py -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/mlb_edge_finder/pitcher_ingestion.py tests/test_pitcher_ingestion.py
git commit -m "feat: add fetch_pitcher_snapshot with IP floor and byDateRange API"
```

---

## Task 5: Add `_select_snapshot_date` helper to `training_data.py`

**Files:**
- Modify: `src/mlb_edge_finder/training_data.py`
- Test: `tests/test_training_data.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_training_data.py`:

```python
def test_select_snapshot_date_returns_latest_preceding():
    from mlb_edge_finder.training_data import _select_snapshot_date
    from datetime import date
    available = [date(2024, 4, 30), date(2024, 6, 1), date(2024, 7, 31), date(2024, 9, 28)]
    assert _select_snapshot_date(date(2024, 5, 10), available) == date(2024, 4, 30)
    assert _select_snapshot_date(date(2024, 6, 15), available) == date(2024, 6, 1)
    assert _select_snapshot_date(date(2024, 8, 5), available) == date(2024, 7, 31)
    assert _select_snapshot_date(date(2024, 9, 29), available) == date(2024, 9, 28)


def test_select_snapshot_date_returns_none_when_no_preceding():
    from mlb_edge_finder.training_data import _select_snapshot_date
    from datetime import date
    available = [date(2024, 4, 30), date(2024, 6, 1)]
    assert _select_snapshot_date(date(2024, 4, 15), available) is None
    assert _select_snapshot_date(date(2024, 4, 30), available) is None  # strictly before


def test_select_snapshot_date_empty_available():
    from mlb_edge_finder.training_data import _select_snapshot_date
    from datetime import date
    assert _select_snapshot_date(date(2024, 9, 1), []) is None
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_training_data.py::test_select_snapshot_date_returns_latest_preceding -v
```

Expected: `FAILED — ImportError: cannot import name '_select_snapshot_date'`

- [ ] **Step 3: Add `_select_snapshot_date` to `training_data.py`**

Add near the top of `src/mlb_edge_finder/training_data.py`, after the imports:

```python
def _select_snapshot_date(
    game_date: date, available_dates: list[date]
) -> date | None:
    """Return the latest snapshot date strictly before game_date, or None."""
    preceding = [d for d in available_dates if d < game_date]
    return max(preceding) if preceding else None
```

- [ ] **Step 4: Run helper tests**

```bash
pytest tests/test_training_data.py::test_select_snapshot_date_returns_latest_preceding tests/test_training_data.py::test_select_snapshot_date_returns_none_when_no_preceding tests/test_training_data.py::test_select_snapshot_date_empty_available -v
```

Expected: all 3 pass.

- [ ] **Step 5: Commit**

```bash
git add src/mlb_edge_finder/training_data.py tests/test_training_data.py
git commit -m "feat: add _select_snapshot_date helper to training_data"
```

---

## Task 6: Multi-snapshot pitcher join in `_build_season`

This is the core change. Replace the single `fetch_pitcher_stats(date(season, 9, 28))` call with a snapshot-aware group join. Some existing tests will break and must be updated in this task.

**Files:**
- Modify: `src/mlb_edge_finder/training_data.py`
- Modify: `src/mlb_edge_finder/training_data.py` imports (add `fetch_pitcher_snapshot`)
- Test: `tests/test_training_data.py`

- [ ] **Step 1: Write new snapshot join tests**

Add to `tests/test_training_data.py`:

```python
def _write_pitcher_snapshot(tmp_path, snap_date, pitcher_df):
    """Write a pitcher snapshot CSV to tmp_path for use in training_data tests."""
    path = tmp_path / f"pitcher_snapshot_{snap_date}.csv"
    pitcher_df.to_csv(path, index=False)


def test_build_season_uses_snapshot_for_postgame_date(tmp_path):
    """Games after a snapshot date receive pitcher stats from that snapshot."""
    from mlb_edge_finder import training_data
    from datetime import date

    hist = pd.DataFrame([{
        "game_date": "2024-09-30",
        "home_name": "New York Yankees",
        "away_name": "Boston Red Sox",
        "home_score": 5, "away_score": 3, "home_win": 1,
        "home_starter_name": "Cole Pitcher",
        "away_starter_name": "Bello Pitcher",
    }])
    _write_pitcher_snapshot(tmp_path, date(2024, 9, 28), _make_pitcher_stats())

    with patch("mlb_edge_finder.training_data.load_cached_historical", return_value=hist), \
         patch("mlb_edge_finder.training_data.fetch_stats", return_value=_make_stats()), \
         patch("mlb_edge_finder.training_data.config.DATA_RAW_DIR", tmp_path), \
         patch("mlb_edge_finder.training_data.config.DATA_PROCESSED_DIR", tmp_path):
        df = training_data.build_training_set([2024], force=True)

    assert len(df) == 1
    assert abs(df.iloc[0]["home_sp_era"] - 3.50) < 0.01
    assert abs(df.iloc[0]["away_sp_era"] - 4.00) < 0.01


def test_build_season_pitcher_nan_for_pre_snapshot_game(tmp_path):
    """Games before the first snapshot get NaN pitcher stats."""
    from mlb_edge_finder import training_data
    from datetime import date

    hist = pd.DataFrame([{
        "game_date": "2024-04-01",
        "home_name": "New York Yankees",
        "away_name": "Boston Red Sox",
        "home_score": 5, "away_score": 3, "home_win": 1,
        "home_starter_name": "Cole Pitcher",
        "away_starter_name": "Bello Pitcher",
    }])
    # No snapshot files in tmp_path — September 28 fallback would be called,
    # but April 1 is before September 28 so it still gets NaN.
    with patch("mlb_edge_finder.training_data.load_cached_historical", return_value=hist), \
         patch("mlb_edge_finder.training_data.fetch_stats", return_value=_make_stats()), \
         patch("mlb_edge_finder.training_data.fetch_pitcher_stats", return_value=_make_pitcher_stats()), \
         patch("mlb_edge_finder.training_data.config.DATA_RAW_DIR", tmp_path), \
         patch("mlb_edge_finder.training_data.config.DATA_PROCESSED_DIR", tmp_path):
        df = training_data.build_training_set([2024], force=True)

    assert len(df) == 1
    assert pd.isna(df.iloc[0]["home_sp_era"])
    assert pd.isna(df.iloc[0]["away_sp_era"])


def test_build_season_selects_correct_snapshot_for_each_game(tmp_path):
    """Each game uses the latest snapshot strictly before its game_date."""
    from mlb_edge_finder import training_data
    from datetime import date

    snap_april = _make_pitcher_stats().copy()
    snap_april["era"] = 2.00  # distinctive value for April snapshot

    snap_june = _make_pitcher_stats().copy()
    snap_june["era"] = 5.00  # distinctive value for June snapshot

    hist = pd.DataFrame([
        {
            "game_date": "2024-05-10",  # → April 30 snapshot (era=2.00)
            "home_name": "New York Yankees", "away_name": "Boston Red Sox",
            "home_score": 5, "away_score": 3, "home_win": 1,
            "home_starter_name": "Cole Pitcher", "away_starter_name": "Bello Pitcher",
        },
        {
            "game_date": "2024-06-15",  # → June 1 snapshot (era=5.00)
            "home_name": "New York Yankees", "away_name": "Boston Red Sox",
            "home_score": 3, "away_score": 2, "home_win": 1,
            "home_starter_name": "Cole Pitcher", "away_starter_name": "Bello Pitcher",
        },
    ])
    _write_pitcher_snapshot(tmp_path, date(2024, 4, 30), snap_april)
    _write_pitcher_snapshot(tmp_path, date(2024, 6, 1), snap_june)

    with patch("mlb_edge_finder.training_data.load_cached_historical", return_value=hist), \
         patch("mlb_edge_finder.training_data.fetch_stats", return_value=_make_stats()), \
         patch("mlb_edge_finder.training_data.config.DATA_RAW_DIR", tmp_path), \
         patch("mlb_edge_finder.training_data.config.DATA_PROCESSED_DIR", tmp_path):
        df = training_data.build_training_set([2024], force=True)

    df = df.sort_values("game_date").reset_index(drop=True)
    assert abs(df.iloc[0]["home_sp_era"] - 2.00) < 0.01  # May game → April snapshot
    assert abs(df.iloc[1]["home_sp_era"] - 5.00) < 0.01  # June game → June snapshot
```

- [ ] **Step 2: Run new tests to verify they fail**

```bash
pytest tests/test_training_data.py::test_build_season_uses_snapshot_for_postgame_date tests/test_training_data.py::test_build_season_pitcher_nan_for_pre_snapshot_game tests/test_training_data.py::test_build_season_selects_correct_snapshot_for_each_game -v
```

Expected: all 3 fail (current code uses a single `fetch_pitcher_stats` call, not snapshot-based join).

- [ ] **Step 3: Define snapshot dates constant in `training_data.py`**

Add after the `_SNAPSHOT_MONTH`/`_SNAPSHOT_DAY` constants:

```python
_PITCHER_SNAPSHOT_MONTH_DAYS: list[tuple[int, int]] = [
    (4, 30), (6, 1), (7, 31), (9, 28),
]
```

- [ ] **Step 5: Replace the pitcher join block in `_build_season`**

Find the section in `_build_season` that starts with:
```python
    # Join starting pitcher season stats with home_sp_*/away_sp_* prefix
    pitcher_stats = fetch_pitcher_stats(date(season, _SNAPSHOT_MONTH, _SNAPSHOT_DAY))
```
...and ends with the `logger.debug` call for pitcher join match counts.

Replace the entire block with:

```python
    # Load available pitcher snapshots for this season into {snapshot_date: DataFrame}.
    # Snapshot files (pitcher_snapshot_YYYY-MM-DD.csv) are committed to the repo by
    # the GitHub Actions snapshot workflow. September 28 falls back to fetch_pitcher_stats
    # for seasons that predate the snapshot workflow.
    snapshot_dates = [
        date(season, m, d) for m, d in _PITCHER_SNAPSHOT_MONTH_DAYS
    ]
    pitcher_snapshots: dict[date, pd.DataFrame] = {}
    for snap_date in snapshot_dates:
        snap_path = config.DATA_RAW_DIR / f"pitcher_snapshot_{snap_date}.csv"
        if snap_path.exists():
            pitcher_snapshots[snap_date] = pd.read_csv(snap_path)
        elif snap_date == date(season, 9, 28):
            pitcher_snapshots[snap_date] = fetch_pitcher_stats(
                date(season, _SNAPSHOT_MONTH, _SNAPSHOT_DAY)
            )

    available_dates = sorted(pitcher_snapshots.keys())

    # Assign each game to its snapshot (latest snapshot strictly before game_date).
    game_dates_as_date = pd.to_datetime(df["game_date"]).dt.date
    df["_snap"] = game_dates_as_date.map(
        lambda gd: _select_snapshot_date(gd, available_dates)
    )

    # Determine pitcher column names from any available snapshot.
    any_snap_df = next(iter(pitcher_snapshots.values()), pd.DataFrame())
    sp_cols = [c for c in any_snap_df.columns if c not in ("pitcher_name", "pitcher_id")]
    home_sp_cols = [f"home_sp_{c}" for c in sp_cols]
    away_sp_cols = [f"away_sp_{c}" for c in sp_cols]
    all_pitcher_cols = ["home_pitcher_id", "away_pitcher_id"] + home_sp_cols + away_sp_cols

    groups: list[pd.DataFrame] = []

    # Games with no preceding snapshot → NaN pitcher stats.
    no_snap = df[df["_snap"].isna()].copy()
    if not no_snap.empty:
        for col in all_pitcher_cols:
            no_snap[col] = float("nan")
        groups.append(no_snap)

    # Games with a snapshot → join pitcher stats from the matched snapshot.
    for snap_date, pitcher_stats in pitcher_snapshots.items():
        mask = df["_snap"] == snap_date
        if not mask.any():
            continue
        grp = df[mask].copy()
        home_pitcher = pitcher_stats.rename(columns={
            "pitcher_name": "home_starter_name",
            "pitcher_id": "home_pitcher_id",
            **{c: f"home_sp_{c}" for c in sp_cols},
        })
        away_pitcher = pitcher_stats.rename(columns={
            "pitcher_name": "away_starter_name",
            "pitcher_id": "away_pitcher_id",
            **{c: f"away_sp_{c}" for c in sp_cols},
        })
        home_cols = ["home_starter_name", "home_pitcher_id"] + home_sp_cols
        away_cols = ["away_starter_name", "away_pitcher_id"] + away_sp_cols
        grp = grp.merge(home_pitcher[home_cols], on="home_starter_name", how="left")
        grp = grp.merge(away_pitcher[away_cols], on="away_starter_name", how="left")
        groups.append(grp)

    df = pd.concat(groups, ignore_index=True).drop(columns=["_snap"])

    logger.debug(
        "Season %d: pitcher join — %d/%d home starters matched, %d/%d away starters matched",
        season,
        df["home_pitcher_id"].notna().sum(), len(df),
        df["away_pitcher_id"].notna().sum(), len(df),
    )
```

- [ ] **Step 6: Update the existing broken pitcher join tests**

The tests `test_build_training_set_includes_pitcher_sp_cols`, `test_build_training_set_pitcher_join_values_correct`, and `test_build_training_set_keeps_starter_name_columns` use `game_date="2024-04-01"` which now pre-dates all snapshots and gets NaN pitcher stats. Update them to use a September 30 game date with a snapshot file.

Replace `_make_hist_with_starters` with this version:

```python
def _make_hist_with_starters(home="New York Yankees", away="Boston Red Sox",
                              home_starter="Cole Pitcher", away_starter="Bello Pitcher",
                              game_date="2024-09-30"):
    return pd.DataFrame([{
        "game_date": game_date,
        "home_name": home,
        "away_name": away,
        "home_score": 5,
        "away_score": 3,
        "home_win": 1,
        "home_starter_name": home_starter,
        "away_starter_name": away_starter,
    }])
```

Update `test_build_training_set_includes_pitcher_sp_cols`:

```python
def test_build_training_set_includes_pitcher_sp_cols(tmp_path):
    from mlb_edge_finder import training_data
    from datetime import date
    _write_pitcher_snapshot(tmp_path, date(2024, 9, 28), _make_pitcher_stats())
    with patch("mlb_edge_finder.training_data.load_cached_historical",
               return_value=_make_hist_with_starters()), \
         patch("mlb_edge_finder.training_data.fetch_stats", return_value=_make_stats()), \
         patch("mlb_edge_finder.training_data.config.DATA_RAW_DIR", tmp_path), \
         patch("mlb_edge_finder.training_data.config.DATA_PROCESSED_DIR", tmp_path):
        df = training_data.build_training_set([2024], force=True)
    for col in ("home_sp_era", "away_sp_era", "home_sp_fip_computed", "away_sp_fip_computed",
                "home_sp_k_per_9", "away_sp_k_per_9"):
        assert col in df.columns, f"Missing column: {col}"
```

Update `test_build_training_set_pitcher_join_values_correct`:

```python
def test_build_training_set_pitcher_join_values_correct(tmp_path):
    from mlb_edge_finder import training_data
    from datetime import date
    _write_pitcher_snapshot(tmp_path, date(2024, 9, 28), _make_pitcher_stats())
    with patch("mlb_edge_finder.training_data.load_cached_historical",
               return_value=_make_hist_with_starters()), \
         patch("mlb_edge_finder.training_data.fetch_stats", return_value=_make_stats()), \
         patch("mlb_edge_finder.training_data.config.DATA_RAW_DIR", tmp_path), \
         patch("mlb_edge_finder.training_data.config.DATA_PROCESSED_DIR", tmp_path):
        df = training_data.build_training_set([2024], force=True)
    assert abs(df.iloc[0]["home_sp_era"] - 3.50) < 0.01
    assert abs(df.iloc[0]["away_sp_era"] - 4.00) < 0.01
```

Update `test_build_training_set_pitcher_nan_when_starter_absent`:

```python
def test_build_training_set_pitcher_nan_when_starter_absent(tmp_path):
    from mlb_edge_finder import training_data
    from datetime import date
    hist = _make_hist_with_starters(home_starter=None, away_starter=None)
    _write_pitcher_snapshot(tmp_path, date(2024, 9, 28), _make_pitcher_stats())
    with patch("mlb_edge_finder.training_data.load_cached_historical", return_value=hist), \
         patch("mlb_edge_finder.training_data.fetch_stats", return_value=_make_stats()), \
         patch("mlb_edge_finder.training_data.config.DATA_RAW_DIR", tmp_path), \
         patch("mlb_edge_finder.training_data.config.DATA_PROCESSED_DIR", tmp_path):
        df = training_data.build_training_set([2024], force=True)
    assert pd.isna(df.iloc[0]["home_sp_era"])
    assert pd.isna(df.iloc[0]["away_sp_era"])
```

Update `test_build_training_set_keeps_starter_name_columns`:

```python
def test_build_training_set_keeps_starter_name_columns(tmp_path):
    from mlb_edge_finder import training_data
    from datetime import date
    _write_pitcher_snapshot(tmp_path, date(2024, 9, 28), _make_pitcher_stats())
    with patch("mlb_edge_finder.training_data.load_cached_historical",
               return_value=_make_hist_with_starters()), \
         patch("mlb_edge_finder.training_data.fetch_stats", return_value=_make_stats()), \
         patch("mlb_edge_finder.training_data.config.DATA_RAW_DIR", tmp_path), \
         patch("mlb_edge_finder.training_data.config.DATA_PROCESSED_DIR", tmp_path):
        df = training_data.build_training_set([2024], force=True)
    assert "home_starter_name" in df.columns
    assert "away_starter_name" in df.columns
```

- [ ] **Step 7: Run all training_data tests**

```bash
pytest tests/test_training_data.py -v
```

Expected: all tests pass, including the 3 new snapshot tests and 4 updated pitcher tests.

- [ ] **Step 8: Run full test suite to check for regressions**

```bash
pytest tests/ -v
```

Expected: all 183 + new tests pass.

- [ ] **Step 9: Commit**

```bash
git add src/mlb_edge_finder/training_data.py tests/test_training_data.py
git commit -m "feat: replace single pitcher join with time-matched snapshot join in _build_season"
```

---

## Task 7: Update `.gitignore` and add `snapshot.yml` workflow

**Files:**
- Modify: `.gitignore`
- Create: `.github/workflows/snapshot.yml`

- [ ] **Step 1: Update `.gitignore`**

In `.gitignore`, after the line `!data/raw/historical_*.csv`, add:

```
!data/raw/pitcher_snapshot_*.csv
```

- [ ] **Step 2: Create `.github/workflows/snapshot.yml`**

```yaml
name: Pitcher Stats Snapshot

on:
  schedule:
    - cron: '30 14 30 4 *'   # April 30, 2:30 PM UTC (10:30 AM EDT)
    - cron: '30 14 1 6 *'    # June 1,  2:30 PM UTC
    - cron: '30 14 31 7 *'   # July 31, 2:30 PM UTC
  workflow_dispatch:
    inputs:
      snapshot_date:
        description: 'Snapshot date (YYYY-MM-DD). Leave blank to use today (for backfilling past dates, enter the historical date).'
        required: false
        default: ''

permissions:
  contents: write

jobs:
  snapshot:
    runs-on: ubuntu-latest
    timeout-minutes: 10

    steps:
      - name: Checkout repo
        uses: actions/checkout@v4
        with:
          token: ${{ secrets.GITHUB_TOKEN }}

      - name: Set up Python 3.12
        uses: actions/setup-python@v5
        with:
          python-version: '3.12'
          cache: 'pip'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -e .

      - name: Resolve snapshot date
        id: resolve_date
        run: |
          SNAP_DATE="${{ github.event.inputs.snapshot_date }}"
          if [ -z "$SNAP_DATE" ]; then
            SNAP_DATE=$(date -u +%Y-%m-%d)
          fi
          echo "SNAP_DATE=$SNAP_DATE" >> $GITHUB_ENV
          echo "Snapshot date: $SNAP_DATE"

      - name: Fetch pitcher snapshot
        run: |
          python3 << 'PYEOF'
          import os
          from datetime import date
          from mlb_edge_finder.pitcher_ingestion import fetch_pitcher_snapshot
          snap = date.fromisoformat(os.environ["SNAP_DATE"])
          df = fetch_pitcher_snapshot(snap, force=True)
          print(f"Snapshot written: {len(df)} qualified pitchers for {snap}")
          PYEOF

      - name: Commit snapshot
        run: |
          git config user.name "github-actions[bot]"
          git config user.email "github-actions[bot]@users.noreply.github.com"
          git add "data/raw/pitcher_snapshot_${SNAP_DATE}.csv"
          if git diff --staged --quiet; then
            echo "Nothing to commit (snapshot unchanged)."
          else
            git commit -m "chore: pitcher snapshot ${SNAP_DATE}"
            git push origin HEAD:${{ github.ref_name }}
          fi
```

- [ ] **Step 3: Verify files look correct**

```bash
cat .gitignore | grep pitcher
cat .github/workflows/snapshot.yml | head -20
```

Expected output includes:
```
!data/raw/pitcher_snapshot_*.csv
name: Pitcher Stats Snapshot
```

- [ ] **Step 4: Commit**

```bash
git add .gitignore .github/workflows/snapshot.yml
git commit -m "feat: add snapshot.yml workflow and unignore pitcher_snapshot_*.csv"
```

---

## Task 8: Backfill snapshots and retrain

These are operational steps, not code changes. Run them in sequence after all code tasks are pushed.

- [ ] **Step 1: Push the branch and verify snapshot.yml appears in GitHub Actions**

```bash
git push origin main
```

Go to `https://github.com/<your-repo>/actions` and confirm `Pitcher Stats Snapshot` appears in the workflow list.

- [ ] **Step 2: Backfill the April 30, 2026 snapshot**

In GitHub Actions → "Pitcher Stats Snapshot" → "Run workflow":
- `snapshot_date`: `2026-04-30`
- Click "Run workflow"

Wait for it to complete. Verify `data/raw/pitcher_snapshot_2026-04-30.csv` was committed.

- [ ] **Step 3: Backfill historical seasons locally**

For seasons 2019, 2021–2025, run `fetch_pitcher_snapshot` locally for each of the four snapshot dates. These will use the fallback full-season stats (since historical stats can't be fetched by date range), but with the IP floor applied:

```bash
python3 << 'PYEOF'
from datetime import date
from mlb_edge_finder.pitcher_ingestion import fetch_pitcher_snapshot

seasons = [2019, 2021, 2022, 2023, 2024, 2025]
snapshot_month_days = [(4, 30), (6, 1), (7, 31), (9, 28)]

for season in seasons:
    for m, d in snapshot_month_days:
        snap_date = date(season, m, d)
        try:
            df = fetch_pitcher_snapshot(snap_date, force=True)
            print(f"OK  {snap_date}: {len(df)} pitchers")
        except Exception as e:
            print(f"ERR {snap_date}: {e}")
PYEOF
```

Expected: 24 snapshot files written to `data/raw/`.

- [ ] **Step 4: Commit historical snapshots**

```bash
git add data/raw/pitcher_snapshot_*.csv
git commit -m "chore: backfill pitcher snapshots for 2019, 2021-2025"
git push origin main
```

- [ ] **Step 5: Rebuild training set with new snapshots**

```bash
python3 << 'PYEOF'
import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.insert(0, "src")
from mlb_edge_finder.training_data import build_training_set
from mlb_edge_finder.feedback import _TRAINING_SEASONS
df = build_training_set(_TRAINING_SEASONS, force=True)
print(f"Training set: {len(df)} rows, {len(df.columns)} columns")
PYEOF
```

- [ ] **Step 6: Retrain and save the model**

```bash
python3 << 'PYEOF'
import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.insert(0, "src")
from datetime import date
from mlb_edge_finder.training_data import load_training_set
from mlb_edge_finder.feedback import _TRAINING_SEASONS
from mlb_edge_finder import model

df = load_training_set(_TRAINING_SEASONS)
clf_raw, X_val, X_test, y_val, y_test = model.train(df)
clf = model.calibrate(clf_raw, X_val, y_val)
metrics = model.evaluate(clf, X_test, y_test)
print("Metrics:", metrics)
model.save_model(clf, metrics, date.today())
PYEOF
```

- [ ] **Step 7: Run the full pipeline on today's date to verify edges look reasonable**

```bash
python -m mlb_edge_finder
```

Check that:
- `model_prob` values on any edges are < 0.80
- `prob_flag` is `False` on all edges
- Total edge count is 0–2 (acceptable)

- [ ] **Step 8: Commit new model**

```bash
git add models/
git commit -m "feat: retrain model with time-matched pitcher snapshots and IP floor"
git push origin main
```
