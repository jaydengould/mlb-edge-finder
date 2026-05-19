# GitHub Actions Daily Workflow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a GitHub Actions workflow that runs `python -m mlb_edge_finder` at 9:30 AM ET daily and commits the resulting edges CSV to `outputs/edges_YYYY-MM-DD.csv`.

**Architecture:** A single workflow YAML file triggers on a cron schedule, installs deps, runs the existing CLI, and uses a shell script to promote the output file from `data/processed/` (gitignored) to `outputs/` (committed). No code changes to the pipeline itself — the workflow is purely infrastructure.

**Tech Stack:** GitHub Actions, `actions/checkout@v4`, `actions/setup-python@v5`, bash.

---

## File Map

| File | Action | Purpose |
|---|---|---|
| `requirements.txt` | Modify | Add `MLB-StatsAPI>=1.7` — currently missing, workflow `pip install` will fail without it |
| `outputs/.gitkeep` | Create | Tracks the `outputs/` directory in git before any edges files exist |
| `.github/workflows/daily.yml` | Create | Workflow definition — cron trigger, install, run, commit |

---

### Task 1: Add MLB-StatsAPI to requirements.txt

The workflow runner does a fresh `pip install -r requirements.txt` on every run. `statsapi` is imported by `historical_ingestion.py`, `pitcher_ingestion.py`, and `stats_ingestion.py` but is not declared in `requirements.txt`. The pipeline will crash on the runner without it.

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Add the dependency**

Open `requirements.txt` and add one line so it reads:

```
pandas>=2.0
pybaseball>=2.2
requests>=2.31
scikit-learn>=1.4
xgboost>=2.0
python-dotenv>=1.0
MLB-StatsAPI>=1.7
jupyter>=1.0
notebook>=7.0
pytest>=8.0
```

- [ ] **Step 2: Verify the package name resolves**

```bash
pip install MLB-StatsAPI>=1.7
```

Expected: already satisfied (it's installed locally as `MLB-StatsAPI 1.9.0`).

- [ ] **Step 3: Commit**

```bash
git add requirements.txt
git commit -m "fix: add MLB-StatsAPI to requirements.txt for CI"
```

---

### Task 2: Create the outputs/ directory

`outputs/` must exist in the repo so the workflow can write into it after checkout. A `.gitkeep` file is the convention for tracking an otherwise-empty directory.

**Files:**
- Create: `outputs/.gitkeep`

- [ ] **Step 1: Create the directory and placeholder file**

```bash
mkdir -p outputs
touch outputs/.gitkeep
```

- [ ] **Step 2: Verify outputs/ is not gitignored**

```bash
git check-ignore -v outputs/
```

Expected: no output (meaning it is NOT ignored). If output appears, remove the relevant line from `.gitignore`.

- [ ] **Step 3: Commit**

```bash
git add outputs/.gitkeep
git commit -m "chore: add outputs/ directory for daily edges files"
```

---

### Task 3: Create .github/workflows/daily.yml

This is the core of the feature. The workflow:
- Triggers on cron and on manual dispatch (for testing)
- Grants write permission so it can push the edges file back
- Installs deps, runs the CLI, promotes the output, commits if new

**Files:**
- Create: `.github/workflows/daily.yml`

- [ ] **Step 1: Create the workflows directory**

```bash
mkdir -p .github/workflows
```

- [ ] **Step 2: Write the workflow file**

Create `.github/workflows/daily.yml` with exactly this content:

```yaml
name: Daily MLB Edge Finder

on:
  schedule:
    - cron: '30 13 * * *'   # 9:30 AM EDT (UTC-4, regular season)
  workflow_dispatch:          # allows manual trigger from the Actions tab

permissions:
  contents: write

jobs:
  find-edges:
    runs-on: ubuntu-latest

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

      - name: Run pipeline
        env:
          ODDS_API_KEY: ${{ secrets.ODDS_API_KEY }}
        run: python -m mlb_edge_finder

      - name: Promote edges file to outputs/
        run: |
          DATE=$(date -u +%Y-%m-%d)
          EDGES_SRC="data/processed/edges_${DATE}.csv"
          EDGES_DST="outputs/edges_${DATE}.csv"
          if [ -f "$EDGES_SRC" ]; then
            cp "$EDGES_SRC" "$EDGES_DST"
          else
            # Pipeline ran successfully but found no edges — write header-only file
            # so there is still a commit recording that the workflow ran.
            echo "game_id,home_team,away_team,bet_side,american_odds,model_prob,ev,kelly_fraction,prob_flag" > "$EDGES_DST"
          fi

      - name: Commit and push edges file
        run: |
          DATE=$(date -u +%Y-%m-%d)
          git config user.name "github-actions[bot]"
          git config user.email "github-actions[bot]@users.noreply.github.com"
          git add "outputs/edges_${DATE}.csv"
          if git diff --staged --quiet; then
            echo "Edges file unchanged — nothing to commit."
          else
            git commit -m "chore: edges ${DATE}"
            git push
          fi
```

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/daily.yml
git commit -m "feat: add GitHub Actions daily edge-finder workflow"
```

---

### Task 4: Add ODDS_API_KEY secret to GitHub

This is a manual step in the GitHub UI — the secret cannot be set via git or the CLI without the GitHub CLI authenticated.

**Files:** None — this is a GitHub UI action.

- [ ] **Step 1: Open your repo's secrets page**

Go to: `https://github.com/jaydengould/mlb-edge-finder/settings/secrets/actions`

- [ ] **Step 2: Add the secret**

Click **New repository secret**.
- Name: `ODDS_API_KEY`
- Value: paste your Odds API key (same value as in your local `.env` file)

Click **Add secret**.

- [ ] **Step 3: Verify it appears in the list**

The secrets page should now show `ODDS_API_KEY` in the repository secrets list (the value is hidden — that's expected).

---

### Task 5: Push and trigger the workflow manually

Before waiting for the 9:30 AM cron, trigger the workflow manually to verify the full end-to-end path works.

**Files:** None.

- [ ] **Step 1: Push all commits to remote**

```bash
git push origin main
```

- [ ] **Step 2: Trigger a manual run**

Go to: `https://github.com/jaydengould/mlb-edge-finder/actions/workflows/daily.yml`

Click **Run workflow** → **Run workflow** (branch: main).

- [ ] **Step 3: Watch the run**

Click into the running job. Each step should go green. Expected output in the "Run pipeline" step:

```
No edges found for YYYY-MM-DD.
```

or

```
Found N edge(s) for YYYY-MM-DD:
...
```

- [ ] **Step 4: Verify the commit was made**

Go to: `https://github.com/jaydengould/mlb-edge-finder/commits/main`

You should see a new commit from `github-actions[bot]` with message `chore: edges YYYY-MM-DD`.

- [ ] **Step 5: Verify the edges file in the repo**

Go to: `https://github.com/jaydengould/mlb-edge-finder/tree/main/outputs`

You should see `edges_YYYY-MM-DD.csv`. Click it — it should contain either the header + edge rows, or just the header (no edges today).

---

## Notes

- **Cron timing:** `30 13 * * *` UTC = 9:30 AM EDT. During EST (Nov–Mar, off-season), this fires at 8:30 AM ET. Adjust to `30 14 * * *` if you want 9:30 AM year-round, but EDT coverage is what matters for the regular season.
- **Runner timezone:** The runner is UTC. `date -u +%Y-%m-%d` in the shell and `date.today()` in Python (on a UTC machine) both return the same date at 13:30 UTC — no mismatch.
- **Cache hits:** `actions/setup-python@v5` with `cache: 'pip'` caches the pip download cache between runs. The first run installs everything fresh (~2–3 min); subsequent runs are faster.
- **Model file:** The committed `models/xgb_*.pkl` is loaded by `pipeline.run()` via auto-discovery (glob sorted by filename date). No retraining occurs.
- **data/ directories:** `data/raw/`, `data/processed/`, and `logs/` are gitignored but are created on-the-fly by the pipeline via `mkdir(parents=True, exist_ok=True)` calls in config and each module.
