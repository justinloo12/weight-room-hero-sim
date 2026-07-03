# Homerun Model

An MLB home-run prediction model built on Statcast data. It estimates each
hitter's probability of homering in today's games, compares that probability
against sportsbook home-run prop odds to find positive-edge bets, renders a
static HTML dashboard, and sends Discord alerts when a tracked pick actually
goes deep.

> **This project is for educational purposes only.** See the
> [disclaimer](#disclaimer) at the bottom.

## Architecture

```
pull_data.py ──> homerun_data_all.csv        (raw Statcast pitch-level data)
        │
build_features.py ──> homerun_data_enriched.csv,
        │             hitter_profiles.csv, pitcher_profiles.csv
        │
train_model.py ──> hr_model.pkl, model_features.csv
        │
dashboard.py ──> index.html + picks_history.csv   (daily picks + edges)
        │
alert_homers.py (GitHub Actions cron) ──> Discord alerts when picks homer
check_results.py ──> resolves picks_history.csv results and P&L
```

- **`pull_data.py`** — pulls pitch-level Statcast data for the 2025 and 2026
  seasons via `pybaseball` and writes `homerun_data_all.csv`.
- **`build_features.py`** — engineers per-hitter and per-pitcher profiles
  (barrel rate, exit velocity, launch angle, platoon splits, pitch-mix
  vulnerability, recent form, batter-vs-pitcher history, park/context) and
  writes the enriched training set plus `hitter_profiles.csv` /
  `pitcher_profiles.csv`.
- **`train_model.py`** — trains the classifier (see below) and saves the
  fitted pipeline to `hr_model.pkl` with its feature list in
  `model_features.csv`.
- **`dashboard.py`** — the main daily driver. Loads the model and profiles,
  fetches today's matchups, weather, and lineups, pulls home-run prop odds
  from The Odds API, computes each hitter's model probability and edge vs the
  book, sizes stakes with fractional Kelly, appends tracked picks to
  `picks_history.csv`, and writes a self-contained `index.html` dashboard.
- **`alert_homers.py`** — runs every 15 minutes during game hours via the
  GitHub Actions workflow (`.github/workflows/discord-homer-alerts.yml`).
  Checks the free MLB Stats API for same-day home runs by tracked picks and
  posts a Discord alert for each hit, marking `alert_sent` in
  `picks_history.csv` (the workflow commits the update).
- **`check_results.py`** — resolves past picks (HR / No HR) via the MLB Stats
  API, fills in P&L, and prints hit-rate / ROI summary stats.
- **`betting_math.py`** — small pure-function module (odds conversion, edge,
  expected ROI, Kelly fraction, P&L) shared by the scripts and covered by the
  unit tests.
- **`train_lr_model.py`** — an earlier, simpler logistic-regression baseline
  (`lr_model.pkl` etc.); kept for reference.

## ML approach

- **Model:** scikit-learn `Pipeline` of `StandardScaler` →
  `CalibratedClassifierCV` (sigmoid, 3-fold) wrapping a
  `GradientBoostingClassifier` (shallow trees, low learning rate). Calibration
  matters more than raw ranking here because the output probability is
  compared directly against betting odds.
- **Features:** ~85 engineered pre-game features (see `model_features.csv`)
  across four groups — hitter power profile and recent form, pitcher
  vulnerability and pitch mix, hitter-vs-pitch-type / platoon matchup terms,
  and game context. Only information knowable *before* the game is used.
- **Evaluation:** held-out test split scored with ROC AUC and Brier score
  (printed by `train_model.py`).
- **Edge:** for each hitter, `edge = model probability − book implied
  probability` (both in percent), where implied probability comes from the
  best available American odds across major books. Picks require configurable
  minimum model probability and minimum edge thresholds.
- **Staking:** fractional Kelly — the full-Kelly fraction is scaled by
  `KELLY_FRACTION` (0.20) and capped by per-bet bankroll limits before being
  recorded with each tracked pick.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Configure environment variables (see `.env.example`):

| Variable | Used by | Purpose |
|---|---|---|
| `ODDS_API_KEY` | `dashboard.py` | The Odds API key for HR prop odds |
| `DISCORD_WEBHOOK_URL` | `alert_homers.py` | Discord webhook for live HR alerts |

Locally you can `cp .env.example .env`, fill it in, and export the values
(e.g. `set -a; source .env; set +a`). In GitHub Actions,
`DISCORD_WEBHOOK_URL` is provided as a repository secret by the workflow.

## Running

```bash
# 1. Pull raw Statcast data (large download; writes ~500MB CSV)
python3 pull_data.py

# 2. Build features and player profiles (writes ~1.2GB enriched CSV)
python3 build_features.py

# 3. Train the model
python3 train_model.py

# 4. Generate today's dashboard and picks (requires ODDS_API_KEY)
python3 dashboard.py        # writes index.html, updates picks_history.csv

# Resolve past picks and print hit rate / ROI
python3 check_results.py

# Send Discord alerts for today's picks that homered (requires DISCORD_WEBHOOK_URL)
python3 alert_homers.py
```

The large data files (`homerun_data_*.csv`) are intentionally not committed —
regenerate them with steps 1–2. The trained model (`hr_model.pkl`), player
profiles, and `picks_history.csv` are tracked in the repo (the CI workflow
reads and commits `picks_history.csv`).

## Tests

```bash
python3 -m unittest discover tests
```

The suite covers the deterministic betting math (odds ↔ implied probability,
edge, expected ROI, Kelly fraction, P&L) and runs fully offline.

## Disclaimer

This project exists to explore sports analytics and probability modeling. It
is **not** betting advice. Model probabilities are estimates with real error
bars, past ROI does not predict future results, and sportsbooks build a margin
into every price. If you choose to bet, only wager money you can afford to
lose, respect the laws of your jurisdiction, and seek help if gambling stops
being fun — in the US, call or text 1-800-GAMBLER.
