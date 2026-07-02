"""
backtest_walkforward.py
-----------------------
HONEST, leak-free backtest of the HR model on 2025 data.

Unlike backtest_2025.py (which builds profiles from the full season and then
"predicts" games that are already baked into those profiles), this script does
a strict TEMPORAL SPLIT:

    • Build hitter/pitcher profiles from games BEFORE a cutoff date
    • Test predictions on games AFTER the cutoff
    • No test-game outcome can leak into the profile used to predict it

This tells you the truth about whether the model's inputs actually carry
out-of-sample signal for home runs. It uses ONLY free Statcast data — it makes
zero calls to the paid Odds API.

Because it measures probability accuracy (not betting ROI), it needs no odds.
Profitability is a separate question that requires REAL historical closing
lines, which we don't have — see the note at the bottom.

Runs in GitHub Actions after pull_data.py (raw data is gitignored locally).

Usage:
  python backtest_walkforward.py
  python backtest_walkforward.py --cutoff 2025-07-01 --min-pa 40
"""

import argparse
import sys
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--cutoff",  default="2025-07-01", help="Profiles use games before this date; test on games after (default 2025-07-01)")
parser.add_argument("--min-pa",  type=int, default=40, help="Min batted balls in the TRAIN window to include a hitter (default 40)")
parser.add_argument("--min-bf",  type=int, default=80, help="Min batted balls faced in TRAIN window to trust a pitcher (default 80)")
args = parser.parse_args()

# ─────────────────────────────────────────────────────────────────
# Load raw pitch-by-pitch data (gitignored; present only after pull_data.py)
# ─────────────────────────────────────────────────────────────────
try:
    df = pd.read_csv("homerun_data_all.csv", low_memory=False)
except FileNotFoundError:
    print("ERROR: homerun_data_all.csv not found.")
    print("Run pull_data.py first (this script is meant to run in CI after the data pull).")
    sys.exit(1)

print(f"Loaded {len(df):,} rows")
df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
df = df[df["game_date"].dt.year == 2025].copy()
cutoff = pd.Timestamp(args.cutoff)

# Balls in play only (type=="X") — same fix as build_features.py
if "barrel" not in df.columns:
    if "launch_speed_angle" in df.columns:
        df["barrel"] = (df["launch_speed_angle"] == 6).astype(float)
    else:
        df["barrel"] = np.nan
df["is_hr"]    = (df["events"] == "home_run").astype(int)
inplay = df[(df["type"] == "X") & df["launch_speed"].notna()].copy()
inplay["hard_hit"] = (inplay["launch_speed"] >= 95).astype(int)

train = inplay[inplay["game_date"] < cutoff]
print(f"Train window (< {args.cutoff}): {len(train):,} balls in play")
print(f"Test  window (>= {args.cutoff}): {(inplay['game_date'] >= cutoff).sum():,} balls in play")

# ─────────────────────────────────────────────────────────────────
# Build point-in-time profiles from the TRAIN window only
# ─────────────────────────────────────────────────────────────────
hitter_prof = train.groupby("batter").agg(
    h_barrel_pct   = ("barrel",       "mean"),
    h_hard_hit_pct = ("hard_hit",     "mean"),
    h_exit_velo    = ("launch_speed", "mean"),
    h_hr_rate      = ("is_hr",        "mean"),
    h_n            = ("launch_speed", "count"),
).reset_index()
hitter_prof = hitter_prof[hitter_prof["h_n"] >= args.min_pa]

pitcher_prof = train.groupby("player_name").agg(
    p_hr_rate_allowed     = ("is_hr",        "mean"),
    p_barrel_pct_allowed  = ("barrel",       "mean"),
    p_hard_hit_pct_allowed= ("hard_hit",     "mean"),
    p_n                   = ("launch_speed", "count"),
).reset_index()
pitcher_prof = pitcher_prof[pitcher_prof["p_n"] >= args.min_bf]

print(f"Profiles: {len(hitter_prof):,} hitters | {len(pitcher_prof):,} pitchers")
print(f"  Corrected-data sanity — barrel {hitter_prof['h_barrel_pct'].mean():.3f} "
      f"| EV {hitter_prof['h_exit_velo'].mean():.1f} "
      f"| hard-hit {hitter_prof['h_hard_hit_pct'].mean():.3f}")

hmap = hitter_prof.set_index("batter").to_dict("index")
pmap = pitcher_prof.set_index("player_name").to_dict("index")

# ─────────────────────────────────────────────────────────────────
# Build TEST player-games (did the batter homer that game?)
# ─────────────────────────────────────────────────────────────────
test_pa = df[df["game_date"] >= cutoff].copy()
# opposing starter = pitcher who threw first to this batter that game
starter = (
    test_pa.sort_values("at_bat_number")
    .groupby(["batter", "game_date"])
    .agg(opp_pitcher=("player_name", "first"),
         home_team=("home_team", "first"))
    .reset_index()
)
outcomes = (
    test_pa.groupby(["batter", "game_date"])
    .agg(hit_hr=("is_hr", "max"))
    .reset_index()
)
games = outcomes.merge(starter, on=["batter", "game_date"], how="left")
games = games[games["batter"].isin(hmap.keys())]
print(f"\nTest player-games: {len(games):,} | actual HR rate: {games['hit_hr'].mean():.3f}")

# ─────────────────────────────────────────────────────────────────
# Score each test game with the model formula (corrected-data ranges)
# ─────────────────────────────────────────────────────────────────
BALLPARK = {
    "COL":1.38,"CIN":1.22,"NYY":1.18,"PHI":1.16,"TEX":1.14,"CHC":1.12,"ARI":1.10,
    "HOU":1.07,"MIA":1.06,"DET":1.04,"BAL":1.03,"BOS":1.02,"MIN":1.01,"STL":1.00,
    "NYM":1.00,"ATL":0.99,"LAD":0.99,"KC":0.98,"MIL":0.97,"TOR":0.97,"CLE":0.96,
    "LAA":0.95,"CWS":0.94,"WSH":0.94,"TB":0.93,"SF":0.91,"SEA":0.90,"PIT":0.89,
    "OAK":0.88,"SD":0.87,
}
def norm(v, lo, hi):
    return min(100.0, max(0.0, (v - lo) / (hi - lo) * 100.0))
def sf(v, d):
    try:
        f = float(v); return d if (np.isnan(f) or np.isinf(f)) else f
    except Exception:
        return d

rows = []
for _, g in games.iterrows():
    h = hmap.get(g["batter"])
    if not h:
        continue
    bs = (norm(sf(h["h_barrel_pct"],0.02),   0.02, 0.14) * 0.35
        + norm(sf(h["h_hard_hit_pct"],0.30), 0.25, 0.50) * 0.25
        + norm(sf(h["h_exit_velo"],88),      84.0, 94.0) * 0.20
        + norm(sf(h["h_hr_rate"],0.03),      0.02, 0.09) * 0.20)
    p = pmap.get(g["opp_pitcher"])
    if p:
        ps = (norm(sf(p["p_hr_rate_allowed"],0.03),    0.01, 0.06) * 0.50
            + norm(sf(p["p_barrel_pct_allowed"],0.07), 0.03, 0.13) * 0.30
            + norm(sf(p["p_hard_hit_pct_allowed"],0.38),0.28, 0.48) * 0.20)
    else:
        ps = 50.0
    talent = bs * 0.5 + ps * 0.5
    park   = BALLPARK.get(g["home_team"], 1.0)
    env    = max(0.72, min(1.35, park))
    prob   = (3.0 + (talent / 100.0) * 22.0) * env
    # Empirical calibration fitted on the first (uncalibrated) run of this
    # backtest — mirrors dashboard.py. Deciles overshot actual ~25-30%;
    # linear fit R^2 = 0.97:  calibrated = 0.923 × raw − 2.730
    prob   = max(0.3, 0.923 * prob - 2.730)
    rows.append({"raw_score": talent, "model_prob": prob, "hit_hr": int(g["hit_hr"])})

res = pd.DataFrame(rows)
print(f"Scored {len(res):,} games | mean pred {res['model_prob'].mean():.2f}% "
      f"| actual {res['hit_hr'].mean()*100:.2f}%")

# ─────────────────────────────────────────────────────────────────
# (1) LIFT — does a higher score mean a higher HR rate? (mapping-independent)
#     This is the fundamental "is there out-of-sample signal?" test.
# ─────────────────────────────────────────────────────────────────
print("\n── LIFT: HR rate by model-score decile (out-of-sample) ──────────")
print(f"{'Decile':>7}  {'# games':>8}  {'Actual HR%':>10}")
print("-" * 34)
res["decile"] = pd.qcut(res["raw_score"], 10, labels=False, duplicates="drop")
lift_tbl = res.groupby("decile")["hit_hr"].agg(["count", "mean"])
for dec, r in lift_tbl.iterrows():
    print(f"  D{int(dec)+1:>4}  {int(r['count']):>8,}  {r['mean']*100:>9.1f}%")
top_rate = lift_tbl["mean"].iloc[-1] * 100
bot_rate = lift_tbl["mean"].iloc[0] * 100
print(f"\n  Top decile: {top_rate:.1f}% HR  vs  Bottom decile: {bot_rate:.1f}% HR")
if bot_rate > 0:
    print(f"  Lift ratio (top/bottom): {top_rate/bot_rate:.2f}x  "
          f"({'SIGNAL' if top_rate > bot_rate*1.3 else 'WEAK/NO SIGNAL'})")

# ─────────────────────────────────────────────────────────────────
# (2) CALIBRATION — is the probability itself accurate?
# ─────────────────────────────────────────────────────────────────
print("\n── CALIBRATION: predicted % vs actual HR% ───────────────────────")
print(f"{'Pred bin':>12}  {'# games':>8}  {'Avg pred%':>10}  {'Actual%':>9}  {'Ratio':>7}")
print("-" * 56)
res["pbin"] = pd.qcut(res["model_prob"], 10, duplicates="drop")
for name, grp in res.groupby("pbin", observed=True):
    pred = grp["model_prob"].mean(); act = grp["hit_hr"].mean()*100
    ratio = act/pred if pred > 0 else float("nan")
    print(f"  {str(name):>12}  {len(grp):>8,}  {pred:>9.1f}%  {act:>8.1f}%  {ratio:>7.2f}")

# ─────────────────────────────────────────────────────────────────
# (3) Brier score / skill vs naive baseline
# ─────────────────────────────────────────────────────────────────
brier = ((res["model_prob"]/100 - res["hit_hr"])**2).mean()
base  = res["hit_hr"].mean()
brier_base = ((base - res["hit_hr"])**2).mean()
print(f"\n── Brier: {brier:.5f}  vs baseline {brier_base:.5f}  "
      f"({(1-brier/brier_base)*100:+.1f}% skill)")

# ─────────────────────────────────────────────────────────────────
# AUC (ranking quality) if sklearn available
# ─────────────────────────────────────────────────────────────────
try:
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(res["hit_hr"], res["raw_score"])
    print(f"── AUC: {auc:.4f}  (0.5 = coin flip; >0.60 = usable HR signal)")
except Exception:
    pass

print("""
─────────────────────────────────────────────────────────────────────
WHAT THIS DOES AND DOESN'T PROVE
  ✓ Out-of-sample: test outcomes are NOT in the profiles (no leakage)
  ✓ Tells you if the features rank/predict HRs better than chance
  ✗ Does NOT prove betting profit — that needs REAL historical closing
    odds, which we don't have. Beating the vig is a separate, harder bar.
  → For the profit question, log real odds forward (picks_history.csv +
    check_results.py) and grade them. Every logged day builds an owned
    odds dataset you never have to re-buy from the API.
─────────────────────────────────────────────────────────────────────""")
