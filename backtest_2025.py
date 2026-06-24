"""
backtest_2025.py
----------------
Simulate the HR prop model against 2025 historical data.

Steps:
  1. Load homerun_data_all.csv (raw pitch-by-pitch Statcast)
  2. Build player-game level outcomes (did the batter HR that game?)
  3. Load season-aggregated hitter/pitcher profiles
  4. For each player-game: compute model prediction using the same
     logic as dashboard.py → predict_with_reasons()
  5. Simulate flat $100 bets at various edge thresholds
  6. Print calibration table + ROI summary

NOTE: This is an IN-SAMPLE backtest because hitter/pitcher profiles
were built from the same 2025 data. Expect the real out-of-sample
ROI to be lower. Use this to check calibration and compare thresholds,
not to project profitability.

Usage:
  python backtest_2025.py
  python backtest_2025.py --min-edge 3 --min-pa 50
"""

import argparse
import pickle
import sys
import pandas as pd
import numpy as np

# ─────────────────────────────────────────────────────────────────
# CLI args
# ─────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--min-edge",     type=float, default=0.0,  help="Minimum model_prob - book_implied edge to bet (default 0)")
parser.add_argument("--min-pa",       type=int,   default=30,   help="Minimum batter PAs in profile to include (default 30)")
parser.add_argument("--stake",        type=float, default=100,  help="Flat bet size in dollars (default $100)")
parser.add_argument("--book-hold",    type=float, default=0.10, help="Assumed sportsbook hold on Yes side (default 10%)")
parser.add_argument("--show-top",     type=int,   default=20,   help="Show top N most predicted player-games (default 20)")
args = parser.parse_args()

# ─────────────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────────────
print("Loading Statcast data...")
raw_path = None
for candidate in ["homerun_data_all.csv", "homerun_data_enriched.csv"]:
    try:
        raw = pd.read_csv(candidate, low_memory=False)
        raw_path = candidate
        print(f"  Loaded {raw_path}: {len(raw):,} rows")
        break
    except FileNotFoundError:
        continue

if raw_path is None:
    print("ERROR: homerun_data_all.csv not found. Run pull_data.py first.")
    sys.exit(1)

print("Loading profiles...")
hitter  = pd.read_csv("hitter_profiles.csv")
pitcher = pd.read_csv("pitcher_profiles.csv")
print(f"  {len(hitter):,} hitter profiles | {len(pitcher):,} pitcher profiles")

ML_MODEL   = None
ML_FEATURES = []
try:
    ML_MODEL   = pickle.load(open("hr_model.pkl", "rb"))
    ML_FEATURES = pd.read_csv("model_features.csv")["feature"].tolist()
    print(f"  ML model loaded: {len(ML_FEATURES)} features")
except Exception as e:
    print(f"  ML model not available ({e}) — will use matchup index formula")

# ─────────────────────────────────────────────────────────────────
# Constants (mirror dashboard.py)
# ─────────────────────────────────────────────────────────────────
BALLPARK_HR_FACTORS = {
    "COL":1.38,"CIN":1.22,"NYY":1.18,"PHI":1.16,"TEX":1.14,"CHC":1.12,
    "ARI":1.10,"HOU":1.07,"MIA":1.06,"DET":1.04,"BAL":1.03,"BOS":1.02,
    "MIN":1.01,"STL":1.00,"NYM":1.00,"ATL":0.99,"LAD":0.99,"KC":0.98,
    "MIL":0.97,"TOR":0.97,"CLE":0.96,"LAA":0.95,"CWS":0.94,"WSH":0.94,
    "TB":0.93,"SF":0.91,"SEA":0.90,"PIT":0.89,"OAK":0.88,"SD":0.87,
}
TEAM_BULLPEN_HR_RATE = {
    "NYY":0.024,"PHI":0.026,"LAD":0.027,"MIL":0.027,"ATL":0.028,"HOU":0.028,
    "CLE":0.029,"MIN":0.030,"SEA":0.030,"SD":0.031,"CHC":0.032,"SF":0.032,
    "STL":0.033,"BOS":0.033,"TOR":0.034,"BAL":0.034,"TB":0.034,"LAA":0.035,
    "WSH":0.035,"ARI":0.035,"NYM":0.036,"MIA":0.036,"DET":0.037,"KC":0.037,
    "CIN":0.038,"TEX":0.038,"PIT":0.040,"CWS":0.041,"OAK":0.042,"COL":0.051,
}
LEAGUE_AVG_BULLPEN = 0.034
LEAGUE_AVG_PA      = 4.1
STARTER_PA_FRAC    = 0.67
BULLPEN_PA_FRAC    = 0.33
LEAGUE_AVG_PA_AB   = 0.0082

def _safe_float(v):
    try:
        f = float(v)
        return None if (np.isnan(f) or np.isinf(f)) else f
    except Exception:
        return None

# ─────────────────────────────────────────────────────────────────
# Build player-game level outcomes from raw data
# ─────────────────────────────────────────────────────────────────
print("\nBuilding player-game outcomes...")

# Only keep 2025 regular season (skip 2026 so backtest is on historic data)
if "game_date" in raw.columns:
    raw["game_date"] = pd.to_datetime(raw["game_date"], errors="coerce")
    data_2025 = raw[raw["game_date"].dt.year == 2025].copy()
    print(f"  2025 rows: {len(data_2025):,}")
else:
    data_2025 = raw.copy()

# Flag HRs
data_2025["is_hr"] = (data_2025["events"] == "home_run").fillna(False).astype(int)

# One row per (batter_id, pitcher, game_date, home_team) using first PA of game for metadata
player_games = (
    data_2025.groupby(["batter", "player_name", "game_date"])
    .agg(
        hit_hr     = ("is_hr",       "max"),   # 1 if HR anytime that game
        home_team  = ("home_team",   "first"),
        pitcher_hand_thrown = ("p_throws", "first"),  # first pitcher = starter
        pitcher_name = ("player_name", "first"),      # opponent pitcher name
    )
    .reset_index()
    .rename(columns={"batter": "batter_id"})
)

# Get opposing pitcher name from the pitcher column specifically
if "pitcher" in data_2025.columns:
    # Match pitcher name to the pitcher who started (most pitches early in the game)
    starter_map = (
        data_2025.sort_values("at_bat_number")
        .groupby(["batter", "game_date"])
        .agg(opp_pitcher=("player_name", "first"))  # player_name = pitcher who threw
        .reset_index()
        .rename(columns={"batter": "batter_id"})
    )
    player_games = player_games.merge(starter_map, on=["batter_id", "game_date"], how="left")
else:
    player_games["opp_pitcher"] = None

# Filter to hitters with enough PA
qualified = set(
    hitter.loc[hitter["h_n_batted"] >= args.min_pa, "batter"].values
)
player_games = player_games[player_games["batter_id"].isin(qualified)]
print(f"  Player-games: {len(player_games):,} | HR rate: {player_games['hit_hr'].mean():.3f}")

# ─────────────────────────────────────────────────────────────────
# Compute model predictions
# ─────────────────────────────────────────────────────────────────
print("\nScoring matchups...")

def norm(val, lo, hi):
    return min(100.0, max(0.0, (val - lo) / (hi - lo) * 100.0))

def predict_prob(batter_id, opp_pitcher_name, home_team, pitcher_hand):
    h_row = hitter[hitter["batter"] == batter_id]
    p_row = pitcher[pitcher["player_name"] == opp_pitcher_name] if opp_pitcher_name else pd.DataFrame()

    if len(h_row) == 0:
        return None, None

    hr = h_row.iloc[0]

    # ── ML model path ──────────────────────────────────────────
    if ML_MODEL is not None and len(ML_FEATURES) > 0:
        try:
            feat = {}
            for col in hr.index:
                if col.startswith("h_"):
                    feat[col] = _safe_float(hr.get(col)) or 0.0
            if len(p_row) > 0:
                pr = p_row.iloc[0]
                for col in pr.index:
                    if col.startswith("p_"):
                        feat[col] = _safe_float(pr.get(col)) or 0.0

            batter_right  = int(round(_safe_float(hr.get("h_batter_right")) or 0))
            batter_suffix = "rhh" if batter_right else "lhh"

            # Matchup interactions
            if len(p_row) > 0:
                pr = p_row.iloc[0]
                feat["m_barrel_matchup_score"]   = feat.get("h_barrel_pct", 0) * feat.get("p_barrel_pct_allowed", 0)
                feat["m_hr_contact_matchup"]      = feat.get("h_hr_contact_score", 0) * feat.get("p_hr_contact_risk", 0)
                feat["m_lifted_power_matchup"]    = feat.get("h_lifted_power_score", 0) * feat.get("p_lift_damage_risk", 0)
                feat["m_lift_matchup_score"]      = (
                    feat.get("h_sweet_spot_pct", 0) * feat.get("p_sweet_spot_pct_allowed", 0)
                    + feat.get("h_pull_air_pct", 0) * feat.get("p_pull_air_pct_allowed", 0)
                ) / 2.0
                feat["m_sweet_spot_contact_edge"] = feat.get("h_sweet_spot_pct", 0) * feat.get("p_sweet_spot_pct_allowed", 0)
                feat["m_zone_attack_edge"]        = feat.get("h_zone_contact_pct", 0) * feat.get("p_in_zone_pct", 0)
                # Handed matchup
                hand_label  = "rhp" if pitcher_hand == "R" else "lhp"
                h_hr_hand   = _safe_float(hr.get(f"h_hr_vs_{hand_label}")) or _safe_float(hr.get("h_hr_rate")) or 0
                p_hr_side   = _safe_float(pr.get(f"p_hr_rate_allowed_{batter_suffix}")) or _safe_float(pr.get("p_hr_rate_allowed")) or 0
                feat["m_handed_hr_matchup"]       = h_hr_hand * p_hr_side
                feat["h_hr_rate_vs_hand"]         = h_hr_hand
                feat["p_hr_rate_allowed_vs_side"] = p_hr_side

            feat["batter_right"]  = batter_right
            feat["pitcher_right"] = 1 if pitcher_hand == "R" else 0
            feat["is_coors"]      = 1 if home_team == "COL" else 0
            ALL_TEAMS = sorted(["ATL","ARI","BAL","BOS","CHC","CWS","CIN","CLE","COL","DET",
                                 "HOU","KC","LAA","LAD","MIA","MIL","MIN","NYM","NYY","OAK",
                                 "PHI","PIT","SD","SF","SEA","STL","TB","TEX","TOR","WSH"])
            feat["ballpark_code"] = ALL_TEAMS.index(home_team) if home_team in ALL_TEAMS else 0
            feat["temp_f"]    = 72.0
            feat["humidity"]  = 50.0
            feat["wind_speed"]= 0.0
            feat["wind_dir"]  = 0.5
            # Fill missing matchup features
            for mf in ["m_handed_contact_matchup","m_handed_lift_matchup",
                       "m_pitch_hr_matchup","m_pitch_contact_matchup","m_pitch_ev_matchup"]:
                feat.setdefault(mf, 0.0)
            for mf in ["h_xwoba_vs_hand","h_barrel_pct_vs_hand","h_hard_hit_pct_vs_hand",
                       "h_launch_angle_vs_hand","h_sweet_spot_pct_vs_hand","h_pull_air_pct_vs_hand",
                       "h_hr_contact_score_vs_hand","h_lifted_power_score_vs_hand",
                       "p_barrel_pct_allowed_vs_side","p_hard_hit_pct_allowed_vs_side",
                       "p_exit_velo_allowed_vs_side","p_launch_angle_allowed_vs_side",
                       "p_sweet_spot_pct_allowed_vs_side","p_pull_air_pct_allowed_vs_side",
                       "p_hr_contact_risk_vs_side","p_lift_damage_risk_vs_side"]:
                feat.setdefault(mf, 0.0)

            X = pd.DataFrame([feat])
            for f in ML_FEATURES:
                if f not in X.columns:
                    X[f] = 0.0
            X = X[ML_FEATURES].fillna(0)
            prob_ab = float(ML_MODEL.predict_proba(X)[0, 1])

            # Per-game via starter+bullpen PA blend
            bullpen_rate = TEAM_BULLPEN_HR_RATE.get(home_team, LEAGUE_AVG_BULLPEN)
            quality_mult = prob_ab / LEAGUE_AVG_PA_AB if LEAGUE_AVG_PA_AB > 0 else 1.0
            bp_ab        = bullpen_rate * quality_mult
            s_pa         = LEAGUE_AVG_PA * STARTER_PA_FRAC
            b_pa         = LEAGUE_AVG_PA * BULLPEN_PA_FRAC
            prob_game    = 1 - (1 - prob_ab) ** s_pa * (1 - bp_ab) ** b_pa
            # Environment multiplier — park-based fallback (historical game totals
            # aren't in Statcast data; live dashboard uses the market total instead)
            park_f = BALLPARK_HR_FACTORS.get(home_team, 1.0)
            env    = max(0.72, min(1.35, park_f))
            return round(prob_game * 100 * env, 2), None
        except Exception:
            pass

    # ── Matchup index fallback ──────────────────────────────────
    barrel   = _safe_float(hr.get("h_barrel_pct"))   or 0.0
    hard_hit = _safe_float(hr.get("h_hard_hit_pct")) or 0.0
    ev       = _safe_float(hr.get("h_exit_velo"))    or 82.0
    hr_rate  = _safe_float(hr.get("h_hr_rate"))      or 0.019
    batter_score = (
        norm(barrel,   0.00, 0.08) * 0.35
      + norm(hard_hit, 0.13, 0.34) * 0.25
      + norm(ev,       77.0, 87.0) * 0.20
      + norm(hr_rate,  0.00, 0.04) * 0.20
    )
    if len(p_row) > 0:
        pr = p_row.iloc[0]
        p_hr  = _safe_float(pr.get("p_hr_rate_allowed"))    or 0.027
        p_bb  = _safe_float(pr.get("p_barrel_pct_allowed")) or 0.049
        p_hh  = _safe_float(pr.get("p_hard_hit_pct_allowed")) or 0.262
        pitcher_score = (
            norm(p_hr, 0.000, 0.055) * 0.50
          + norm(p_bb, 0.000, 0.100) * 0.30
          + norm(p_hh, 0.140, 0.380) * 0.20
        )
    else:
        pitcher_score = 50.0

    park_f   = BALLPARK_HR_FACTORS.get(home_team, 1.0)
    ctx      = norm(park_f, 0.86, 1.40) * 100
    matchup  = batter_score * 0.40 + pitcher_score * 0.40 + ctx * 0.20
    park_mult = park_f
    prob_game = (3.0 + (matchup / 100.0) * 22.0) * park_mult
    return round(prob_game, 2), round(matchup, 1)


preds = []
n = len(player_games)
for i, (_, row) in enumerate(player_games.iterrows()):
    if i % 2000 == 0:
        print(f"  {i:,}/{n:,}...", end="\r")
    prob, score = predict_prob(
        int(row["batter_id"]),
        str(row.get("opp_pitcher") or ""),
        str(row.get("home_team") or ""),
        str(row.get("pitcher_hand_thrown") or "R"),
    )
    preds.append({
        "batter_id":  row["batter_id"],
        "game_date":  row["game_date"],
        "hit_hr":     int(row["hit_hr"]),
        "home_team":  row.get("home_team",""),
        "model_prob": prob,
        "hit_hr":     int(row["hit_hr"]),
    })
print(f"  {n:,}/{n:,} done.    ")

results = pd.DataFrame(preds).dropna(subset=["model_prob"])
print(f"\nScored {len(results):,} player-games | HR rate: {results['hit_hr'].mean():.3f}")
print(f"Model prob — mean: {results['model_prob'].mean():.2f}%  median: {results['model_prob'].median():.2f}%  max: {results['model_prob'].max():.2f}%")

# ─────────────────────────────────────────────────────────────────
# Calibration table — does 10% prob -> 10% actual HR rate?
# ─────────────────────────────────────────────────────────────────
print("\n── Calibration (are predicted probs accurate?) ──────────────────")
print(f"{'Decile':>10}  {'Pred range':>14}  {'# games':>8}  {'Actual HR%':>10}  {'Avg pred%':>10}  {'Ratio':>7}")
print("-" * 70)
results["prob_decile"] = pd.qcut(results["model_prob"], q=10, duplicates="drop")
for name, grp in results.groupby("prob_decile", observed=True):
    actual  = grp["hit_hr"].mean() * 100
    pred    = grp["model_prob"].mean()
    ratio   = actual / pred if pred > 0 else float("nan")
    print(f"  {str(name):>14}  {len(grp):>8}  {actual:>9.1f}%  {pred:>9.1f}%  {ratio:>7.2f}")

# ─────────────────────────────────────────────────────────────────
# Simulated betting at typical sportsbook odds
# Assumption: book implied = model_prob × (1 + hold)
# We bet when model_prob > implied (i.e., always if hold = 0)
# Real results depend on actual closing lines.
# ─────────────────────────────────────────────────────────────────
print("\n── Simulated flat bet ROI at various edge thresholds ────────────")
print("(Assumes sportsbook hold={:.0f}% applied to model prob for implied odds)".format(args.book_hold * 100))
print(f"{'Edge≥':>8}  {'# bets':>8}  {'HR hits':>8}  {'Hit%':>7}  {'Avg odds':>9}  {'Net P&L':>10}  {'ROI':>8}")
print("-" * 70)

results["book_implied"] = results["model_prob"] * (1 + args.book_hold)
# Convert implied back to approximate American odds (positive, since HR props >100)
results["approx_odds"] = np.where(
    results["book_implied"] < 50,
    (100 / (results["book_implied"] / 100) - 100).clip(lower=100),
    -(results["book_implied"] / (100 - results["book_implied"]) * 100).clip(lower=100),
)
results["edge"] = results["model_prob"] - results["book_implied"]

for min_edge in [-999, 0, 1, 2, 3, 5, 8]:
    subset = results[results["edge"] >= min_edge]
    if len(subset) == 0:
        continue
    hits    = subset["hit_hr"].sum()
    n_bets  = len(subset)
    avg_odds = subset["approx_odds"].mean()
    # PnL per bet: if HR → +odds/100 × stake; if no HR → -stake
    pnl = subset.apply(
        lambda r: (r["approx_odds"] / 100 * args.stake) if r["hit_hr"] else -args.stake,
        axis=1
    ).sum()
    roi = pnl / (n_bets * args.stake) * 100
    label = f"≥{min_edge:+.0f}%" if min_edge > -999 else "all"
    print(f"  {label:>7}  {n_bets:>8,}  {hits:>8,}  {hits/n_bets*100:>6.1f}%  {avg_odds:>+9.0f}  ${pnl:>+9.0f}  {roi:>+7.1f}%")

# ─────────────────────────────────────────────────────────────────
# Top predicted player-games
# ─────────────────────────────────────────────────────────────────
if args.show_top > 0:
    top = results.nlargest(args.show_top, "model_prob")
    print(f"\n── Top {args.show_top} highest-predicted player-games ──────────────")
    print(f"{'batter_id':>10}  {'date':>12}  {'home':>6}  {'pred%':>7}  {'HR?':>5}")
    for _, r in top.iterrows():
        hr_flag = "✓ HR" if r["hit_hr"] else "✗"
        print(f"  {int(r['batter_id']):>10}  {str(r['game_date'])[:10]:>12}  {r['home_team']:>6}  {r['model_prob']:>6.1f}%  {hr_flag}")

# ─────────────────────────────────────────────────────────────────
# Brier score — probability accuracy metric
# ─────────────────────────────────────────────────────────────────
brier = ((results["model_prob"] / 100 - results["hit_hr"]) ** 2).mean()
baseline_brier = ((results["hit_hr"].mean() - results["hit_hr"]) ** 2).mean()
print(f"\n── Brier Score: {brier:.5f}  (baseline: {baseline_brier:.5f} | lower = better)")
print(f"   Skill vs baseline: {(1 - brier/baseline_brier)*100:.1f}% reduction in error")

print("\nDone. Remember: this is IN-SAMPLE. Real edge requires out-of-sample testing.")
