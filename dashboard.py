import math
import pickle
import unicodedata
import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
try:
    from scipy.special import expit as scipy_expit
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    def scipy_expit(x): return 1.0 / (1.0 + math.exp(-max(-500, min(500, x))))
 
ODDS_API_KEY = "3f2e2d867484541b580084248cdb1d1c"
 
print("Loading profiles...")
hitter  = pd.read_csv("hitter_profiles.csv")
pitcher = pd.read_csv("pitcher_profiles.csv")
print(f"  Hitter profiles:  {len(hitter):,}")
print(f"  Pitcher profiles: {len(pitcher):,}")
 
ET = ZoneInfo("America/New_York")
 
# ── Population statistics for z-score calculation ─────────────
# For every numeric feature, store (mean, std) across ALL players.
# A z-score tells us how many standard deviations above/below average
# a player is on that stat.  Outliers (z > 2.5) are genuinely elite.
 
def _pop_stats(df, prefix):
    stats = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        if col.startswith(prefix):
            vals = df[col].dropna()
            if len(vals) >= 10 and vals.std() > 0:
                stats[col] = (float(vals.mean()), float(vals.std()))
    return stats
 
hitter_pop  = _pop_stats(hitter,  "h_")
pitcher_pop = _pop_stats(pitcher, "p_")
print(f"  Hitter features with stats:  {len(hitter_pop)}")
print(f"  Pitcher features with stats: {len(pitcher_pop)}")
 
# ── ML model loading ───────────────────────────────────────────
# Load the trained pipeline (GradientBoosting + StandardScaler).
# Falls back gracefully to z-score formula if model not found.
pipeline    = None
lr_model    = None
lr_scaler   = None
lr_features = None
try:
    with open("hr_model.pkl", "rb") as _f:
        pipeline = pickle.load(_f)
    _feat_df    = pd.read_csv("model_features.csv")
    lr_features = _feat_df["feature"].tolist()
    lr_scaler   = pipeline.named_steps["scaler"]
    lr_model    = pipeline.named_steps["model"]
    # Check how many features overlap with available profile columns
    h_overlap = [f for f in lr_features if f in hitter.columns]
    p_overlap = [f for f in lr_features if f in pitcher.columns]
    print(f"  ML model loaded — {len(lr_features)} features "
          f"({len([f for f in lr_features if f.startswith('h_')])} hitter, "
          f"{len([f for f in lr_features if f.startswith('p_')])} pitcher)")
    print(f"  Feature overlap: {len(h_overlap)} hitter cols, {len(p_overlap)} pitcher cols found in profiles")
except FileNotFoundError:
    print("  WARNING: hr_model.pkl not found — using z-score fallback. Run train_model.py first.")
except Exception as _e:
    print(f"  WARNING: Model load failed ({_e}) — using z-score fallback.")
 
# ── Ballpark GPS coordinates ───────────────────────────────────
ballpark_coords = {
    "NYY": (40.8296,-73.9262), "NYM": (40.7571,-73.8458),
    "BOS": (42.3467,-71.0972), "TB":  (27.7683,-82.6534),
    "TOR": (43.6414,-79.3894), "BAL": (39.2838,-76.6216),
    "CLE": (41.4962,-81.6852), "CWS": (41.8299,-87.6338),
    "DET": (42.3390,-83.0485), "KC":  (39.0517,-94.4803),
    "MIN": (44.9817,-93.2777), "HOU": (29.7573,-95.3555),
    "LAA": (33.8003,-117.8827),"OAK": (37.7516,-122.2005),
    "SEA": (47.5914,-122.3325),"TEX": (32.7473,-97.0845),
    "ATL": (33.8908,-84.4678), "MIA": (25.7781,-80.2197),
    "PHI": (39.9061,-75.1665), "WSH": (38.8730,-77.0074),
    "CHC": (41.9484,-87.6553), "CIN": (39.0979,-84.5082),
    "MIL": (43.0280,-87.9712), "PIT": (40.4469,-80.0057),
    "STL": (38.6226,-90.1928), "ARI": (33.4455,-112.0667),
    "COL": (39.7559,-104.9942),"LAD": (34.0739,-118.2400),
    "SD":  (32.7076,-117.1570),"SF":  (37.7786,-122.3893),
}
 
BOOK_NAMES = {
    "draftkings":     "DraftKings",
    "betmgm":         "BetMGM",
    "betrivers":      "BetRivers",
    "betonlineag":    "BetOnline",
    "williamhill_us": "William Hill",
    "caesars":        "Caesars",
    "fanduel":        "FanDuel",
    "pointsbet":      "PointsBet",
    "betus":          "BetUS",
    "mybookieag":     "MyBookie",
    "bovada":         "Bovada",
    "lowvig":         "LowVig",
    "consensus":      "Consensus",
}
 
def implied_to_american(implied_pct):
    """Convert implied probability % back to American odds (always positive for HR props)."""
    p = max(0.01, min(0.99, implied_pct / 100.0))
    if p >= 0.5:
        return -round(p / (1 - p) * 100)
    return round((1 - p) / p * 100)
 
BOOK_PRIORITY = ["draftkings","betmgm","betrivers","betonlineag",
                 "williamhill_us","caesars","pointsbet","betus","mybookieag"]
 
def book_rank(k):
    try:    return BOOK_PRIORITY.index(k)
    except: return 99
 
# Top-tier books whose odds we trust for display / edge calculation
TOP_BOOKS = {"draftkings", "betmgm", "caesars", "williamhill_us", "fanduel"}
 
# ── Team bullpen HR rate per at-bat (2025 estimates) ───────────
# League average is ~0.034.  Update these periodically as the season moves.
# Lower = tougher bullpen for hitters.  Higher = HR-friendly bullpen.
TEAM_BULLPEN_HR_RATE = {
    "NYY": 0.024,  # elite shutdown pen
    "PHI": 0.026,  # strong backend
    "LAD": 0.027,  # deep, reliable
    "MIL": 0.027,  # Devin Williams era holdovers
    "ATL": 0.028,
    "HOU": 0.028,
    "CLE": 0.029,
    "MIN": 0.030,
    "SEA": 0.030,
    "SD":  0.031,
    "CHC": 0.032,
    "SF":  0.032,
    "STL": 0.033,
    "BOS": 0.033,
    "TOR": 0.034,  # league average
    "BAL": 0.034,
    "TB":  0.034,
    "LAA": 0.035,
    "WSH": 0.035,
    "ARI": 0.035,
    "NYM": 0.036,
    "MIA": 0.036,
    "DET": 0.037,
    "KC":  0.037,
    "CIN": 0.038,
    "TEX": 0.038,
    "PIT": 0.040,
    "CWS": 0.041,
    "OAK": 0.042,
    "COL": 0.051,  # Coors + thin air hits everyone
}
LEAGUE_AVG_BULLPEN = 0.034
 
def _bullpen_multiplier(team):
    """Convert team bullpen HR rate → adjustment multiplier (same scale as starter tiers)."""
    rate = TEAM_BULLPEN_HR_RATE.get(team, LEAGUE_AVG_BULLPEN)
    if   rate < 0.022: return 0.65
    elif rate < 0.028: return 0.82
    elif rate < 0.032: return 0.92
    elif rate > 0.045: return 1.28
    elif rate > 0.038: return 1.15
    elif rate > 0.034: return 1.08
    else:              return 1.00
 
def _ascii(s):
    """Strip accents and lower-case a name string.
    'Eugenio Suárez' → 'eugenio suarez'
    """
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii").lower().strip()
 
def _name_key(name):
    """Canonical match key: ascii-normalised, suffixes stripped."""
    n = _ascii(name)
    for sfx in (" jr", " sr", " ii", " iii", " iv"):
        if n.endswith(sfx):
            n = n[: -len(sfx)].strip()
    return n
 
def american_to_implied(o):
    if o > 0: return 100 / (o + 100)
    return abs(o) / (abs(o) + 100)
 
# ── Weather ────────────────────────────────────────────────────
weather_cache = {}
def get_forecast(team):
    if team in weather_cache:
        return weather_cache[team]
    coords = ballpark_coords.get(team)
    if not coords:
        return {}
    lat, lon = coords
    try:
        d = requests.get(
            f"https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}"
            f"&hourly=temperature_2m,relativehumidity_2m,windspeed_10m,winddirection_10m"
            f"&temperature_unit=fahrenheit&windspeed_unit=mph"
            f"&timezone=America/New_York&forecast_days=1",
            timeout=10
        ).json()["hourly"]
        result = {
            "temp_f":    d["temperature_2m"][19],
            "humidity":  d["relativehumidity_2m"][19],
            "wind_speed":d["windspeed_10m"][19],
            "wind_dir":  d["winddirection_10m"][19],
        }
        weather_cache[team] = result
        return result
    except:
        return {}
 
# ── Name helpers ───────────────────────────────────────────────
def to_statcast_name(full_name):
    parts = full_name.strip().split()
    if len(parts) >= 2:
        return f"{parts[-1]}, {' '.join(parts[:-1])}"
    return full_name
 
def norm_cdf(z):
    """Approximate normal CDF without scipy."""
    return (1.0 + math.erf(z / math.sqrt(2.0))) / 2.0
 
# ── Reason text builder ────────────────────────────────────────
def _reason_text(feat, val, z, pitcher_hand):
    """
    Returns (text, is_super_outlier, is_cold_outlier).
    Super outlier = z >= 2.5  → ⚡ gold  (elite positive stat)
    Cold outlier  = z <= -1.5 → ❄️ blue  (notably bad stat / tough pitcher)
    """
    super_out = z >= 2.5
    cold_out  = z <= -1.5
    top_pct   = (1.0 - norm_cdf(z)) * 100.0
    bot_pct   = norm_cdf(z) * 100.0
 
    if super_out:
        rank_str  = f"top {top_pct:.1f}% in MLB" if top_pct < 0.5 else f"top {top_pct:.0f}% in MLB"
        sigma_str = f"elite — {rank_str}"
        prefix    = "⚡ "
    elif z >= 2.0:
        sigma_str = f"outstanding — top {top_pct:.0f}% in MLB"
        prefix    = ""
    elif z >= 1.5:
        sigma_str = "well above average"
        prefix    = ""
    elif z >= 1.0:
        sigma_str = "above average"
        prefix    = ""
    elif z >= 0.5:
        sigma_str = "slightly above average"
        prefix    = ""
    elif z >= 0:
        sigma_str = "near average"
        prefix    = ""
    elif cold_out:
        sigma_str = f"well below average — bottom {bot_pct:.0f}% in MLB"
        prefix    = "❄️ "
    else:
        sigma_str = "below average"
        prefix    = ""
 
    ph = "LHP" if pitcher_hand == "L" else "RHP"
 
    templates = {
        "h_barrel_pct":
            f"{prefix}Barrel rate {val*100:.1f}% ({sigma_str})",
        "h_exit_velo":
            f"{prefix}Avg exit velocity {val:.1f} mph ({sigma_str})",
        "h_hard_hit_pct":
            f"{prefix}Hard contact {val*100:.1f}% of batted balls ({sigma_str})",
        "h_pull_air_pct":
            f"{prefix}Pull-side fly ball rate {val*100:.1f}% ({sigma_str})",
        "h_hr_rate":
            (f"{prefix}Hits a HR every {int(1/val)} at-bats ({sigma_str})" if val > 0 else None),
        "h_launch_angle":
            f"{prefix}Avg launch angle {val:.1f}° ({sigma_str})",
        "h_hr_vs_rhp":
            f"{prefix}vs RHP: HR every {int(1/val) if val > 0 else '—'} at-bats ({sigma_str})",
        "h_hr_vs_lhp":
            f"{prefix}vs LHP: HR every {int(1/val) if val > 0 else '—'} at-bats ({sigma_str})",
        "h_hr_vs_4seam":
            f"{prefix}HR rate vs 4-seam fastball {val*100:.2f}% ({sigma_str})",
        "h_hr_vs_slider":
            f"{prefix}HR rate vs slider {val*100:.2f}% ({sigma_str})",
        "h_hr_vs_change":
            f"{prefix}HR rate vs changeup {val*100:.2f}% ({sigma_str})",
        "h_hr_vs_sinker":
            f"{prefix}HR rate vs sinker {val*100:.2f}% ({sigma_str})",
        "h_ev_vs_4seam":
            f"{prefix}Exit velocity vs 4-seam: {val:.1f} mph ({sigma_str})",
        "h_ev_vs_slider":
            f"{prefix}Exit velocity vs slider: {val:.1f} mph ({sigma_str})",
        "h_xwoba_vs_4seam":
            f"{prefix}xwOBA vs fastball {val:.3f} ({sigma_str})",
        "h_xwoba_vs_slider":
            f"{prefix}xwOBA vs slider {val:.3f} ({sigma_str})",
        "p_hr_rate_allowed":
            (f"{prefix}Pitcher allows a HR every {int(1/val)} ABs ({sigma_str})" if val > 0 else None),
        "p_barrel_pct_allowed":
            f"{prefix}Pitcher gives up barrel contact {val*100:.1f}% of ABs ({sigma_str})",
        "p_hard_hit_pct_allowed":
            f"{prefix}Pitcher allows hard contact {val*100:.1f}% ({sigma_str})",
        "p_exit_velo_allowed":
            f"{prefix}Pitcher allows {val:.1f} mph avg exit velocity ({sigma_str})",
        "p_pull_air_pct_allowed":
            f"{prefix}Pitcher gives up pull fly balls {val*100:.1f}% ({sigma_str})",
        "p_spin_into_barrel_pct":
            f"{prefix}Pitcher's spin hits barrel zone {val*100:.1f}% of pitches ({sigma_str})",
    }
    text = templates.get(feat)
    if text is None:
        return None, False, False
    return text, super_out, cold_out
 
def _weather_reason(wx, home_team, wx_adj):
    parts = []
    if home_team == "COL":
        parts.append("Coors Field — highest HR park in baseball")
    temp = wx.get("temp_f", 72)
    wind = wx.get("wind_speed", 0)
    if temp >= 90:
        parts.append(f"{temp:.0f}°F — hot air carries the ball farther")
    elif temp >= 80:
        parts.append(f"{temp:.0f}°F — warm conditions favour fly balls")
    elif temp <= 40:
        parts.append(f"{temp:.0f}°F — cold air suppresses distance")
    elif temp <= 55:
        parts.append(f"{temp:.0f}°F — cool air slightly suppresses carry")
    if wind >= 15:
        parts.append(f"{wind:.0f} mph wind — can boost fly ball distance")
    elif wind >= 10:
        parts.append(f"{wind:.0f} mph wind")
    if not parts:
        return None
    # Hot / Coors = ⚡   Cold suppressing = ❄️   Mild boost = no icon
    if wx_adj >= 0.6:
        icon = "⚡ "
    elif wx_adj <= -0.3:
        icon = "❄️ "
    else:
        icon = ""
    return f"{icon}Weather: {' · '.join(parts)}"
 
# ── Core prediction ────────────────────────────────────────────
# Hitter features where HIGHER = more HR-friendly
HITTER_GOOD_FEATS = [
    "h_barrel_pct", "h_exit_velo", "h_hard_hit_pct", "h_pull_air_pct",
    "h_hr_rate", "h_launch_angle",
    "h_hr_vs_4seam", "h_hr_vs_slider", "h_hr_vs_change", "h_hr_vs_sinker",
    "h_ev_vs_4seam", "h_ev_vs_slider",
    "h_xwoba_vs_4seam", "h_xwoba_vs_slider",
]
# Pitcher features where HIGHER = pitcher is HR-prone (good for hitter)
PITCHER_VULN_FEATS = [
    "p_hr_rate_allowed", "p_barrel_pct_allowed", "p_hard_hit_pct_allowed",
    "p_exit_velo_allowed", "p_pull_air_pct_allowed", "p_spin_into_barrel_pct",
]
 
# Ballpark HR rate factors (for ML feature building)
BALLPARK_HR_FACTORS = {
    "NYY": 1.25, "COL": 1.35, "TEX": 1.15, "HOU": 1.10,
    "ARI": 1.08, "OAK": 0.92, "PIT": 0.88, "SD":  0.85,
}
 
def _wind_dir_encode(direction):
    """Cardinal direction string → 0-1 float."""
    wind_map = {
        "N":0,"NNE":22.5,"NE":45,"ENE":67.5,"E":90,"ESE":112.5,
        "SE":135,"SSE":157.5,"S":180,"SSW":202.5,"SW":225,"WSW":247.5,
        "W":270,"WNW":292.5,"NW":315,"NNW":337.5,
    }
    if not direction:
        return 0.5
    return wind_map.get(str(direction).upper(), 180) / 360.0
 
def predict_with_reasons(batter_id, pitcher_name, home_team, pitcher_hand="R", opp_team=None):
    h_row = hitter[hitter["batter"] == batter_id]
    p_row = pd.DataFrame()
    if pitcher_name and pitcher_name != "TBD":
        p_row = pitcher[pitcher["player_name"] == to_statcast_name(pitcher_name)]
        if len(p_row) == 0:
            last = pitcher_name.split()[-1].lower()
            p_row = pitcher[pitcher["player_name"].str.lower().str.contains(last, na=False)].head(1)
 
    pitcher_found = len(p_row) > 0
 
    # ── Build z-scores (always — used for reasons regardless of model) ──
    zscores = {}  # feat -> (z, raw_val, group)
    h_feats = HITTER_GOOD_FEATS + (
        ["h_hr_vs_lhp"] if pitcher_hand == "L" else ["h_hr_vs_rhp"]
    )
    if len(h_row) > 0:
        hr = h_row.iloc[0]
        for feat in h_feats:
            if feat not in hitter.columns or feat not in hitter_pop:
                continue
            val = hr.get(feat)
            if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
                continue
            mean, std = hitter_pop[feat]
            zscores[feat] = ((float(val) - mean) / std, float(val), "hitter")
 
    if pitcher_found:
        pr = p_row.iloc[0]
        for feat in PITCHER_VULN_FEATS:
            if feat not in pitcher.columns or feat not in pitcher_pop:
                continue
            val = pr.get(feat)
            if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
                continue
            mean, std = pitcher_pop[feat]
            zscores[feat] = ((float(val) - mean) / std, float(val), "pitcher")
 
    # ── Weather ────────────────────────────────────────────────
    wx     = get_forecast(home_team)
    wx_adj = 0.0
    temp   = wx.get("temp_f", 72)
    wind   = wx.get("wind_speed", 0)
    if home_team == "COL": wx_adj += 0.8
    if   temp >= 90: wx_adj += 0.35
    elif temp >= 80: wx_adj += 0.18
    elif temp <= 40: wx_adj -= 0.35
    elif temp <= 55: wx_adj -= 0.18
    if   wind >= 15: wx_adj += 0.18
    elif wind >= 10: wx_adj += 0.10
 
    # ── Probability: ML model (preferred) or z-score fallback ─
    if lr_model is not None and lr_features is not None and len(h_row) > 0:
        # --- Build feature vector for ML model ---
        feature_vals = {}
        for feat in lr_features:
            feature_vals[feat] = np.nan
 
        if len(h_row) > 0:
            hr = h_row.iloc[0]
            for feat in lr_features:
                if feat in hitter.columns and feat.startswith("h_"):
                    v = hr.get(feat)
                    if v is not None and not (isinstance(v, float) and np.isnan(v)):
                        feature_vals[feat] = float(v)
            # Platoon-matched HR rate
            if "platoon_matched_hr_rate" in lr_features:
                feature_vals["platoon_matched_hr_rate"] = float(
                    hr.get("h_hr_vs_rhp" if pitcher_hand == "R" else "h_hr_vs_lhp", np.nan)
                )
 
        if pitcher_found:
            pr = p_row.iloc[0]
            for feat in lr_features:
                if feat in pitcher.columns and feat.startswith("p_"):
                    v = pr.get(feat)
                    if v is not None and not (isinstance(v, float) and np.isnan(v)):
                        feature_vals[feat] = float(v)
 
        # Context features
        feature_vals["batter_right"]      = 1 if pitcher_hand == "R" else 0
        feature_vals["pitcher_right"]     = 1 if pitcher_hand == "R" else 0
        feature_vals["is_coors"]          = 1 if home_team == "COL" else 0
        feature_vals["ballpark_hr_factor"]= BALLPARK_HR_FACTORS.get(home_team, 1.0)
        feature_vals["temp_f"]            = wx.get("temp_f", 72)
        feature_vals["humidity"]          = wx.get("humidity", 50)
        feature_vals["wind_speed"]        = wx.get("wind_speed", 0)
        feature_vals["wind_dir_encoded"]  = _wind_dir_encode(wx.get("wind_dir"))
        feature_vals["ballpark_code"]     = 0   # not used — placeholder
        feature_vals["wind_dir"]          = wx.get("wind_dir", 0)
        feature_vals["temp_f"]            = wx.get("temp_f", 72)
        feature_vals["humidity"]          = wx.get("humidity", 50)
 
        # Build array in feature order, fill NaN with 0
        X_row = np.array([feature_vals.get(f, 0) if not (isinstance(feature_vals.get(f, 0), float) and np.isnan(feature_vals.get(f, 0))) else 0
                          for f in lr_features], dtype=float).reshape(1, -1)
 
        try:
            prob_raw = pipeline.predict_proba(X_row)[0, 1]  # use full pipeline
        except Exception:
            X_scaled = lr_scaler.transform(X_row)
            prob_raw = lr_model.predict_proba(X_scaled)[0, 1]
 
        # ── Sample-size shrinkage (regress small samples toward league avg) ──
        # League avg per-AB HR rate ≈ 3.4%
        # Default to 50 ABs when column is missing — conservative.
        # Need 200+ ABs for 95% trust. Under 100 ABs = over half pulled to avg.
        LEAGUE_AB_RATE = 0.034
        hr = h_row.iloc[0]
        n_batted = float(hr["h_n_batted"]) if "h_n_batted" in hitter.columns and not pd.isna(hr.get("h_n_batted")) else 50
        weight   = min(0.95, max(0.05, (n_batted - 5) / 200.0))
        prob_raw = weight * prob_raw + (1 - weight) * LEAGUE_AB_RATE
 
        # ── Z-score signal multiplier ──────────────────────────
        # Clamp combined_z to ±2.0 so one freak small-sample stat
        # (e.g. 0.830 xwOBA on 3 fastballs seen) can't explode the output.
        h_zs_ml = [z for f, (z, v, t) in zscores.items() if t == "hitter"]
        p_zs_ml = [z for f, (z, v, t) in zscores.items() if t == "pitcher"]
        h_top_ml   = sorted(h_zs_ml, reverse=True)[:5]
        h_score_ml = float(np.mean(h_top_ml)) if h_top_ml else 0.0
        p_score_ml = float(np.mean(p_zs_ml))  if p_zs_ml  else 0.0
        combined_z = h_score_ml * 0.70 + p_score_ml * 0.30 + wx_adj
        combined_z = max(-2.0, min(2.0, combined_z))   # clamp: max ×1.82 boost
        prob_raw   = prob_raw * math.exp(combined_z * 0.30)
 
        # ── Pitcher + bullpen blended adjustment ───────────────
        # Starters pitch ~60% of the game, bullpen covers ~40%.
        # Blend: 60% starter quality + 40% team bullpen quality.
        if pitcher_found:
            p_hr_allowed = float(p_row.iloc[0].get("p_hr_rate_allowed", LEAGUE_AB_RATE))
            if   p_hr_allowed < 0.015: starter_mult = 0.60
            elif p_hr_allowed < 0.022: starter_mult = 0.80
            elif p_hr_allowed > 0.045: starter_mult = 1.30
            elif p_hr_allowed > 0.035: starter_mult = 1.15
            else:                      starter_mult = 1.00
        else:
            starter_mult = 0.92  # unknown starter: slight penalty
 
        bullpen_mult = _bullpen_multiplier(opp_team or home_team)
        blended_mult = 0.60 * starter_mult + 0.40 * bullpen_mult
        prob_raw    *= blended_mult
 
        # ── Hard cap by sample size (applied AFTER all multipliers) ────
        # Prevents small-sample flukes from surviving the z-score boost.
        if   n_batted < 100: prob_raw = min(prob_raw, 0.038)  # →  ~12% per-game max
        elif n_batted < 150: prob_raw = min(prob_raw, 0.052)  # →  ~17% per-game max
 
        # ── Convert per-AB → per-game ──────────────────────────
        # 3.5 PA/game average.  per_game = 1 − (1 − p_ab)^3.5
        #   p_ab=0.020 →  6.7%  p_ab=0.034 → 11%
        #   p_ab=0.060 → 19%    p_ab=0.090 → 28%  p_ab=0.120 → 35%→cap
        n_pa = 3.5
        prob_raw_clamped = max(0.001, min(0.30, prob_raw))
        prob = (1.0 - (1.0 - prob_raw_clamped) ** n_pa) * 100
        prob = max(2.0, min(30.0, prob))
 
    else:
        # --- Z-score fallback (no trained model) ---
        # Base rate = 11% per game (league avg ~3.4% per-AB × 3.5 PA)
        # Slope = 0.80 gives:  z=0→11%  z=1→19%  z=2→29%  z=3→40%→cap30%
        h_zs = [z for f, (z, v, t) in zscores.items() if t == "hitter"]
        p_zs = [z for f, (z, v, t) in zscores.items() if t == "pitcher"]
        h_top    = sorted(h_zs, reverse=True)[:5]
        h_score  = float(np.mean(h_top)) if h_top else 0.0
        p_score  = float(np.mean(p_zs))  if p_zs  else 0.0
        total_z  = h_score * 0.70 + p_score * 0.30 + wx_adj
        # BASE_RATE 0.11 → ~10% per-game avg  SLOPE 0.45 → star(z=1.5)≈16%
        BASE_RATE = 0.11
        SLOPE     = 0.45
        log_odds  = math.log(BASE_RATE / (1.0 - BASE_RATE)) + total_z * SLOPE
        prob      = min(25.0, 100.0 / (1.0 + math.exp(-log_odds)))
 
    # ── Pick reasons: top positive z-scores + worst negatives ──
    # Positive: good stats / HR-prone pitcher → show top 2-3
    pos_scored = sorted(
        [(f, z, v) for f, (z, v, _) in zscores.items() if z > 0.0],
        key=lambda x: x[1], reverse=True,
    )
    # Negative: bad hitter stats or tough pitcher (z ≤ -1.5) → show worst 1
    neg_scored = sorted(
        [(f, z, v) for f, (z, v, _) in zscores.items() if z <= -1.5],
        key=lambda x: x[1],
    )
 
    reasons = []
    for feat, z, val in pos_scored[:5]:
        text, _, _ = _reason_text(feat, val, z, pitcher_hand)
        if text:
            reasons.append(text)
        if len(reasons) == 2:
            break
 
    # Add weather note if meaningful
    if abs(wx_adj) >= 0.3:
        wx_txt = _weather_reason(wx, home_team, wx_adj)
        if wx_txt:
            reasons.append(wx_txt)
    elif len(reasons) < 3:
        # Fill up to 3 with more positive reasons
        for feat, z, val in pos_scored[2:5]:
            text, _, _ = _reason_text(feat, val, z, pitcher_hand)
            if text and text not in reasons:
                reasons.append(text)
            if len(reasons) == 3:
                break
 
    # Append worst cold stat as a warning (max 1)
    for feat, z, val in neg_scored[:1]:
        text, _, cold = _reason_text(feat, val, z, pitcher_hand)
        if text:
            reasons.append(text)
 
    # ── Bullpen note (opponent's pen, not the batter's own team) ─
    opp = opp_team or home_team
    bp_rate = TEAM_BULLPEN_HR_RATE.get(opp, LEAGUE_AVG_BULLPEN)
    if bp_rate <= 0.027:
        reasons.append(f"❄️ {opp} bullpen is one of the toughest in MLB for home runs")
    elif bp_rate <= 0.030:
        reasons.append(f"❄️ {opp} bullpen is above average — tough late-game matchup")
    elif bp_rate >= 0.042:
        reasons.append(f"⚡ {opp} bullpen gives up home runs at one of the highest rates in MLB")
    elif bp_rate >= 0.038:
        reasons.append(f"{opp} bullpen is HR-prone — good chance of seeing relievers late")
 
    if not reasons:
        reasons = ["Limited historical data for this matchup"]
 
    return round(prob, 2), reasons
 
# ── Roster ─────────────────────────────────────────────────────
def get_roster(team_id):
    try:
        data = requests.get(
            f"https://statsapi.mlb.com/api/v1/teams/{team_id}/roster?rosterType=active",
            timeout=10
        ).json()
        return [
            {"id": p["person"]["id"], "name": p["person"]["fullName"]}
            for p in data.get("roster", [])
            if p.get("position", {}).get("abbreviation", "") not in ["P","SP","RP"]
        ]
    except Exception as e:
        print(f"  Roster error {team_id}: {e}")
        return []
 
# ── Odds ───────────────────────────────────────────────────────
def fetch_odds():
    """Fetch HR props from all books — pregame only, median consensus.
 
    Strategy:
      1. Skip any game whose commence_time is already in the past (no live/stale odds)
      2. Collect every book's line into raw_all[canonical_key][book_key]
      3. Resolve each player's line using the MEDIAN implied probability across
         all books — this automatically eliminates outlier lines like +4000 from
         a single soft book while the sharp market sits at +350.
    """
    print("Fetching HR odds from all sportsbooks (pregame only)...")
    now_utc = datetime.now(timezone.utc)
    # raw_all[canonical_key] = {book_key: implied_pct}
    raw_all  = {}   # {ckey: {book_key: {"player", "book_odds", "book_implied", "book"}}}
 
    try:
        ev_resp = requests.get(
            f"https://api.the-odds-api.com/v4/sports/baseball_mlb/events"
            f"?apiKey={ODDS_API_KEY}", timeout=15
        )
        if ev_resp.status_code != 200:
            print(f"  Events error: {ev_resp.status_code}")
            return {}
 
        events = ev_resp.json()
        if not isinstance(events, list):
            return {}
 
        # ── Filter to pregame events only ─────────────────────
        pregame = []
        for ev in events:
            ct = ev.get("commence_time", "")
            try:
                commence_dt = datetime.fromisoformat(ct.replace("Z", "+00:00"))
                # Allow up to 30 min after scheduled start (slight delay tolerance)
                if commence_dt > now_utc - timedelta(minutes=30):
                    pregame.append(ev)
            except Exception:
                pregame.append(ev)  # unknown time — include anyway
        print(f"  {len(events)} total games, {len(pregame)} pregame")
 
        for i, event in enumerate(pregame):
            eid = event.get("id", "")
            if not eid:
                continue
            if i > 0:
                time.sleep(1.2)
 
            pr = requests.get(
                f"https://api.the-odds-api.com/v4/sports/baseball_mlb/events/{eid}/odds"
                f"?apiKey={ODDS_API_KEY}&regions=us&markets=batter_home_runs"
                f"&oddsFormat=american", timeout=15
            )
            if pr.status_code == 429:
                print(f"  Rate limited — {len(raw_all)} players so far.")
                break
            if pr.status_code != 200:
                continue
            if i == 0:
                print(f"  Requests remaining: {pr.headers.get('x-requests-remaining','?')}")
 
            for bm in pr.json().get("bookmakers", []):
                book_key = bm.get("key", "")
                # DraftKings only — sharpest lines, most reliable HR props
                if book_key != "draftkings":
                    continue
                for market in bm.get("markets", []):
                    if market.get("key") != "batter_home_runs":
                        continue
                    for outcome in market.get("outcomes", []):
                        if str(outcome.get("name","")).lower() in ["no","under"]:
                            continue
                        player = outcome.get("description") or outcome.get("name","")
                        price  = outcome.get("price")
                        if not player or price is None:
                            continue
                        try:
                            odds_val = int(float(price))
                            implied  = american_to_implied(odds_val) * 100
                            # Valid HR prop range: 2%–38% implied
                            if implied < 2.0 or implied > 38.0:
                                continue
                            ckey = _name_key(player)
                            if ckey not in raw_all:
                                raw_all[ckey] = {}
                            existing = raw_all[ckey].get(book_key)
                            if not existing or odds_val > existing["book_odds"]:
                                raw_all[ckey][book_key] = {
                                    "player":       player.strip(),
                                    "book_odds":    odds_val,
                                    "book_implied": round(implied, 2),
                                    "book":         "draftkings",
                                }
                        except:
                            pass
    except Exception as e:
        print(f"  Odds error: {e}")
 
    # ── Resolve: DraftKings only — one entry per player ───────
    results = {}
    for ckey, books in raw_all.items():
        dk = books.get("draftkings")
        if dk:
            results[ckey] = {**dk, "n_books": 1}
 
    print(f"  {len(results)} HR props from DraftKings" if results else "  No DraftKings HR props posted yet.")
    return results
 
# ── Schedule + pitcher hand ────────────────────────────────────
pitcher_hand_cache = {}
def get_pitcher_hand(pitcher_id):
    if not pitcher_id:
        return "R"
    if pitcher_id in pitcher_hand_cache:
        return pitcher_hand_cache[pitcher_id]
    try:
        resp = requests.get(
            f"https://statsapi.mlb.com/api/v1/people/{pitcher_id}",
            timeout=8
        ).json()
        hand = resp.get("people",[{}])[0].get("pitchHand",{}).get("code","R")
        pitcher_hand_cache[pitcher_id] = hand
        return hand
    except:
        return "R"
 
def fetch_games():
    today = datetime.now(ET).strftime("%Y-%m-%d")
    print(f"Fetching games for {today}...")
    games = {}
    try:
        sched = requests.get(
            f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={today}"
            f"&hydrate=probablePitcher,team", timeout=15
        ).json()
        for de in sched.get("dates", []):
            for game in de.get("games", []):
                home    = game["teams"]["home"]["team"].get("abbreviation","")
                away    = game["teams"]["away"]["team"].get("abbreviation","")
                home_id = game["teams"]["home"]["team"]["id"]
                away_id = game["teams"]["away"]["team"]["id"]
                home_p  = game["teams"]["home"].get("probablePitcher", {})
                away_p  = game["teams"]["away"].get("probablePitcher", {})
                try:
                    dt_utc   = datetime.fromisoformat(game.get("gameDate","").replace("Z","+00:00"))
                    time_str = dt_utc.astimezone(ET).strftime("%-I:%M %p ET")
                except:
                    time_str = "TBD"
 
                home_p_hand = get_pitcher_hand(home_p.get("id"))
                away_p_hand = get_pitcher_hand(away_p.get("id"))
 
                key = f"{away}@{home}"
                games[key] = {
                    "home": home, "away": away,
                    "home_id": home_id, "away_id": away_id,
                    "time": time_str,
                    "home_pitcher":      home_p.get("fullName","TBD"),
                    "away_pitcher":      away_p.get("fullName","TBD"),
                    "home_pitcher_hand": home_p_hand,
                    "away_pitcher_hand": away_p_hand,
                    "weather": get_forecast(home),
                    "players": [],
                }
    except Exception as e:
        print(f"  Schedule error: {e}")
    print(f"  {len(games)} games found")
    return games
 
# ── Build predictions ──────────────────────────────────────────
def build_dashboard():
    odds  = fetch_odds()
    games = fetch_games()
    all_preds = []
 
    def match_odds(name):
        """Match roster name to odds dict with Unicode normalization + fallbacks.
 
        Pass 1 — exact canonical key match (handles accents, suffixes)
        Pass 2 — last name + first initial (catches slight spelling diffs)
        Pass 3 — last name only, only if unique match (catches nickname variants)
        """
        if not name or len(name.split()) < 2:
            return None
 
        # Pass 1: direct canonical key
        ckey = _name_key(name)
        if ckey in odds:
            return odds[ckey]
 
        # Decompose for fuzzy passes
        asc   = _ascii(name)
        parts = asc.split()
        last  = parts[-1]
        first_init = parts[0][0] if parts else ""
 
        # Pass 2: last name + first initial
        matches = [
            v for k, v in odds.items()
            if k.split() and k.split()[-1] == last and k.split()[0][:1] == first_init
        ]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            # Multiple first-initial matches (e.g. two players named "J. Smith") — skip
            return None
 
        # Pass 3: last name only (only accept if exactly one player in odds)
        last_matches = [v for k, v in odds.items() if k.split() and k.split()[-1] == last]
        if len(last_matches) == 1:
            return last_matches[0]
 
        return None
 
    players_scored = 0
    players_tried  = 0
 
    for gkey, gdata in games.items():
        home = gdata["home"]
        away = gdata["away"]
        for batters, opp_pitcher, opp_hand, team, opp_team in [
            # Away batters face the HOME team's starter + bullpen
            (get_roster(gdata["away_id"]), gdata["home_pitcher"], gdata["home_pitcher_hand"], away, home),
            # Home batters face the AWAY team's starter + bullpen
            (get_roster(gdata["home_id"]), gdata["away_pitcher"], gdata["away_pitcher_hand"], home, away),
        ]:
            for b in batters:
                bid, name = b["id"], b["name"]
                in_model  = len(hitter[hitter["batter"] == bid]) > 0
                players_tried += 1
                model_prob = None
                reasons    = []
 
                if in_model and opp_pitcher != "TBD":
                    try:
                        model_prob, reasons = predict_with_reasons(bid, opp_pitcher, home, opp_hand, opp_team)
                        players_scored += 1
                    except Exception as e:
                        print(f"  Error {name}: {e}")
 
                od           = match_odds(name)
                book_odds    = od["book_odds"]    if od else None
                book_implied = od["book_implied"] if od else None
                book_name    = od["book"]         if od else None
                n_books      = od.get("n_books", 1) if od else 0
                edge = round(model_prob - book_implied, 2) \
                       if (model_prob is not None and book_implied is not None) else None
 
                if   edge is None and model_prob is not None: value = "📊 Model only"
                elif edge is None:                             value = "⚠️ No data"
                elif edge >= 8:                                value = "🔥 Strong value"
                elif edge >= 4:                                value = "✅ Value"
                elif edge >= 0:                                value = "➖ Neutral"
                else:                                          value = "❌ Avoid"
 
                rec = {
                    "player":       name,
                    "batter_id":    bid,
                    "team":         team,
                    "game":         gkey,
                    "game_label":   f"{gdata['away']} @ {home}",
                    "time":         gdata["time"],
                    "pitcher":      opp_pitcher,
                    "pitcher_hand": opp_hand,
                    "home_team":    home,
                    "book_odds":    book_odds,
                    "book_implied": book_implied,
                    "book_name":    book_name,
                    "n_books":      n_books,
                    "model_prob":   model_prob,
                    "edge":         edge,
                    "value":        value,
                    "reasons":      reasons,
                }
                all_preds.append(rec)
                gdata["players"].append(rec)
 
    print(f"  Players tried: {players_tried} | Players scored: {players_scored}")
 
    def sort_key(r):
        if r["edge"] is not None:       return (1, r["edge"])
        if r["model_prob"] is not None: return (0, r["model_prob"])
        return (-1, 0)
 
    all_preds.sort(key=sort_key, reverse=True)
    for g in games.values():
        g["players"].sort(key=lambda r: r["model_prob"] or 0, reverse=True)
 
    value_count = sum(1 for r in all_preds if r["edge"] is not None and r["edge"] > 0)
    model_count = sum(1 for r in all_preds if r["model_prob"] is not None)
    print(f"  Model: {model_count} | Value picks: {value_count}")
    return all_preds, games
 
# ── HTML generation ────────────────────────────────────────────
def generate_html(all_preds, games):
    now_et   = datetime.now(ET)
    date_big = now_et.strftime("%A, %B %d, %Y")
    updated  = now_et.strftime("%I:%M %p ET")
 
    def odds_fmt(o):
        if o is None: return "—"
        return f"+{o}" if o > 0 else str(o)
 
    def edge_badge(e):
        if e is None: return '<span class="badge grey">Model Only</span>'
        if e >= 8:    return f'<span class="badge red">+{e:.1f}% edge 🔥</span>'
        if e >= 4:    return f'<span class="badge green">+{e:.1f}% edge ✅</span>'
        if e >= 0:    return f'<span class="badge yellow">+{e:.1f}% edge</span>'
        return              f'<span class="badge dark">Avoid ({e:.1f}%)</span>'
 
    def prob_bar(p):
        if p is None: return "—"
        color = "#e74c3c" if p >= 15 else "#e67e22" if p >= 10 else "#3498db"
        return (f'<div class="prob-wrap">'
                f'<div class="prob-bar" style="width:{min(p*5,100):.0f}%;background:{color}"></div>'
                f'<span class="prob-label">{p:.1f}%</span></div>')
 
    def player_card(r, rank=None):
        rank_html = f'<span class="rank">#{rank}</span>' if rank else ""
        # Reasons: ⚡ = super outlier (gold), ❄️ = cold/bad (icy blue)
        reasons_html = ""
        for reason in (r["reasons"] or []):
            if reason.startswith("⚡"):
                reasons_html += f'<li class="super-outlier">{reason}</li>'
            elif reason.startswith("❄"):
                reasons_html += f'<li class="cold-outlier">{reason}</li>'
            else:
                reasons_html += f'<li>{reason}</li>'
        if not reasons_html:
            reasons_html = "<li>Limited data for this matchup</li>"
 
        pitcher_in_model = len(pitcher[pitcher["player_name"] == to_statcast_name(r["pitcher"])]) > 0
        pitcher_note = "" if pitcher_in_model else \
            '<p class="note">⚠️ Pitcher not in model — using hitter data only</p>'
        book_lbl = "DraftKings"
        edge_str = (f"+{r['edge']:.1f}%" if r["edge"] and r["edge"] > 0
                    else f"{r['edge']:.1f}%" if r["edge"] is not None else "—")
        ph_badge = (f'<span class="ph-badge lhp">vs LHP</span>'
                    if r.get("pitcher_hand") == "L"
                    else f'<span class="ph-badge rhp">vs RHP</span>')
        # Build odds source line
        if r['book_odds'] is not None:
            odds_source = (
                f'<div class="odds-source">'
                f'<span class="book-tag">{book_lbl}</span>'
                f'&nbsp;<span class="odds-num">{odds_fmt(r["book_odds"])}</span>'
                f'&nbsp;<span class="implied-num">({r["book_implied"]:.1f}% implied)</span>'
                f'&nbsp;→&nbsp;<span class="model-num">Model: {r["model_prob"]:.1f}%</span>'
                f'&nbsp;→&nbsp;<span class="edge-num {"edge-pos" if (r["edge"] or 0) > 0 else "edge-neg"}">Edge: {edge_str}</span>'
                f'</div>'
            )
        else:
            odds_source = (
                f'<div class="odds-source">'
                f'<span class="no-odds">No odds posted yet</span>'
                f'&nbsp;·&nbsp;<span class="model-num">Model: {r["model_prob"]:.1f}%</span>'
                f'</div>'
            ) if r["model_prob"] else ''
 
        return f"""
<div class="card">
  <div class="card-header">
    {rank_html}
    <span class="player-name">{r['player']}</span>
    <span class="team-badge">{r['team']}</span>
    {ph_badge}
    {edge_badge(r['edge'])}
  </div>
  <div class="card-body">
    {odds_source}
    <div class="stats-row" style="margin-top:10px">
      <div class="stat"><label>Model Prob</label>{prob_bar(r['model_prob'])}</div>
      <div class="stat"><label>Implied %</label><span>{f"{r['book_implied']:.1f}%" if r['book_implied'] else "—"}</span></div>
      <div class="stat"><label>Edge</label><span class="edge-val">{edge_str}</span></div>
    </div>
    <p class="matchup">vs <strong>{r['pitcher']}</strong> · {r['game_label']} · {r['time']}</p>
    {pitcher_note}
    <div class="reasons">
      <strong>Key factors:</strong>
      <ul>{reasons_html}</ul>
    </div>
  </div>
</div>"""
 
    # ── Two top-picks tabs ────────────────────────────────────
    # Tab 1: Highest Probability (sort by model_prob)
    top_prob = [r for r in all_preds if r["model_prob"] is not None]
    top_prob.sort(key=lambda r: r["model_prob"] or 0, reverse=True)
    top_prob_html = "\n".join(player_card(r, i+1) for i, r in enumerate(top_prob[:10])) \
                    if top_prob else '<p class="empty">No predictions yet.</p>'
 
    # Tab 2: Best Value — sort by edge-to-implied RATIO (relative mispricing).
    # Example: model=14%, book implied=6% → ratio=1.33 ranks higher than
    #          model=22%, book implied=20% → ratio=0.10, even though raw edge is similar.
    # This surfaces underdog plays where the book is proportionally way off,
    # rather than just repeating the highest-probability list.
    top_edge = [
        r for r in all_preds
        if r["edge"] is not None and r["edge"] > 2          # minimum 2% raw edge
        and r["book_implied"] is not None and r["book_implied"] > 0
    ]
    top_edge.sort(
        key=lambda r: (r["edge"] or 0) / max(r["book_implied"] or 1, 1),
        reverse=True,
    )
    top_edge_html = "\n".join(player_card(r, i+1) for i, r in enumerate(top_edge[:10])) \
                    if top_edge else '<p class="empty">No value picks with odds posted yet.</p>'
 
    top_picks_tabs = f"""
  <div class="tab-bar top-tab-bar">
    <button class="top-tab-btn active" onclick="showTopTab('prob')">🎯 Highest Probability</button>
    <button class="top-tab-btn" onclick="showTopTab('edge')">💰 Best Value / Edge</button>
  </div>
  <div id="tab-prob" class="top-tab-panel active">{top_prob_html}</div>
  <div id="tab-edge" class="top-tab-panel">{top_edge_html}</div>"""
 
    # ── Helper: pitcher stat box ───────────────────────────────
    def pitcher_box(pname, hand, team_color="rhp"):
        if not pname or pname == "TBD":
            return f'<div class="pitcher-box"><div class="pb-name">TBD <span class="ph-badge {team_color}">?HP</span></div><div class="pb-stats"><span class="no-data">Pitcher TBD</span></div></div>'
        p_row = pitcher[pitcher["player_name"] == to_statcast_name(pname)]
        if len(p_row) == 0:
            last = pname.split()[-1].lower()
            p_row = pitcher[pitcher["player_name"].str.lower().str.contains(last, na=False)].head(1)
        hc = "lhp" if hand == "L" else "rhp"
        hand_badge = f'<span class="ph-badge {hc}">{hand}HP</span>'
        if len(p_row) == 0:
            return f'<div class="pitcher-box"><div class="pb-name">{pname} {hand_badge}</div><div class="pb-stats"><span class="no-data">No Statcast data</span></div></div>'
        pr = p_row.iloc[0]
        def _ps(col, fmt, label, mult=1):
            v = pr.get(col)
            if v is None or (isinstance(v, float) and np.isnan(v)): return ""
            return f'<div class="pb-stat"><span class="pb-label">{label}</span><span class="pb-val">{fmt.format(float(v)*mult)}</span></div>'
        hr_rate = pr.get("p_hr_rate_allowed")
        hr9 = f'{float(hr_rate)*27:.2f}' if hr_rate and not np.isnan(float(hr_rate)) else "—"
        stats = (
            f'<div class="pb-stat"><span class="pb-label">HR/9</span><span class="pb-val">{hr9}</span></div>'
            + _ps("p_barrel_pct_allowed",   "{:.1f}%", "Barrel%",   100)
            + _ps("p_hard_hit_pct_allowed", "{:.1f}%", "Hard Hit%", 100)
            + _ps("p_exit_velo_allowed",    "{:.1f}",  "Exit Velo")
            + _ps("p_hr_rate_allowed",      "{:.2f}%", "HR/AB",     100)
        )
        return f'<div class="pitcher-box"><div class="pb-name">{pname} {hand_badge}</div><div class="pb-stats">{stats}</div></div>'
 
    # ── Helper: lineup table for one side ─────────────────────
    def lineup_table(players):
        if not players:
            return '<p class="empty" style="padding:12px">No data yet</p>'
        rows = ""
        for r in players:
            if r["model_prob"] is None:
                continue
            prob  = r["model_prob"]
            pcolor = "#e74c3c" if prob >= 15 else "#e67e22" if prob >= 10 else "#58a6ff"
            odds_str = (f'+{r["book_odds"]}' if r["book_odds"] and r["book_odds"] > 0
                        else str(r["book_odds"]) if r["book_odds"] else "—")
            edge_str  = f'+{r["edge"]:.1f}%' if r["edge"] and r["edge"] > 0 else (f'{r["edge"]:.1f}%' if r["edge"] else "—")
            edge_cls  = "ln-edge-pos" if r["edge"] and r["edge"] > 0 else "ln-edge-neg"
 
            # Raw hitter stats from profile
            h_row = hitter[hitter["batter"] == r["batter_id"]]
            stat_cells = ""
            if len(h_row) > 0:
                hr = h_row.iloc[0]
                def _hs(col, fmt, mult=1):
                    v = hr.get(col)
                    if v is None or (isinstance(v, float) and np.isnan(v)): return "<td class='ln-stat'>—</td>"
                    return f"<td class='ln-stat'>{fmt.format(float(v)*mult)}</td>"
                stat_cells = (_hs("h_barrel_pct",   "{:.0f}%", 100)
                            + _hs("h_exit_velo",    "{:.0f}")
                            + _hs("h_hard_hit_pct", "{:.0f}%", 100))
            else:
                stat_cells = "<td class='ln-stat'>—</td><td class='ln-stat'>—</td><td class='ln-stat'>—</td>"
 
            # Top reason (first bullet, strip emoji prefix for compactness)
            top_reason = ""
            if r["reasons"]:
                raw = r["reasons"][0]
                raw = raw.replace("⚡ ", "").replace("❄️ ", "")
                top_reason = raw[:55] + "…" if len(raw) > 55 else raw
 
            rows += f"""<tr class="ln-row">
  <td class="ln-name">{r['player']}</td>
  <td class="ln-prob" style="color:{pcolor}">{prob:.1f}%</td>
  <td class="ln-odds">{odds_str}</td>
  <td class="{edge_cls}">{edge_str}</td>
  {stat_cells}
  <td class="ln-reason">{top_reason}</td>
</tr>"""
        return f"""<table class="lineup-tbl">
<thead><tr>
  <th>Player</th><th>Prob</th><th>DK Odds</th><th>Edge</th>
  <th>Barrel%</th><th>Exit Velo</th><th>Hard Hit%</th><th>Top Factor</th>
</tr></thead>
<tbody>{rows}</tbody>
</table>"""
 
    tab_buttons = ""
    tab_panels  = ""
    compass     = ["N","NE","E","SE","S","SW","W","NW"]
    for i, (gkey, gdata) in enumerate(games.items()):
        w       = gdata.get("weather", {})
        dir_str = compass[int((w.get("wind_dir",0)+22.5)/45)%8] if w.get("wind_dir") else "—"
        weather_str = (f'🌡 {w.get("temp_f",0):.0f}°F &nbsp;·&nbsp; '
                       f'💨 {w.get("wind_speed",0):.0f} mph {dir_str} &nbsp;·&nbsp; '
                       f'💧 {w.get("humidity",0):.0f}% humidity') if w else ""
 
        away_players = sorted([r for r in gdata["players"] if r["team"] == gdata["away"] and r["model_prob"]],
                              key=lambda r: r["model_prob"] or 0, reverse=True)
        home_players = sorted([r for r in gdata["players"] if r["team"] == gdata["home"] and r["model_prob"]],
                              key=lambda r: r["model_prob"] or 0, reverse=True)
 
        away_pb = pitcher_box(gdata["away_pitcher"], gdata["away_pitcher_hand"])
        home_pb = pitcher_box(gdata["home_pitcher"], gdata["home_pitcher_hand"])
 
        active = " active" if i == 0 else ""
        tab_buttons += (f'<button class="tab-btn{active}" onclick="showTab(\'{gkey}\')">'
                        f'{gkey}<br><small>{gdata["time"]}</small></button>\n')
        tab_panels += f"""
<div id="tab-{gkey}" class="tab-panel{active}">
  <div class="game-weather">{weather_str}</div>
  <div class="matchup-grid">
    <div class="side-col">
      <div class="side-header away-header">✈️ {gdata['away']} <span class="side-sub">(Away)</span></div>
      {away_pb}
      <div class="lineup-wrap">{lineup_table(away_players)}</div>
    </div>
    <div class="vs-divider">VS</div>
    <div class="side-col">
      <div class="side-header home-header">🏠 {gdata['home']} <span class="side-sub">(Home)</span></div>
      {home_pb}
      <div class="lineup-wrap">{lineup_table(home_players)}</div>
    </div>
  </div>
</div>"""
 
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MLB HR Model — {date_big}</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');
  :root {{
    --bg:#0c0c0e;--surface:#111114;--card:#18181c;--border:#242428;
    --text:#f0f0f2;--muted:#5a5a6e;--soft:#8a8a9e;
    --green:#21c45d;--red:#f04f58;--amber:#f59e0b;--blue:#4f86f7;--purple:#8b5cf6;
  }}
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{background:var(--bg);color:var(--text);font-family:'Inter',system-ui,sans-serif;-webkit-font-smoothing:antialiased;line-height:1.5}}
 
  /* ── Header ── */
  .header{{background:#ffffff;padding:36px 20px 24px;text-align:center;border-bottom:1px solid #e2e2e6}}
  .header .date{{font-size:clamp(24px,4vw,44px);font-weight:900;color:#0a0a0c;letter-spacing:-1px}}
  .header .subtitle{{color:#5a5a6e;margin-top:6px;font-size:12px;letter-spacing:.06em;text-transform:uppercase;font-weight:600}}
 
  /* ── Layout ── */
  .container{{max-width:1120px;margin:0 auto;padding:24px 20px}}
  h2{{font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.12em;color:var(--soft);margin:32px 0 14px}}
 
  /* ── Cards ── */
  .card{{background:var(--card);border:1px solid var(--border);border-radius:12px;margin-bottom:10px;overflow:hidden;transition:border-color .15s}}
  .card:hover{{border-color:#3a3a44}}
  .card-header{{display:flex;align-items:center;gap:10px;flex-wrap:wrap;padding:14px 16px;border-bottom:1px solid var(--border)}}
  .rank{{background:var(--blue);color:#fff;font-weight:700;border-radius:6px;width:24px;height:24px;display:flex;align-items:center;justify-content:center;font-size:11px;flex-shrink:0}}
  .player-name{{font-size:17px;font-weight:800;letter-spacing:-.3px;color:var(--text)}}
  .team-badge{{background:var(--border);border-radius:4px;padding:2px 7px;font-size:11px;font-weight:600;color:var(--soft)}}
  .ph-badge{{border-radius:4px;padding:2px 7px;font-size:10px;font-weight:700;letter-spacing:.03em}}
  .ph-badge.lhp{{background:rgba(139,92,246,.15);color:var(--purple)}}
  .ph-badge.rhp{{background:rgba(79,134,247,.12);color:var(--blue)}}
  .card-body{{padding:14px 16px}}
  .stats-row{{display:flex;gap:24px;flex-wrap:wrap;margin-bottom:12px}}
  .stat{{display:flex;flex-direction:column;gap:2px}}
  .stat label{{font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:.06em;font-weight:500}}
  .edge-val{{font-weight:700;color:var(--green)}}
  .matchup{{font-size:12px;color:var(--muted);margin-bottom:10px}}
  .note{{font-size:12px;color:var(--amber);margin-bottom:10px}}
 
  /* ── Key factors ── */
  .reasons{{background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:10px 12px}}
  .reasons strong{{font-size:10px;text-transform:uppercase;letter-spacing:.07em;color:var(--muted);font-weight:600}}
  .reasons ul{{margin-top:6px;padding-left:14px}}
  .reasons li{{font-size:12px;margin-bottom:3px;line-height:1.55;color:#c0c0d0}}
  .super-outlier{{color:#f0b429 !important;font-weight:600}}
  .cold-outlier{{color:#60a5fa !important;font-weight:600}}
 
  /* ── Odds source row ── */
  .odds-source{{background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:9px 12px;font-size:13px;display:flex;align-items:center;gap:8px;flex-wrap:wrap}}
  .book-tag{{background:rgba(79,134,247,.1);color:var(--blue);border-radius:4px;padding:2px 8px;font-weight:700;font-size:10px;text-transform:uppercase;letter-spacing:.04em}}
  .odds-num{{font-size:19px;font-weight:900;color:var(--green);letter-spacing:-.4px}}
  .implied-num{{color:var(--muted);font-size:12px}}
  .model-num{{color:var(--blue);font-weight:600;font-size:13px}}
  .edge-pos{{color:var(--green);font-weight:700}}
  .edge-neg{{color:var(--red);font-weight:700}}
  .no-odds{{color:var(--muted);font-size:12px}}
 
  /* ── Prob bar ── */
  .prob-wrap{{display:flex;align-items:center;gap:8px}}
  .prob-bar{{height:4px;border-radius:2px;min-width:4px;max-width:100px;opacity:.85}}
  .prob-label{{font-size:17px;font-weight:900;letter-spacing:-.4px;color:var(--text)}}
 
  /* ── Badges ── */
  .badge{{border-radius:6px;padding:3px 8px;font-size:11px;font-weight:600;letter-spacing:.02em}}
  .badge.red{{background:rgba(240,79,88,.12);color:var(--red)}}
  .badge.green{{background:rgba(33,196,93,.12);color:var(--green)}}
  .badge.yellow{{background:rgba(245,158,11,.12);color:var(--amber)}}
  .badge.grey{{background:rgba(90,90,110,.15);color:var(--soft)}}
  .badge.dark{{background:rgba(79,134,247,.1);color:var(--blue)}}
 
  /* ── Game tabs ── */
  .tab-bar{{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:18px;border-bottom:1px solid var(--border);padding-bottom:0}}
  .tab-btn{{background:transparent;border:none;border-bottom:2px solid transparent;color:var(--muted);padding:8px 12px;cursor:pointer;font-size:12px;font-weight:600;text-align:center;line-height:1.4;transition:color .15s,border-color .15s;margin-bottom:-1px;font-family:inherit}}
  .tab-btn.active{{color:var(--text);border-bottom-color:var(--blue)}}
  .tab-btn:hover:not(.active){{color:var(--soft)}}
  .tab-panel{{display:none}}.tab-panel.active{{display:block}}
 
  /* ── Top picks tabs ── */
  .top-tab-bar{{display:flex;gap:4px;margin-bottom:18px;background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:4px;width:fit-content}}
  .top-tab-btn{{background:transparent;border:none;color:var(--muted);padding:7px 16px;border-radius:6px;cursor:pointer;font-size:13px;font-weight:600;transition:all .12s;font-family:inherit}}
  .top-tab-btn.active{{background:var(--card);color:var(--text);box-shadow:0 1px 3px rgba(0,0,0,.4)}}
  .top-tab-btn:hover:not(.active){{color:var(--soft)}}
  .top-tab-panel{{display:none}}.top-tab-panel.active{{display:block}}
 
  /* ── Game matchup layout ── */
  .game-weather{{font-size:11px;color:var(--muted);text-align:center;padding:6px 0 16px;letter-spacing:.03em}}
  .matchup-grid{{display:grid;grid-template-columns:1fr 48px 1fr;gap:0;align-items:start}}
  .vs-divider{{display:flex;align-items:flex-start;justify-content:center;padding-top:52px;font-weight:700;color:var(--border);font-size:11px;letter-spacing:.1em}}
  .side-col{{min-width:0}}
  .side-header{{font-size:14px;font-weight:800;padding:8px 0 10px;border-bottom:1px solid var(--border);margin-bottom:12px;letter-spacing:-.2px}}
  .away-header{{color:var(--blue)}}
  .home-header{{color:var(--green)}}
  .side-sub{{font-size:11px;font-weight:400;color:var(--muted);margin-left:4px}}
 
  /* ── Pitcher box ── */
  .pitcher-box{{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:12px 14px;margin-bottom:12px}}
  .pb-name{{font-size:13px;font-weight:700;margin-bottom:10px;letter-spacing:-.1px}}
  .pb-stats{{display:flex;gap:16px;flex-wrap:wrap}}
  .pb-stat{{display:flex;flex-direction:column;gap:2px}}
  .pb-label{{font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:.05em;font-weight:500}}
  .pb-val{{font-size:13px;font-weight:700}}
 
  /* ── Lineup table ── */
  .lineup-wrap{{overflow-x:auto}}
  .lineup-tbl{{width:100%;border-collapse:collapse;font-size:12px}}
  .lineup-tbl thead th{{padding:5px 8px;text-align:left;font-size:10px;text-transform:uppercase;letter-spacing:.05em;color:var(--muted);border-bottom:1px solid var(--border);white-space:nowrap;font-weight:500}}
  .ln-row{{border-bottom:1px solid var(--border);transition:background .1s}}
  .ln-row:hover{{background:rgba(255,255,255,.025)}}
  .ln-row td{{padding:8px 8px;vertical-align:middle}}
  .ln-name{{font-weight:700;white-space:nowrap;font-size:13px;color:var(--text)}}
  .ln-prob{{font-weight:900;font-size:14px;white-space:nowrap;letter-spacing:-.3px}}
  .ln-odds{{color:var(--green);font-weight:700;font-size:12px}}
  .ln-edge-pos{{color:var(--green);font-weight:700;font-size:12px}}
  .ln-edge-neg{{color:var(--red);font-size:12px}}
  .ln-stat{{color:var(--soft);font-size:12px}}
  .ln-reason{{color:var(--muted);font-size:11px;max-width:200px;line-height:1.4}}
  .no-data{{color:var(--muted);font-style:italic;font-size:12px}}
  .empty{{color:var(--muted);padding:20px 0;text-align:center;font-size:13px}}
  .footer{{text-align:center;color:var(--muted);font-size:11px;padding:24px 20px;border-top:1px solid var(--border);margin-top:40px;letter-spacing:.03em}}
  @media(max-width:700px){{.matchup-grid{{grid-template-columns:1fr}}.vs-divider{{display:none}}.stats-row{{gap:16px}}.card-header{{flex-direction:column;align-items:flex-start}}.top-tab-bar{{width:100%}}}}
</style>
</head>
<body>
<div class="header">
  <div class="date">{date_big}</div>
  <div class="subtitle">MLB Home Run Model &nbsp;·&nbsp; Updated {updated}</div>
</div>
<div class="container">
  <h2>🏆 Top Picks Today</h2>
  {top_picks_tabs}
  <h2>📅 Today's Games</h2>
  {'<p class="empty">No games scheduled today.</p>' if not games else ''}
  <div class="tab-bar">{tab_buttons}</div>
  {tab_panels}
</div>
<div class="footer">Built with Statcast data · Probabilities based on standard deviations from MLB average · For entertainment only</div>
<script>
function showTab(id) {{
  document.querySelectorAll(".tab-panel").forEach(el => el.classList.remove("active"));
  document.querySelectorAll(".tab-btn").forEach(el => el.classList.remove("active"));
  document.getElementById("tab-" + id).classList.add("active");
  event.currentTarget.classList.add("active");
}}
function showTopTab(id) {{
  document.querySelectorAll(".top-tab-panel").forEach(el => el.classList.remove("active"));
  document.querySelectorAll(".top-tab-btn").forEach(el => el.classList.remove("active"));
  document.getElementById("tab-" + id).classList.add("active");
  event.currentTarget.classList.add("active");
}}
</script>
</body>
</html>"""
 
if __name__ == "__main__":
    all_preds, games = build_dashboard()
    html = generate_html(all_preds, games)
    with open("index.html", "w") as f:
        f.write(html)
    print("\nindex.html saved — open it in your browser!")