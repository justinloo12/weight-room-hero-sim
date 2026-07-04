import math
import os
import pickle
import unicodedata
import csv
import pathlib
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
 
from betting_math import (
    american_to_implied,
    implied_to_american,
    edge_percentage,
    expected_roi,
    kelly_fraction,
)

ODDS_API_KEY = os.environ.get("ODDS_API_KEY")
if not ODDS_API_KEY:
    raise SystemExit(
        "ODDS_API_KEY environment variable is not set. "
        "Get a key from https://the-odds-api.com and export it "
        "(see .env.example)."
    )
 
print("Loading profiles...")
hitter  = pd.read_csv("hitter_profiles.csv")
pitcher = pd.read_csv("pitcher_profiles.csv")
print(f"  Hitter profiles:  {len(hitter):,}")
print(f"  Pitcher profiles: {len(pitcher):,}")

print("Loading batter vs pitcher history...")
BVP_HISTORY = pd.DataFrame()
for _history_path in ["homerun_data_all.csv", "homerun_data_enriched.csv"]:
    try:
        _raw = pd.read_csv(
            _history_path,
            usecols=lambda c: c in {
                "game_pk", "at_bat_number", "batter", "player_name", "events",
                "launch_speed", "estimated_woba_using_speedangle", "launch_speed_angle", "barrel"
            },
            low_memory=False,
        )
        _raw = _raw[_raw["events"].notna()].copy()
        if _raw.empty:
            continue
        _raw = _raw.drop_duplicates(["game_pk", "at_bat_number", "batter", "player_name"], keep="last")
        official_ab_events = {
            "single", "double", "triple", "home_run", "field_out", "force_out",
            "grounded_into_double_play", "fielders_choice", "fielders_choice_out",
            "double_play", "triple_play", "strikeout", "strikeout_double_play",
            "other_out", "reached_on_error", "field_error",
        }
        hit_events = {"single", "double", "triple", "home_run"}
        _raw["is_ab"] = _raw["events"].isin(official_ab_events).astype(int)
        _raw["is_hit"] = _raw["events"].isin(hit_events).astype(int)
        _raw["is_hr"] = (_raw["events"] == "home_run").astype(int)
        _raw["hard_hit"] = (_raw.get("launch_speed", pd.Series(index=_raw.index, dtype=float)).fillna(0) >= 95).astype(int)
        if "barrel" not in _raw.columns:
            _raw["barrel"] = (_raw.get("launch_speed_angle", pd.Series(index=_raw.index, dtype=float)) == 6).astype(int)
        else:
            _raw["barrel"] = _raw["barrel"].fillna(0).astype(int)
        BVP_HISTORY = (
            _raw.groupby(["batter", "player_name"], as_index=False)
            .agg(
                bvp_ab=("is_ab", "sum"),
                bvp_hits=("is_hit", "sum"),
                bvp_hr=("is_hr", "sum"),
                bvp_xwoba=("estimated_woba_using_speedangle", "mean"),
                bvp_barrel_pct=("barrel", "mean"),
                bvp_hard_hit_pct=("hard_hit", "mean"),
            )
        )
        BVP_HISTORY["bvp_avg"] = np.where(
            BVP_HISTORY["bvp_ab"] > 0,
            BVP_HISTORY["bvp_hits"] / BVP_HISTORY["bvp_ab"],
            np.nan,
        )
        print(f"  Loaded BvP history from {_history_path}: {len(BVP_HISTORY):,} batter/pitcher pairs")
        break
    except Exception as _e:
        continue
if BVP_HISTORY.empty:
    print("  No BvP history loaded")
 
ET = ZoneInfo("America/New_York")
PICKS_HISTORY = pathlib.Path("picks_history.csv")
TRACKED_STAKE = 100.0
DEFAULT_PROB_SHRINK = 0.82
MIN_CALIBRATION_SAMPLES = 25
CALIBRATION_LOOKBACK_DAYS = 45
MIN_TRACK_MODEL_PROB = 8.0
MIN_TRACK_EDGE = 1.25
MIN_VALUE_EDGE = 2.75
MAX_VALUE_AMERICAN_ODDS = 700
PROFIT_POLICY_MIN_SAMPLE = 8
PROFIT_POLICY_MIN_DAYS = 2
DEFAULT_MAX_TRACKED_PER_DAY = 3
ENABLE_VALUE_TRACKING = False
BASE_BANKROLL = 5000.0
KELLY_FRACTION = 0.20
MAX_STAKE_PCT_BANKROLL = 0.03
MIN_STAKE = 25.0
MAX_STAKE = 150.0
SURFACE_EDGE_MIN = 2.0
STRONG_EDGE_MIN = 10.0
MAX_SURFACE_JUICE = -140

EXPECTED_PA_BY_SLOT = {
    1: 4.85, 2: 4.65, 3: 4.55, 4: 4.45, 5: 4.30,
    6: 4.15, 7: 4.00, 8: 3.88, 9: 3.78,
}
DEFAULT_EXPECTED_PA = 4.15

# ── Probability source ─────────────────────────────────────────
# 'structural_v2' (default): transparent structural model with lineup-slot
# E[PA] and park factors by handedness. On the 2026 holdout it beat the GBM
# on every metric (see EVALUATION.md), so it is the default price setter.
# 'gbm': the legacy GBM + heuristic stack (predict_with_reasons).
PROB_SOURCE = os.environ.get("HR_PROB_SOURCE", "structural_v2").strip().lower()
STRUCTURAL_V2_PATH = "structural_v2.pkl"
STRUCTURAL_V2 = None
if PROB_SOURCE == "structural_v2":
    try:
        from structural_model import StructuralModelV2
        STRUCTURAL_V2 = StructuralModelV2.load(STRUCTURAL_V2_PATH)
        print(f"Probability source: structural_v2 ({STRUCTURAL_V2_PATH})")
    except Exception as _e:
        print(f"WARNING: could not load {STRUCTURAL_V2_PATH} ({_e}); "
              f"falling back to the gbm source. Rebuild with: python3 structural_model.py build")
        PROB_SOURCE = "gbm"
if PROB_SOURCE != "structural_v2":
    print("Probability source: gbm (legacy)")

# ── Odds API quota guard ───────────────────────────────────────
# The Odds API plan has a 20,000 credit/month budget and player-prop
# endpoints bill ~1 credit per event per market per region. Keep the
# per-event pulls to ONE region + ONE market, and refuse to dip into the
# reserve below this floor (override with ODDS_API_MIN_REMAINING).
ODDS_API_MIN_REMAINING = int(os.environ.get("ODDS_API_MIN_REMAINING", "500"))
ODDS_QUOTA_LOG = pathlib.Path("odds_quota_log.csv")

PARK_OUTFIELD_BEARINGS = {
    "ARI": 20, "ATL": 35, "BAL": 45, "BOS": 55, "CHC": 35, "CIN": 38,
    "CLE": 35, "COL": 20, "CWS": 32, "DET": 30, "HOU": 42, "KC": 40,
    "LAA": 55, "LAD": 45, "MIA": 15, "MIL": 45, "MIN": 30, "NYM": 28,
    "NYY": 50, "OAK": 45, "PHI": 38, "PIT": 38, "SD": 52, "SEA": 40,
    "SF": 45, "STL": 42, "TB": 40, "TEX": 42, "TOR": 35, "WSH": 35,
}


def _load_recent_model_calibration():
    """Estimate over/under-confidence from recently settled picks."""
    default = {
        "factor": DEFAULT_PROB_SHRINK,
        "samples": 0,
        "avg_model": None,
        "avg_actual": None,
    }
    if not PICKS_HISTORY.exists():
        return default

    def _to_float(val):
        try:
            return float(val)
        except Exception:
            return None

    preds = []
    actuals = []
    today = datetime.now(ET).date()

    try:
        with open(PICKS_HISTORY, newline="") as f:
            rows = list(csv.DictReader(f))
    except Exception:
        return default

    for row in rows:
        result = (row.get("result") or "").strip()
        if result not in {"HR", "No HR"}:
            continue

        date_str = (row.get("date") or row.get("game_date") or "").strip()
        if date_str:
            try:
                age_days = (today - datetime.strptime(date_str, "%Y-%m-%d").date()).days
                if age_days < 0 or age_days > CALIBRATION_LOOKBACK_DAYS:
                    continue
            except Exception:
                continue

        model_prob = _to_float(row.get("model_prob"))
        if model_prob is None or model_prob <= 0 or model_prob >= 100:
            continue

        preds.append(model_prob / 100.0)
        actuals.append(1.0 if result == "HR" else 0.0)

    n = len(preds)
    if n < MIN_CALIBRATION_SAMPLES:
        return default

    avg_model = float(np.mean(preds))
    avg_actual = float(np.mean(actuals))
    raw_factor = (avg_actual / avg_model) if avg_model > 0 else DEFAULT_PROB_SHRINK
    shrink = n / (n + 120.0)
    factor = (1.0 - shrink) * DEFAULT_PROB_SHRINK + shrink * raw_factor
    factor = max(0.55, min(1.00, factor))
    return {
        "factor": factor,
        "samples": n,
        "avg_model": avg_model,
        "avg_actual": avg_actual,
    }


def _load_profit_policy():
    """Walk-forward policy selection: tune on past days, score on next unseen day."""
    policy = {
        "min_model_prob": 16.0,
        "min_edge": 0.0,
        "max_odds": 700.0,
        "max_picks_per_day": DEFAULT_MAX_TRACKED_PER_DAY,
        "roi": None,
        "sample": 0,
        "days": 0,
    }
    if not PICKS_HISTORY.exists():
        return policy

    try:
        with open(PICKS_HISTORY, newline="") as f:
            rows = list(csv.DictReader(f))
    except Exception:
        return policy

    settled = []
    for row in rows:
        result = (row.get("result") or "").strip()
        if result not in {"HR", "No HR"}:
            continue
        try:
            model_prob = float(row.get("model_prob") or "")
            edge = float(row.get("edge") or "")
            odds = float(row.get("book_odds") or "")
        except Exception:
            continue
        try:
            pnl = float(row.get("pnl") or "")
        except Exception:
            pnl = (100.0 * (odds / 100.0 if odds > 0 else 100.0 / abs(odds))) if result == "HR" else -100.0
        settled.append(
            {
                "date": (row.get("date") or "").strip(),
                "player_key": (row.get("batter_id") or row.get("player") or "").strip(),
                "model_prob": model_prob,
                "edge": edge,
                "odds": odds,
                "pnl": pnl,
            }
        )

    if len(settled) < PROFIT_POLICY_MIN_SAMPLE:
        return policy

    # De-duplicate same player/date across overlapping pick groups.
    collapsed = {}
    for row in settled:
        key = (row["date"], row["player_key"])
        prev = collapsed.get(key)
        if prev is None or (row["edge"], row["model_prob"]) > (prev["edge"], prev["model_prob"]):
            collapsed[key] = row
    base = list(collapsed.values())

    by_day = {}
    for r in base:
        if not r["date"]:
            continue
        by_day.setdefault(r["date"], []).append(r)
    dates = sorted(by_day.keys())
    if len(dates) < max(4, PROFIT_POLICY_MIN_DAYS + 1):
        return policy

    candidates = []
    for min_prob in [8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]:
        for min_edge in [-5, -2, 0, 1, 2, 3, 4, 5, 6, 8]:
            for max_odds in [300, 350, 400, 500, 600, 700, 900]:
                for topk in [1, 2, 3, 4, 5]:
                    candidates.append((float(min_prob), float(min_edge), float(max_odds), int(topk)))

    def _apply_param(rows_for_day, param):
        min_prob, min_edge, max_odds, topk = param
        filt = [
            r for r in rows_for_day
            if r["model_prob"] >= min_prob and r["edge"] >= min_edge and r["odds"] <= max_odds
        ]
        filt = sorted(filt, key=lambda x: (x["edge"], x["model_prob"]), reverse=True)
        return filt[:topk]

    min_train_days = max(2, PROFIT_POLICY_MIN_DAYS)
    oos_summary = {}  # param -> stats
    selected_counts = {}  # param -> count selected by train-only optimizer

    for idx in range(min_train_days, len(dates)):
        train_dates = set(dates[:idx])
        test_date = dates[idx]

        train_rows = [r for d in train_dates for r in by_day.get(d, [])]
        if len(train_rows) < PROFIT_POLICY_MIN_SAMPLE:
            continue

        # Choose best param using TRAIN ONLY
        best_train = None
        for param in candidates:
            picks = []
            for d in train_dates:
                picks.extend(_apply_param(by_day.get(d, []), param))
            n = len(picks)
            if n < PROFIT_POLICY_MIN_SAMPLE:
                continue
            pnl = float(sum(r["pnl"] for r in picks))
            roi = pnl / (100.0 * n)
            robust = roi * (n / (n + 12.0))
            score = (robust, roi, pnl, n)
            if best_train is None or score > best_train["score"]:
                best_train = {"param": param, "score": score}

        if best_train is None:
            continue

        chosen = best_train["param"]
        selected_counts[chosen] = selected_counts.get(chosen, 0) + 1

        # Evaluate chosen param on unseen NEXT day
        test_picks = _apply_param(by_day.get(test_date, []), chosen)
        if not test_picks:
            continue
        n_test = len(test_picks)
        pnl_test = float(sum(r["pnl"] for r in test_picks))
        hit_test = float(sum(1 for r in test_picks if r["pnl"] > 0) / n_test)

        stats = oos_summary.setdefault(chosen, {"pnl": 0.0, "n": 0, "days": 0, "hits": 0.0})
        stats["pnl"] += pnl_test
        stats["n"] += n_test
        stats["days"] += 1
        stats["hits"] += hit_test * n_test

    if not oos_summary:
        return policy

    # Pick deploy param by OOS robustness, with tie-break on how often it was selected.
    best_param = None
    best_score = None
    for param, stats in oos_summary.items():
        roi = stats["pnl"] / (100.0 * stats["n"]) if stats["n"] else -999.0
        robust = roi * (stats["n"] / (stats["n"] + 8.0))
        sel_cnt = selected_counts.get(param, 0)
        score = (robust, roi, stats["pnl"], stats["n"], sel_cnt)
        if best_score is None or score > best_score:
            best_score = score
            best_param = (param, stats, roi)

    if best_param is None:
        return policy

    (min_prob, min_edge, max_odds, topk), stats, roi = best_param
    policy.update(
        {
            "min_model_prob": min_prob,
            "min_edge": min_edge,
            "max_odds": max_odds,
            "max_picks_per_day": topk,
            "roi": roi,
            "sample": stats["n"],
            "days": stats["days"],
        }
    )
    return policy


MODEL_CALIBRATION = _load_recent_model_calibration()
if MODEL_CALIBRATION["samples"] >= MIN_CALIBRATION_SAMPLES:
    print(
        "Calibration: "
        f"{MODEL_CALIBRATION['samples']} settled picks | "
        f"model avg {MODEL_CALIBRATION['avg_model']*100:.1f}% | "
        f"actual {MODEL_CALIBRATION['avg_actual']*100:.1f}% | "
        f"shrink factor {MODEL_CALIBRATION['factor']:.3f}"
    )
else:
    print(
        "Calibration: insufficient settled picks for fresh estimate "
        f"(using default shrink {DEFAULT_PROB_SHRINK:.2f})"
    )


PROFIT_POLICY = _load_profit_policy()
if PROFIT_POLICY["sample"] >= PROFIT_POLICY_MIN_SAMPLE:
    print(
        "Profit policy: "
        f"ROI {PROFIT_POLICY['roi']*100:+.1f}% | "
        f"n={PROFIT_POLICY['sample']} picks across {PROFIT_POLICY['days']} days | "
        f"min model {PROFIT_POLICY['min_model_prob']:.1f}% | "
        f"min edge {PROFIT_POLICY['min_edge']:.2f}% | "
        f"max odds +{int(PROFIT_POLICY['max_odds'])} | "
        f"max picks/day {PROFIT_POLICY['max_picks_per_day']}"
    )
else:
    print(
        "Profit policy: using conservative defaults "
        f"(min model {PROFIT_POLICY['min_model_prob']:.1f}%, "
        f"min edge {PROFIT_POLICY['min_edge']:.2f}%, "
        f"max odds +{int(PROFIT_POLICY['max_odds'])}, "
        f"max picks/day {PROFIT_POLICY['max_picks_per_day']})"
    )
 
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
    "fanatics":       "Fanatics",
    "espnbet":        "ESPN BET",
    "betparx":        "betPARX",
    "ballybet":       "Bally Bet",
    "hardrockbet":    "Hard Rock Bet",
    "fliff":          "Fliff",
    "rebet":          "ReBet",
    "pointsbet":      "PointsBet",
    "betus":          "BetUS",
    "mybookieag":     "MyBookie",
    "bovada":         "Bovada",
    "lowvig":         "LowVig",
    "consensus":      "Consensus",
}
 
# implied_to_american is imported from betting_math

BOOK_PRIORITY = [
    "draftkings", "fanduel", "betmgm", "williamhill_us", "caesars",
    "espnbet", "hardrockbet", "betparx", "ballybet", "betrivers",
    "betonlineag", "fliff", "fanatics", "betus", "mybookieag", "bovada"
]
 
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
 
# american_to_implied is imported from betting_math

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


def expected_pa_multiplier(slot):
    base = EXPECTED_PA_BY_SLOT.get(slot, DEFAULT_EXPECTED_PA)
    return base / DEFAULT_EXPECTED_PA


def wind_blowing_alignment(team, wind_dir_degrees):
    """Return positive when wind is blowing out to CF, negative when blowing in."""
    try:
        wind_from = float(wind_dir_degrees)
    except Exception:
        return 0.0
    bearing = PARK_OUTFIELD_BEARINGS.get(team)
    if bearing is None:
        return 0.0
    wind_to = (wind_from + 180.0) % 360.0
    diff = ((wind_to - bearing + 180.0) % 360.0) - 180.0
    return math.cos(math.radians(diff))


def compute_weather_stack(team, wx):
    temp = _safe_float(wx.get("temp_f")) if wx else None
    wind = _safe_float(wx.get("wind_speed")) if wx else None
    wind_dir = _safe_float(wx.get("wind_dir")) if wx else None
    if temp is None:
        temp = 72.0
    if wind is None:
        wind = 0.0
    if wind_dir is None:
        wind_dir = 0.0

    alignment = wind_blowing_alignment(team, wind_dir)
    temp_mod = ((temp - 72.0) / 10.0) * 0.015
    wind_mod = 0.0
    if abs(alignment) > 0.15:
        wind_mod = (wind / 10.0) * 0.025 * alignment
    full_stack = temp > 75.0 and wind > 8.0 and alignment > 0.45
    combined_mod = temp_mod + wind_mod
    if full_stack:
        combined_mod *= 1.00
    else:
        combined_mod *= 0.45

    factors = 0
    if temp > 75:
        factors += 1
    if temp > 85:
        factors += 1
    if wind > 8 and alignment > 0.25:
        factors += 1
    if wind > 12 and alignment > 0.45:
        factors += 1
    if full_stack:
        factors += 1

    direction = "neutral"
    if alignment > 0.35:
        direction = "out"
    elif alignment < -0.35:
        direction = "in"

    return {
        "modifier": combined_mod,
        "score": int(max(0, min(5, factors))),
        "wind_alignment": alignment,
        "wind_direction": direction,
        "full_stack": full_stack,
        "temp_f": temp,
        "wind_speed": wind,
    }


def classify_edge(model_prob, book_implied, book_odds):
    if model_prob is None or book_implied is None or book_odds is None:
        return "Model Only", None, False
    edge = edge_percentage(model_prob, book_implied)
    bad_juice = book_odds < 0 and book_odds <= MAX_SURFACE_JUICE
    if bad_juice:
        return "Exclude", edge, False
    if edge >= STRONG_EDGE_MIN:
        return "Strong", edge, True
    if edge >= SURFACE_EDGE_MIN:
        return "Moderate", edge, True
    return "Monitor", edge, False


def detect_trap_flag(player, date_str, current_odds):
    if current_odds is None or not PICKS_HISTORY.exists():
        return None
    try:
        with open(PICKS_HISTORY, newline="") as f:
            rows = list(csv.DictReader(f))
    except Exception:
        return None
    same_day = [r for r in rows if (r.get("date") == date_str and r.get("player") == player and r.get("book_odds"))]
    if not same_day:
        return None
    try:
        earliest = int(float(same_day[0]["book_odds"]))
        current = int(float(current_odds))
    except Exception:
        return None
    if earliest > 0 and current > 0 and current <= earliest - 40:
        return f"DK line shortened from +{earliest} to +{current}"
    if earliest < 0 and current < 0 and current <= earliest - 20:
        return f"DK line got juicier from {earliest} to {current}"
    return None
 
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
        sigma_str = "well above MLB average"
        prefix    = ""
    elif z >= 1.0:
        sigma_str = "strong positive for HR upside"
        prefix    = ""
    elif z >= 0.5:
        sigma_str = "supportive HR signal"
        prefix    = ""
    elif z >= 0:
        sigma_str = "playable baseline"
        prefix    = ""
    elif cold_out:
        sigma_str = f"well below average — bottom {bot_pct:.0f}% in MLB"
        prefix    = "❄️ "
    else:
        sigma_str = "mild concern"
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
        "h_sweet_spot_pct":
            f"{prefix}Sweet-spot launch angle on {val*100:.1f}% of balls in play ({sigma_str})",
        "h_hr_contact_score":
            f"{prefix}Power-contact profile is built for home runs ({sigma_str})",
        "h_lifted_power_score":
            f"{prefix}Lifted power profile is especially HR-friendly ({sigma_str})",
        "h_hr_rate_vs_hand":
            f"{prefix}Today's handedness split is favorable for HR upside ({sigma_str})",
        "h_xwoba_vs_hand":
            f"{prefix}Quality of contact vs today's handedness is strong ({sigma_str})",
        "h_barrel_pct_vs_hand":
            f"{prefix}Barrel rate vs today's handedness stands out ({sigma_str})",
        "h_hard_hit_pct_vs_hand":
            f"{prefix}Hard-hit rate vs today's handedness is strong ({sigma_str})",
        "h_sweet_spot_pct_vs_hand":
            f"{prefix}Sweet-spot rate vs today's handedness is supportive ({sigma_str})",
        "h_pull_air_pct_vs_hand":
            f"{prefix}Pull-air profile vs today's handedness is HR-friendly ({sigma_str})",
        "h_hr_contact_score_vs_hand":
            f"{prefix}Power-contact profile vs today's handedness is strong ({sigma_str})",
        "h_lifted_power_score_vs_hand":
            f"{prefix}Lifted damage profile vs today's handedness is strong ({sigma_str})",
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
        "p_launch_angle_allowed":
            f"{prefix}Pitcher allows {val:.1f}° average launch angle ({sigma_str})",
        "p_sweet_spot_pct_allowed":
            f"{prefix}Pitcher allows sweet-spot launch angle on {val*100:.1f}% of contact ({sigma_str})",
        "p_pull_air_pct_allowed":
            f"{prefix}Pitcher gives up pull fly balls {val*100:.1f}% ({sigma_str})",
        "p_spin_into_barrel_pct":
            f"{prefix}Pitcher's spin hits barrel zone {val*100:.1f}% of pitches ({sigma_str})",
        "p_hr_contact_risk":
            f"{prefix}Pitcher's contact profile is very homer-prone ({sigma_str})",
        "p_lift_damage_risk":
            f"{prefix}Pitcher gives up the kind of lifted contact that turns into HRs ({sigma_str})",
        "p_hr_rate_allowed_vs_side":
            (f"{prefix}Pitcher is especially homer-prone to this side of the plate ({sigma_str})" if val > 0 else None),
        "p_barrel_pct_allowed_vs_side":
            f"{prefix}Pitcher gives up barrels to this side of the plate ({sigma_str})",
        "p_hard_hit_pct_allowed_vs_side":
            f"{prefix}Pitcher allows hard-hit contact to this side ({sigma_str})",
        "p_exit_velo_allowed_vs_side":
            f"{prefix}Pitcher allows loud contact to this side of the plate ({sigma_str})",
        "p_launch_angle_allowed_vs_side":
            f"{prefix}Pitcher gives this side of the plate a friendly launch window ({sigma_str})",
        "p_sweet_spot_pct_allowed_vs_side":
            f"{prefix}Pitcher allows ideal launch angle contact to this side ({sigma_str})",
        "p_pull_air_pct_allowed_vs_side":
            f"{prefix}Pitcher gives this side of the plate HR-friendly pull air ({sigma_str})",
        "p_hr_contact_risk_vs_side":
            f"{prefix}Pitcher is especially vulnerable to this side's HR contact ({sigma_str})",
        "p_lift_damage_risk_vs_side":
            f"{prefix}Pitcher allows lifted damage to this side ({sigma_str})",
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
    "h_hr_rate", "h_launch_angle", "h_sweet_spot_pct",
    "h_hr_contact_score", "h_lifted_power_score",
    "h_hr_rate_vs_hand", "h_xwoba_vs_hand", "h_barrel_pct_vs_hand",
    "h_hard_hit_pct_vs_hand", "h_sweet_spot_pct_vs_hand", "h_pull_air_pct_vs_hand",
    "h_hr_contact_score_vs_hand", "h_lifted_power_score_vs_hand",
    "h_hr_vs_4seam", "h_hr_vs_slider", "h_hr_vs_change", "h_hr_vs_sinker",
    "h_ev_vs_4seam", "h_ev_vs_slider",
    "h_xwoba_vs_4seam", "h_xwoba_vs_slider",
]
# Pitcher features where HIGHER = pitcher is HR-prone (good for hitter)
PITCHER_VULN_FEATS = [
    "p_hr_rate_allowed", "p_barrel_pct_allowed", "p_hard_hit_pct_allowed",
    "p_exit_velo_allowed", "p_launch_angle_allowed", "p_sweet_spot_pct_allowed",
    "p_pull_air_pct_allowed", "p_spin_into_barrel_pct",
    "p_hr_contact_risk", "p_lift_damage_risk",
    "p_hr_rate_allowed_vs_side", "p_barrel_pct_allowed_vs_side", "p_hard_hit_pct_allowed_vs_side",
    "p_exit_velo_allowed_vs_side", "p_launch_angle_allowed_vs_side",
    "p_sweet_spot_pct_allowed_vs_side", "p_pull_air_pct_allowed_vs_side",
    "p_hr_contact_risk_vs_side", "p_lift_damage_risk_vs_side",
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

def _safe_float(val):
    try:
        if val is None or pd.isna(val):
            return None
        return float(val)
    except Exception:
        return None


def _blend_form_value(row, base_col, season_col=None, recent_col=None, season_n_col=None, recent_n_col=None,
                      season_scale=120.0, recent_scale=45.0, season_weight=0.90, recent_weight=1.35):
    base = _safe_float(row.get(base_col))
    if base is None:
        return None
    season = _safe_float(row.get(season_col or f"{base_col}_2026"))
    recent = _safe_float(row.get(recent_col or f"{base_col}_recent"))
    season_n = _safe_float(row.get(season_n_col)) if season_n_col else None
    recent_n = _safe_float(row.get(recent_n_col)) if recent_n_col else None
    season_conf = min(1.0, max(0.0, (season_n or 0.0) / season_scale)) if season is not None else 0.0
    recent_conf = min(1.0, max(0.0, (recent_n or 0.0) / recent_scale)) if recent is not None else 0.0
    sw = season_weight * season_conf if season is not None else 0.0
    rw = recent_weight * recent_conf if recent is not None else 0.0
    base_weight = max(0.30, 1.0 - 0.45 * season_conf - 0.60 * recent_conf)
    total = base_weight + sw + rw
    return ((base or 0.0) * base_weight + (season or 0.0) * sw + (recent or 0.0) * rw) / total

def _pitch_label_from_feat(feat):
    prefixes = ("h_hr_vs_", "h_ev_vs_", "h_xwoba_vs_")
    for prefix in prefixes:
        if feat.startswith(prefix):
            return feat[len(prefix):]
    return None

def _pitch_sample_weight(hr, label):
    pa = _safe_float(hr.get(f"h_pa_vs_{label}"))
    if pa is None:
        return 0.35
    return min(1.0, max(0.0, (pa - 12.0) / 45.0))

def _shrunk_pitch_value(hr, feat):
    label = _pitch_label_from_feat(feat)
    if not label:
        return _safe_float(hr.get(feat))

    raw = _safe_float(hr.get(feat))
    if raw is None:
        return None

    weight = _pitch_sample_weight(hr, label)
    if feat.startswith("h_hr_vs_"):
        baseline = _safe_float(hr.get("h_hr_rate")) or 0.034
    elif feat.startswith("h_ev_vs_"):
        baseline = _safe_float(hr.get("h_exit_velo")) or 88.0
    else:
        baseline = 0.320
    return weight * raw + (1.0 - weight) * baseline

def _batter_hand_info(h_row):
    if len(h_row) == 0:
        return 0, "lhh"
    batter_right = _safe_float(h_row.iloc[0].get("h_batter_right"))
    batter_right = int(round(batter_right)) if batter_right is not None else 0
    return batter_right, ("rhh" if batter_right == 1 else "lhh")

def _hand_sample_weight(hr, pitcher_hand):
    label = "rhp" if pitcher_hand == "R" else "lhp"
    n = _safe_float(hr.get(f"h_n_batted_vs_{label}"))
    if n is None:
        return 0.6
    return min(1.0, max(0.0, (n - 10.0) / 90.0))

def _pitcher_sample_weight(pr, side_label=None):
    n_col = f"p_n_faced_{side_label}" if side_label else "p_n_faced"
    n = _safe_float(pr.get(n_col))
    if n is None:
        return 0.35
    return min(1.0, max(0.0, (n - 25.0) / 220.0))

def _shrunk_pitcher_value(pr, feat, side_label=None):
    raw = _safe_float(pr.get(feat))
    if raw is None:
        return None

    weight = _pitcher_sample_weight(pr, side_label)
    baseline = None
    if feat in pitcher_pop:
        baseline = pitcher_pop[feat][0]
    elif side_label and feat.endswith(f"_{side_label}"):
        overall_feat = feat[: -(len(side_label) + 1)]
        baseline = _safe_float(pr.get(overall_feat))
        if baseline is None and overall_feat in pitcher_pop:
            baseline = pitcher_pop[overall_feat][0]
    if baseline is None:
        fallback = {
            "p_hr_rate_allowed": 0.034,
            "p_exit_velo_allowed": 88.0,
            "p_barrel_pct_allowed": 0.07,
            "p_hard_hit_pct_allowed": 0.34,
            "p_launch_angle_allowed": 14.0,
            "p_sweet_spot_pct_allowed": 0.32,
            "p_pull_air_pct_allowed": 0.18,
            "p_hr_contact_risk": 0.18,
            "p_lift_damage_risk": 0.16,
        }
        overall_feat = feat[: -(len(side_label) + 1)] if side_label and feat.endswith(f"_{side_label}") else feat
        baseline = fallback.get(overall_feat, raw)
    return weight * raw + (1.0 - weight) * baseline

def _resolve_pitcher_row(pitcher_name, allow_fuzzy=False):
    if not pitcher_name or pitcher_name == "TBD":
        return pd.DataFrame()
    exact = pitcher[pitcher["player_name"] == to_statcast_name(pitcher_name)]
    if len(exact) > 0:
        return exact.head(1)
    if not allow_fuzzy:
        return pd.DataFrame()
    last = pitcher_name.split()[-1].lower()
    return pitcher[pitcher["player_name"].str.lower().str.contains(last, na=False)].head(1)

def _derive_matchup_feature(feat, hr, pr, batter_side_suffix, pitcher_hand="R"):
    hand_label = "rhp" if pitcher_hand == "R" else "lhp"
    side_label = batter_side_suffix

    if feat.startswith("h_form_"):
        metric = feat[len("h_form_"):]
        return _blend_form_value(
            hr,
            f"h_{metric}",
            season_col=f"h_2026_{metric}",
            recent_col=f"h_recent_{metric}",
            season_n_col="h_2026_n_batted",
            recent_n_col="h_recent_n_batted",
        )
    if feat.startswith("p_form_"):
        metric = feat[len("p_form_"):]
        return _blend_form_value(
            pr,
            f"p_{metric}",
            season_col=f"p_2026_{metric}",
            recent_col=f"p_recent_{metric}",
            season_n_col="p_2026_n_faced",
            recent_n_col="p_recent_n_faced",
            season_scale=180.0,
            recent_scale=70.0,
        )

    if feat in {
        "h_hr_rate_vs_hand", "h_xwoba_vs_hand", "h_barrel_pct_vs_hand", "h_hard_hit_pct_vs_hand",
        "h_launch_angle_vs_hand", "h_sweet_spot_pct_vs_hand", "h_pull_air_pct_vs_hand",
        "h_woba_vs_hand", "h_iso_vs_hand", "h_k_pct_vs_hand", "h_hr_pa_vs_hand",
        "h_hr_contact_score_vs_hand", "h_lifted_power_score_vs_hand",
    }:
        field_map = {
            "h_hr_rate_vs_hand": f"h_hr_vs_{hand_label}",
            "h_hr_pa_vs_hand": f"h_hr_pa_vs_{hand_label}",
            "h_woba_vs_hand": f"h_woba_vs_{hand_label}",
            "h_iso_vs_hand": f"h_iso_vs_{hand_label}",
            "h_k_pct_vs_hand": f"h_k_pct_vs_{hand_label}",
            "h_xwoba_vs_hand": f"h_xwoba_vs_{hand_label}",
            "h_barrel_pct_vs_hand": f"h_barrel_pct_vs_{hand_label}",
            "h_hard_hit_pct_vs_hand": f"h_hard_hit_pct_vs_{hand_label}",
            "h_launch_angle_vs_hand": f"h_launch_angle_vs_{hand_label}",
            "h_sweet_spot_pct_vs_hand": f"h_sweet_spot_pct_vs_{hand_label}",
            "h_pull_air_pct_vs_hand": f"h_pull_air_pct_vs_{hand_label}",
            "h_hr_contact_score_vs_hand": f"h_hr_contact_score_vs_{hand_label}",
            "h_lifted_power_score_vs_hand": f"h_lifted_power_score_vs_{hand_label}",
        }
        raw = _safe_float(hr.get(field_map[feat]))
        if raw is None:
            return None
        baseline_map = {
            "h_hr_rate_vs_hand": _safe_float(hr.get("h_hr_rate")) or 0.034,
            "h_hr_pa_vs_hand": _safe_float(hr.get("h_hr_rate")) or 0.034,
            "h_woba_vs_hand": 0.320,
            "h_iso_vs_hand": 0.160,
            "h_k_pct_vs_hand": 0.220,
            "h_xwoba_vs_hand": _safe_float(hr.get("estimated_woba_using_speedangle")) or 0.320,
            "h_barrel_pct_vs_hand": _safe_float(hr.get("h_barrel_pct")) or 0.0,
            "h_hard_hit_pct_vs_hand": _safe_float(hr.get("h_hard_hit_pct")) or 0.0,
            "h_launch_angle_vs_hand": _safe_float(hr.get("h_launch_angle")) or 12.0,
            "h_sweet_spot_pct_vs_hand": _safe_float(hr.get("h_sweet_spot_pct")) or 0.0,
            "h_pull_air_pct_vs_hand": _safe_float(hr.get("h_pull_air_pct")) or 0.0,
            "h_hr_contact_score_vs_hand": _safe_float(hr.get("h_hr_contact_score")) or 0.0,
            "h_lifted_power_score_vs_hand": _safe_float(hr.get("h_lifted_power_score")) or 0.0,
        }
        weight = _hand_sample_weight(hr, pitcher_hand)
        return weight * raw + (1.0 - weight) * baseline_map[feat]

    if feat in {"m_bvp_woba", "m_bvp_xwoba", "m_bvp_barrel_pct", "m_bvp_hard_hit_pct", "m_bvp_hr_pa", "m_bvp_weight"}:
        if BVP_HISTORY.empty:
            return None
        subset = BVP_HISTORY[
            (BVP_HISTORY["batter"] == hr.get("batter", hr.get("batter_id", hr.get("batter")))) &
            (BVP_HISTORY["player_name"] == to_statcast_name(pr.get("player_name", "")))
        ]
        if subset.empty:
            return 0.0 if feat == "m_bvp_weight" else None
        row = subset.iloc[0]
        pa = _safe_float(row.get("bvp_ab")) or 0.0
        if feat == "m_bvp_weight":
            if pa >= 15:
                return 0.20
            if pa >= 8:
                return 0.10
            return 0.0
        field_map = {
            "m_bvp_woba": "bvp_avg",
            "m_bvp_xwoba": "bvp_xwoba",
            "m_bvp_barrel_pct": "bvp_barrel_pct",
            "m_bvp_hard_hit_pct": "bvp_hard_hit_pct",
            "m_bvp_hr_pa": "bvp_hr",
        }
        val = _safe_float(row.get(field_map[feat]))
        if feat == "m_bvp_hr_pa":
            return (val / pa) if val is not None and pa > 0 else None
        return val

    if feat in {
        "p_hr_rate_allowed_vs_side", "p_barrel_pct_allowed_vs_side", "p_hard_hit_pct_allowed_vs_side",
        "p_exit_velo_allowed_vs_side", "p_launch_angle_allowed_vs_side",
        "p_sweet_spot_pct_allowed_vs_side", "p_pull_air_pct_allowed_vs_side",
        "p_hr_contact_risk_vs_side", "p_lift_damage_risk_vs_side",
    }:
        field_map = {
            "p_hr_rate_allowed_vs_side": f"p_hr_rate_allowed_{side_label}",
            "p_barrel_pct_allowed_vs_side": f"p_barrel_pct_allowed_{side_label}",
            "p_hard_hit_pct_allowed_vs_side": f"p_hard_hit_pct_allowed_{side_label}",
            "p_exit_velo_allowed_vs_side": f"p_exit_velo_allowed_{side_label}",
            "p_launch_angle_allowed_vs_side": f"p_launch_angle_allowed_{side_label}",
            "p_sweet_spot_pct_allowed_vs_side": f"p_sweet_spot_pct_allowed_{side_label}",
            "p_pull_air_pct_allowed_vs_side": f"p_pull_air_pct_allowed_{side_label}",
            "p_hr_contact_risk_vs_side": f"p_hr_contact_risk_{side_label}",
            "p_lift_damage_risk_vs_side": f"p_lift_damage_risk_{side_label}",
        }
        return _shrunk_pitcher_value(pr, field_map[feat], side_label)

    if feat == "h_hr_contact_score":
        barrel = _safe_float(hr.get("h_barrel_pct"))
        sweet = _safe_float(hr.get("h_sweet_spot_pct"))
        hard_hit = _safe_float(hr.get("h_hard_hit_pct"))
        pull_air = _safe_float(hr.get("h_pull_air_pct"))
        exit_velo = _safe_float(hr.get("h_exit_velo"))
        if None in (barrel, sweet, hard_hit, pull_air, exit_velo):
            return None
        ev_bonus = max(0.0, (exit_velo - 88.0) / 8.0)
        return barrel * 0.35 + sweet * 0.25 + hard_hit * 0.20 + pull_air * 0.20 + ev_bonus * 0.10
    if feat == "h_lifted_power_score":
        barrel = _safe_float(hr.get("h_barrel_pct"))
        sweet = _safe_float(hr.get("h_sweet_spot_pct"))
        pull_air = _safe_float(hr.get("h_pull_air_pct"))
        launch_angle = _safe_float(hr.get("h_launch_angle"))
        if None in (barrel, sweet, pull_air, launch_angle):
            return None
        launch_window = max(0.0, (16.0 - abs(launch_angle - 18.0)) / 16.0)
        return barrel * 0.45 + sweet * 0.30 + pull_air * 0.15 + launch_window * 0.10
    if feat == "p_hr_contact_risk":
        barrel = _shrunk_pitcher_value(pr, "p_barrel_pct_allowed")
        hard_hit = _shrunk_pitcher_value(pr, "p_hard_hit_pct_allowed")
        sweet = _shrunk_pitcher_value(pr, "p_sweet_spot_pct_allowed")
        pull_air = _shrunk_pitcher_value(pr, "p_pull_air_pct_allowed")
        hr_rate = _shrunk_pitcher_value(pr, "p_hr_rate_allowed")
        ev_allowed = _shrunk_pitcher_value(pr, "p_exit_velo_allowed")
        if None in (barrel, hard_hit, sweet, pull_air, hr_rate, ev_allowed):
            return None
        ev_risk = max(0.0, (ev_allowed - 88.0) / 8.0)
        return barrel * 0.30 + hard_hit * 0.20 + sweet * 0.20 + pull_air * 0.15 + hr_rate * 0.15 + ev_risk * 0.10
    if feat == "p_lift_damage_risk":
        sweet = _shrunk_pitcher_value(pr, "p_sweet_spot_pct_allowed")
        pull_air = _shrunk_pitcher_value(pr, "p_pull_air_pct_allowed")
        launch_angle = _shrunk_pitcher_value(pr, "p_launch_angle_allowed")
        barrel = _shrunk_pitcher_value(pr, "p_barrel_pct_allowed")
        if None in (sweet, pull_air, launch_angle, barrel):
            return None
        launch_window = max(0.0, (16.0 - abs(launch_angle - 18.0)) / 16.0)
        return sweet * 0.35 + pull_air * 0.25 + launch_window * 0.20 + barrel * 0.20
    if feat == "m_sweet_spot_contact_edge":
        h_val = _safe_float(hr.get("h_sweet_spot_pct"))
        p_val = _safe_float(pr.get("p_sweet_spot_pct_allowed"))
        return None if h_val is None or p_val is None else h_val * p_val
    if feat == "m_zone_attack_edge":
        h_val = _safe_float(hr.get("h_zone_contact_pct"))
        p_val = _safe_float(pr.get("p_in_zone_pct"))
        return None if h_val is None or p_val is None else h_val * p_val
    if feat == "m_barrel_matchup_score":
        h_val = _safe_float(hr.get("h_barrel_pct"))
        p_val = _safe_float(pr.get("p_barrel_pct_allowed"))
        return None if h_val is None or p_val is None else h_val * p_val
    if feat == "m_lift_matchup_score":
        h_sweet = _safe_float(hr.get("h_sweet_spot_pct"))
        p_sweet = _safe_float(pr.get("p_sweet_spot_pct_allowed"))
        h_pull = _safe_float(hr.get("h_pull_air_pct"))
        p_pull = _safe_float(pr.get("p_pull_air_pct_allowed"))
        if None in (h_sweet, p_sweet, h_pull, p_pull):
            return None
        return (h_sweet * p_sweet + h_pull * p_pull) / 2.0
    if feat == "m_hr_contact_matchup":
        h_score = _derive_matchup_feature("h_hr_contact_score", hr, pr, batter_side_suffix, pitcher_hand)
        p_score = _derive_matchup_feature("p_hr_contact_risk", hr, pr, batter_side_suffix, pitcher_hand)
        return None if h_score is None or p_score is None else h_score * p_score
    if feat == "m_lifted_power_matchup":
        h_score = _derive_matchup_feature("h_lifted_power_score", hr, pr, batter_side_suffix, pitcher_hand)
        p_score = _derive_matchup_feature("p_lift_damage_risk", hr, pr, batter_side_suffix, pitcher_hand)
        return None if h_score is None or p_score is None else h_score * p_score
    if feat == "m_handed_hr_matchup":
        h_score = _derive_matchup_feature("h_hr_rate_vs_hand", hr, pr, batter_side_suffix, pitcher_hand)
        p_score = _derive_matchup_feature("p_hr_rate_allowed_vs_side", hr, pr, batter_side_suffix, pitcher_hand)
        return None if h_score is None or p_score is None else h_score * p_score
    if feat == "m_handed_contact_matchup":
        h_score = _derive_matchup_feature("h_hr_contact_score_vs_hand", hr, pr, batter_side_suffix, pitcher_hand)
        p_score = _derive_matchup_feature("p_hr_contact_risk_vs_side", hr, pr, batter_side_suffix, pitcher_hand)
        return None if h_score is None or p_score is None else h_score * p_score
    if feat == "m_handed_lift_matchup":
        h_score = _derive_matchup_feature("h_lifted_power_score_vs_hand", hr, pr, batter_side_suffix, pitcher_hand)
        p_score = _derive_matchup_feature("p_lift_damage_risk_vs_side", hr, pr, batter_side_suffix, pitcher_hand)
        return None if h_score is None or p_score is None else h_score * p_score
    if feat.startswith("p_") and feat.endswith("_usage_matchup"):
        label = feat[len("p_"):-len("_usage_matchup")]
        return _safe_float(pr.get(f"p_{label}_usage_{batter_side_suffix}"))
    if feat.startswith("m_") and feat.endswith("_hr_exposure"):
        label = feat[len("m_"):-len("_hr_exposure")]
        h_val = _shrunk_pitch_value(hr, f"h_hr_vs_{label}")
        usage = _safe_float(pr.get(f"p_{label}_usage_{batter_side_suffix}"))
        return None if h_val is None or usage is None else h_val * usage
    if feat.startswith("m_") and feat.endswith("_xwoba_exposure"):
        label = feat[len("m_"):-len("_xwoba_exposure")]
        h_val = _shrunk_pitch_value(hr, f"h_xwoba_vs_{label}")
        usage = _safe_float(pr.get(f"p_{label}_usage_{batter_side_suffix}"))
        return None if h_val is None or usage is None else h_val * usage
    if feat.startswith("m_") and feat.endswith("_ev_delta"):
        label = feat[len("m_"):-len("_ev_delta")]
        h_val = _shrunk_pitch_value(hr, f"h_ev_vs_{label}")
        p_val = _safe_float(pr.get(f"p_ev_allowed_{label}"))
        return None if h_val is None or p_val is None else h_val - p_val
    if feat == "m_pitch_hr_matchup":
        vals = []
        for label in ["4seam", "sinker", "slider", "change", "curve", "cutter"]:
            h_val = _shrunk_pitch_value(hr, f"h_hr_vs_{label}")
            usage = _safe_float(pr.get(f"p_{label}_usage_{batter_side_suffix}"))
            if h_val is not None and usage is not None:
                vals.append(h_val * usage)
        return sum(vals) if vals else None
    if feat == "m_pitch_contact_matchup":
        vals = []
        for label in ["4seam", "sinker", "slider", "change", "curve", "cutter"]:
            h_val = _shrunk_pitch_value(hr, f"h_xwoba_vs_{label}")
            usage = _safe_float(pr.get(f"p_{label}_usage_{batter_side_suffix}"))
            if h_val is not None and usage is not None:
                vals.append(h_val * usage)
        return sum(vals) if vals else None
    if feat == "m_pitch_ev_matchup":
        vals = []
        for label in ["4seam", "sinker", "slider", "change", "curve", "cutter"]:
            h_val = _shrunk_pitch_value(hr, f"h_ev_vs_{label}")
            p_val = _safe_float(pr.get(f"p_ev_allowed_{label}"))
            usage = _safe_float(pr.get(f"p_{label}_usage_{batter_side_suffix}"))
            if h_val is not None and p_val is not None and usage is not None:
                vals.append((h_val - p_val) * usage)
        return sum(vals) if vals else None
    return None

def _matchup_reasons(hr, pr, batter_side_suffix):
    reasons = []
    pitch_labels = ["4seam", "slider", "change", "sinker", "curve", "cutter"]
    positive = []
    negative = []
    base_hr = _safe_float(hr.get("h_hr_rate")) or 0.0
    for label in pitch_labels:
        usage = _safe_float(pr.get(f"p_{label}_usage_{batter_side_suffix}"))
        if usage is None or usage < 0.14:
            continue
        sample_weight = _pitch_sample_weight(hr, label)
        if sample_weight < 0.35:
            continue
        hr_vs = _shrunk_pitch_value(hr, f"h_hr_vs_{label}")
        xwoba = _shrunk_pitch_value(hr, f"h_xwoba_vs_{label}")
        ev_vs = _shrunk_pitch_value(hr, f"h_ev_vs_{label}")
        ev_allowed = _safe_float(pr.get(f"p_ev_allowed_{label}"))
        score = 0.0
        if hr_vs is not None and base_hr > 0:
            score += usage * ((hr_vs / base_hr) - 1.0)
        if xwoba is not None:
            score += usage * max(0.0, xwoba - 0.320) * 3.0
        if ev_vs is not None and ev_allowed is not None:
            score += usage * ((ev_vs - ev_allowed) / 8.0)
        if score >= 0.12:
            positive.append((score, label, usage, hr_vs, xwoba, ev_vs))
        elif score <= -0.08:
            negative.append((score, label, usage, hr_vs, xwoba, ev_vs))

    positive.sort(reverse=True)
    negative.sort()

    if positive:
        _, label, usage, hr_vs, xwoba, ev_vs = positive[0]
        if hr_vs is not None and base_hr > 0 and hr_vs > base_hr * 1.25:
            reasons.append(f"⚡ Crushes {label}s, and this pitcher leans on them {usage*100:.0f}% of the time")
        elif xwoba is not None and xwoba >= 0.380:
            reasons.append(f"⚡ Strong contact quality vs {label}s, and he should see that pitch often")
        elif ev_vs is not None:
            reasons.append(f"⚡ Above-average damage profile vs {label}s in this matchup")

    if negative:
        _, label, usage, hr_vs, xwoba, ev_vs = negative[0]
        if hr_vs is not None and base_hr > 0 and hr_vs < base_hr * 0.75:
            reasons.append(f"❄️ Weaker-than-usual HR profile vs {label}s, and this pitcher uses them {usage*100:.0f}% of the time")
        elif xwoba is not None and xwoba < 0.300:
            reasons.append(f"❄️ Contact quality vs {label}s has been below average")

    sweet = _safe_float(hr.get("h_sweet_spot_pct"))
    sweet_allowed = _safe_float(pr.get("p_sweet_spot_pct_allowed"))
    if sweet is not None and sweet_allowed is not None and sweet >= 0.34 and sweet_allowed >= 0.34:
        reasons.append("Sweet-spot launch profile matches a pitcher who gives up ideal HR launch angle contact")

    h_contact = _derive_matchup_feature("h_hr_contact_score", hr, pr, batter_side_suffix)
    p_contact = _derive_matchup_feature("p_hr_contact_risk", hr, pr, batter_side_suffix)
    if h_contact is not None and p_contact is not None and h_contact >= 0.22 and p_contact >= 0.18:
        reasons.append("⚡ Hitter's power-contact shape lines up with a pitcher who allows HR-quality contact")

    return reasons[:2]

def _power_foundation_reasons(hr):
    reasons = []
    hr_contact = _safe_float(hr.get("h_hr_contact_score"))
    lifted = _safe_float(hr.get("h_lifted_power_score"))
    barrel = _safe_float(hr.get("h_barrel_pct"))
    hard_hit = _safe_float(hr.get("h_hard_hit_pct"))
    pull_air = _safe_float(hr.get("h_pull_air_pct"))
    launch_angle = _safe_float(hr.get("h_launch_angle"))

    if hr_contact is not None and hr_contact >= 0.24:
        reasons.append("⚡ Power-contact foundation is strong enough to support real HR upside")
    elif lifted is not None and lifted >= 0.20:
        reasons.append("⚡ Lift and contact shape are both working in a homer-friendly direction")

    if barrel is not None and hard_hit is not None and barrel >= 0.10 and hard_hit >= 0.40:
        reasons.append(f"⚡ Barrel {barrel*100:.1f}% and hard-hit {hard_hit*100:.1f}% give him real game power")
    elif pull_air is not None and launch_angle is not None and pull_air >= 0.20 and 12 <= launch_angle <= 22:
        reasons.append("⚡ Pull-air profile and launch angle both fit a homer path")

    return reasons

def _handedness_reasons(hr, pr, pitcher_hand, batter_side_suffix):
    reasons = []
    hand_label = "LHP" if pitcher_hand == "L" else "RHP"
    hitter_hr = _derive_matchup_feature("h_hr_rate_vs_hand", hr, pr, batter_side_suffix, pitcher_hand)
    hitter_xwoba = _derive_matchup_feature("h_xwoba_vs_hand", hr, pr, batter_side_suffix, pitcher_hand)
    hitter_contact = _derive_matchup_feature("h_hr_contact_score_vs_hand", hr, pr, batter_side_suffix, pitcher_hand)
    pitcher_hr = _derive_matchup_feature("p_hr_rate_allowed_vs_side", hr, pr, batter_side_suffix, pitcher_hand)
    pitcher_contact = _derive_matchup_feature("p_hr_contact_risk_vs_side", hr, pr, batter_side_suffix, pitcher_hand)
    base_hr = _safe_float(hr.get("h_hr_rate")) or 0.034
    base_xwoba = 0.320
    base_pitcher_hr = _safe_float(pr.get("p_hr_rate_allowed")) or 0.034

    if hitter_hr is not None and pitcher_hr is not None:
        if hitter_hr >= base_hr * 1.20 and pitcher_hr >= base_pitcher_hr * 1.12:
            reasons.append(f"⚡ Handedness is a real plus: he hits {hand_label} well and this pitcher is weaker to his side")
        elif hitter_hr <= base_hr * 0.82 and pitcher_hr <= base_pitcher_hr * 0.92:
            reasons.append(f"❄️ Handedness is a drag: this is not his best side and the pitcher is tougher here")

    if not reasons and hitter_xwoba is not None and hitter_contact is not None:
        if hitter_xwoba >= base_xwoba + 0.035 and hitter_contact >= (_safe_float(hr.get("h_hr_contact_score")) or 0) * 1.05:
            reasons.append(f"⚡ Contact quality vs {hand_label} is clearly stronger than his baseline")

    if not reasons and pitcher_contact is not None:
        base_pitcher_contact = _safe_float(pr.get("p_hr_contact_risk")) or 0.0
        if pitcher_contact >= base_pitcher_contact * 1.08:
            reasons.append("⚡ Pitcher is especially vulnerable to this side of the plate")

    return reasons

def _bvp_reason(batter_id, pitcher_name):
    if BVP_HISTORY.empty or not pitcher_name or pitcher_name == "TBD":
        return None
    target = to_statcast_name(pitcher_name)
    subset = BVP_HISTORY[
        (BVP_HISTORY["batter"] == batter_id) &
        (BVP_HISTORY["player_name"] == target)
    ]
    if subset.empty:
        last = pitcher_name.split()[-1].lower()
        subset = BVP_HISTORY[
            (BVP_HISTORY["batter"] == batter_id) &
            (BVP_HISTORY["player_name"].str.lower().str.contains(last, na=False))
        ].head(1)
    if subset.empty:
        return None
    row = subset.iloc[0]
    ab = int(row.get("bvp_ab", 0) or 0)
    hits = int(row.get("bvp_hits", 0) or 0)
    hr = int(row.get("bvp_hr", 0) or 0)
    # Tiny BvP samples create more noise than signal.
    # Default to 5+ AB, but allow a small-sample exception only for truly loud HR history.
    if ab < 5 and not (ab >= 3 and hr >= 2):
        return None
    if hr > 0:
        return f"BvP: {hits}-for-{ab} vs {pitcher_name} with {hr} HR"
    return f"BvP: {hits}-for-{ab} vs {pitcher_name}"


def _bvp_snapshot(batter_id, pitcher_name):
    if BVP_HISTORY.empty or not pitcher_name or pitcher_name == "TBD":
        return {}
    target = to_statcast_name(pitcher_name)
    subset = BVP_HISTORY[
        (BVP_HISTORY["batter"] == batter_id) &
        (BVP_HISTORY["player_name"] == target)
    ]
    if subset.empty:
        last = pitcher_name.split()[-1].lower()
        subset = BVP_HISTORY[
            (BVP_HISTORY["batter"] == batter_id) &
            (BVP_HISTORY["player_name"].str.lower().str.contains(last, na=False))
        ].head(1)
    if subset.empty:
        return {}
    row = subset.iloc[0]
    ab = _safe_float(row.get("bvp_ab")) or 0.0
    hr = _safe_float(row.get("bvp_hr")) or 0.0
    return {
        "m_bvp_woba": _safe_float(row.get("bvp_avg")),
        "m_bvp_xwoba": _safe_float(row.get("bvp_xwoba")),
        "m_bvp_barrel_pct": _safe_float(row.get("bvp_barrel_pct")),
        "m_bvp_hard_hit_pct": _safe_float(row.get("bvp_hard_hit_pct")),
        "m_bvp_hr_pa": (hr / ab) if ab > 0 else None,
        "m_bvp_weight": 0.20 if ab >= 15 else 0.10 if ab >= 8 else 0.0,
    }

def _platoon_note(h_row, pitcher_hand):
    if len(h_row) == 0 or pitcher_hand not in {"R", "L"}:
        return None, False
    hr = h_row.iloc[0]
    vs_r = _safe_float(hr.get("h_hr_vs_rhp"))
    vs_l = _safe_float(hr.get("h_hr_vs_lhp"))
    base = _safe_float(hr.get("h_hr_rate"))
    if vs_r is None or vs_l is None or base is None or min(vs_r, vs_l) <= 0:
        return None, False

    stronger_side = "LHP" if vs_l > vs_r else "RHP"
    today_side = "LHP" if pitcher_hand == "L" else "RHP"
    stronger_split = max(vs_r, vs_l)
    weaker_split = min(vs_r, vs_l)
    if stronger_split < weaker_split * 1.25 and stronger_split < base * 1.20:
        return None, False

    on_stronger_side = today_side == stronger_side
    if on_stronger_side:
        note = f"Platoon bat: much stronger vs {stronger_side}, and today is the favorable side"
    else:
        note = f"Platoon bat: much stronger vs {stronger_side}, but today is the weaker side"
    return note, on_stronger_side

def _reason_priority(reason):
    txt = (reason or "").lower()
    high_signal = [
        "sweet-spot", "sweet spot", "barrel", "hr rate vs", "xwoba", "crushes",
        "pitcher allows", "spin hits barrel zone", "launch profile matches"
    ]
    medium_signal = [
        "hard contact", "hard hit", "pull-side fly", "pull fly", "weather",
        "bullpen", "contact quality"
    ]
    low_signal = ["exit velocity", "avg exit velocity", "vs 4-seam", "vs slider"]
    for needle in high_signal:
        if needle in txt:
            return 0
    for needle in medium_signal:
        if needle in txt:
            return 1
    for needle in low_signal:
        if needle in txt:
            return 3
    return 2
 
def predict_with_reasons(batter_id, pitcher_name, home_team, pitcher_hand="R", opp_team=None):
    h_row = hitter[hitter["batter"] == batter_id]
    p_row = _resolve_pitcher_row(pitcher_name, allow_fuzzy=False)

    pitcher_found = len(p_row) > 0
    batter_right, batter_side_suffix = _batter_hand_info(h_row)
 
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
            if _pitch_label_from_feat(feat):
                val = _shrunk_pitch_value(hr, feat)
            else:
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
            val = _shrunk_pitcher_value(pr, feat)
            if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
                continue
            mean, std = pitcher_pop[feat]
            zscores[feat] = ((float(val) - mean) / std, float(val), "pitcher")
 
    # ── Weather ────────────────────────────────────────────────
    wx = get_forecast(home_team)
    weather_stack = compute_weather_stack(home_team, wx)
    wx_adj = weather_stack["modifier"]
    temp = weather_stack["temp_f"]
    wind = weather_stack["wind_speed"]
 
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
                    if _pitch_label_from_feat(feat):
                        v = _shrunk_pitch_value(hr, feat)
                    else:
                        v = hr.get(feat)
                    if v is not None and not (isinstance(v, float) and np.isnan(v)):
                        feature_vals[feat] = float(v)
                elif feat.startswith("h_"):
                    derived = _derive_matchup_feature(feat, hr, pd.Series(dtype=float), batter_side_suffix, pitcher_hand)
                    if derived is not None:
                        feature_vals[feat] = derived
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
                elif feat not in feature_vals or pd.isna(feature_vals[feat]):
                    derived = _derive_matchup_feature(feat, hr, pr, batter_side_suffix, pitcher_hand)
                    if derived is not None:
                        feature_vals[feat] = derived

        bvp_snapshot = _bvp_snapshot(batter_id, pitcher_name)
        for feat, value in bvp_snapshot.items():
            if feat in lr_features and value is not None:
                feature_vals[feat] = float(value)

        # Context features
        feature_vals["batter_right"]      = batter_right
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
 
        LEAGUE_AB_RATE = 0.034
        hr = h_row.iloc[0]
        pr = p_row.iloc[0] if pitcher_found else pd.Series(dtype=float)
        n_batted = float(hr["h_n_batted"]) if "h_n_batted" in hitter.columns and not pd.isna(hr.get("h_n_batted")) else 50.0

        def _score_unit(value, center, scale):
            if value is None:
                return 0.0
            return max(-1.75, min(1.75, (float(value) - center) / scale))

        def _ratio_signal(value, baseline, lo=0.72, hi=1.35):
            if value is None or baseline is None or baseline <= 0:
                return 0.0
            ratio = max(lo, min(hi, float(value) / float(baseline)))
            span = (hi - lo) / 2.0
            return max(-1.5, min(1.5, (ratio - 1.0) / span))

        base_rate = _safe_float(hr.get("h_hr_rate")) or LEAGUE_AB_RATE
        form_rate = _derive_matchup_feature("h_form_hr_rate", hr, pr, batter_side_suffix, pitcher_hand) or base_rate
        matched_split = _derive_matchup_feature("h_hr_rate_vs_hand", hr, pr, batter_side_suffix, pitcher_hand) or form_rate
        iso_vs_hand = _derive_matchup_feature("h_iso_vs_hand", hr, pr, batter_side_suffix, pitcher_hand)
        woba_vs_hand = _derive_matchup_feature("h_woba_vs_hand", hr, pr, batter_side_suffix, pitcher_hand)
        hand_contact = _derive_matchup_feature("h_hr_contact_score_vs_hand", hr, pr, batter_side_suffix, pitcher_hand)

        side_pitcher_hr = _derive_matchup_feature("p_hr_rate_allowed_vs_side", hr, pr, batter_side_suffix, pitcher_hand)
        if side_pitcher_hr is None:
            side_pitcher_hr = _derive_matchup_feature("p_form_hr_rate_allowed", hr, pr, batter_side_suffix, pitcher_hand)
        if side_pitcher_hr is None:
            side_pitcher_hr = _safe_float(pr.get("p_hr_rate_allowed")) if pitcher_found else LEAGUE_AB_RATE
        side_pitcher_hr = side_pitcher_hr or LEAGUE_AB_RATE

        form_exit_velo = _derive_matchup_feature("h_form_exit_velo", hr, pr, batter_side_suffix, pitcher_hand) or _safe_float(hr.get("h_exit_velo")) or 88.0
        form_barrel = _derive_matchup_feature("h_form_barrel_pct", hr, pr, batter_side_suffix, pitcher_hand) or _safe_float(hr.get("h_barrel_pct")) or 0.07
        form_hard_hit = _derive_matchup_feature("h_form_hard_hit_pct", hr, pr, batter_side_suffix, pitcher_hand) or _safe_float(hr.get("h_hard_hit_pct")) or 0.34
        form_launch = _derive_matchup_feature("h_form_launch_angle", hr, pr, batter_side_suffix, pitcher_hand) or _safe_float(hr.get("h_launch_angle")) or 14.0
        form_sweet = _derive_matchup_feature("h_form_sweet_spot_pct", hr, pr, batter_side_suffix, pitcher_hand) or _safe_float(hr.get("h_sweet_spot_pct")) or 0.32
        form_pull_air = _derive_matchup_feature("h_form_pull_air_pct", hr, pr, batter_side_suffix, pitcher_hand) or _safe_float(hr.get("h_pull_air_pct")) or 0.18
        form_contact = _derive_matchup_feature("h_form_hr_contact_score", hr, pr, batter_side_suffix, pitcher_hand) or _safe_float(hr.get("h_hr_contact_score")) or 0.18

        pitcher_form_hr = _derive_matchup_feature("p_form_hr_rate_allowed", hr, pr, batter_side_suffix, pitcher_hand) or side_pitcher_hr
        pitcher_form_contact = _derive_matchup_feature("p_form_hr_contact_risk", hr, pr, batter_side_suffix, pitcher_hand) or _safe_float(pr.get("p_hr_contact_risk")) or 0.18
        pitcher_form_lift = _derive_matchup_feature("p_form_lift_damage_risk", hr, pr, batter_side_suffix, pitcher_hand) or _safe_float(pr.get("p_lift_damage_risk")) or 0.16
        pitcher_vulnerability = _safe_float(pr.get("p_hr_vulnerability_score")) or 0.0

        pitch_hr_matchup = _derive_matchup_feature("m_pitch_hr_matchup", hr, pr, batter_side_suffix, pitcher_hand)
        pitch_contact_matchup = _derive_matchup_feature("m_pitch_contact_matchup", hr, pr, batter_side_suffix, pitcher_hand)
        pitch_ev_matchup = _derive_matchup_feature("m_pitch_ev_matchup", hr, pr, batter_side_suffix, pitcher_hand)

        bvp_weight = _safe_float(bvp_snapshot.get("m_bvp_weight")) or 0.0
        bvp_hr_pa = _safe_float(bvp_snapshot.get("m_bvp_hr_pa"))
        bvp_xwoba = _safe_float(bvp_snapshot.get("m_bvp_xwoba"))

        power_score = (
            0.28 * _score_unit(form_barrel, 0.085, 0.035)
            + 0.22 * _score_unit(form_exit_velo, 88.8, 2.8)
            + 0.18 * _score_unit(form_hard_hit, 0.36, 0.08)
            + 0.12 * _score_unit(form_launch, 15.0, 7.0)
            + 0.10 * _score_unit(form_sweet, 0.34, 0.09)
            + 0.10 * _score_unit(form_pull_air, 0.18, 0.08)
        )
        power_score = max(-1.6, min(1.8, power_score))

        hand_score = (
            0.38 * _ratio_signal(matched_split, form_rate)
            + 0.20 * _score_unit(iso_vs_hand, 0.170, 0.070)
            + 0.16 * _score_unit(woba_vs_hand, 0.320, 0.045)
            + 0.12 * _ratio_signal(hand_contact, form_contact if form_contact > 0 else 0.18)
            + 0.14 * _ratio_signal(side_pitcher_hr, LEAGUE_AB_RATE, lo=0.70, hi=1.45)
        )
        hand_score = max(-1.7, min(1.9, hand_score))

        pitcher_score = (
            0.42 * _ratio_signal(pitcher_form_hr, LEAGUE_AB_RATE, lo=0.72, hi=1.45)
            + 0.24 * _score_unit(pitcher_form_contact, 0.18, 0.05)
            + 0.18 * _score_unit(pitcher_form_lift, 0.16, 0.05)
            + 0.16 * _score_unit(pitcher_vulnerability, 0.0, 0.16)
        )
        pitcher_score = max(-1.6, min(1.8, pitcher_score))

        pitch_score = (
            0.45 * _score_unit(pitch_hr_matchup, 0.0, 0.012)
            + 0.35 * _score_unit(pitch_contact_matchup, 0.0, 0.090)
            + 0.20 * _score_unit(pitch_ev_matchup, 0.0, 2.8)
        )
        pitch_score = max(-1.5, min(1.6, pitch_score))

        bullpen_mult = _bullpen_multiplier(opp_team or home_team)
        park_factor = BALLPARK_HR_FACTORS.get(home_team, 1.0)
        context_score = (
            0.42 * _score_unit(wx_adj, 0.0, 0.020)
            + 0.34 * _score_unit(park_factor, 1.0, 0.12)
            + 0.24 * _score_unit(bullpen_mult, 1.0, 0.12)
        )
        context_score = max(-1.3, min(1.4, context_score))

        bvp_score = 0.0
        if bvp_weight > 0 and (bvp_hr_pa is not None or bvp_xwoba is not None):
            bvp_score = (
                0.6 * _score_unit(bvp_hr_pa, LEAGUE_AB_RATE, 0.030)
                + 0.4 * _score_unit(bvp_xwoba, 0.320, 0.060)
            ) * min(0.22, bvp_weight * 0.22)

        season_n_batted = _safe_float(hr.get("h_2026_n_batted")) or 0.0
        recent_n_batted = _safe_float(hr.get("h_recent_n_batted")) or 0.0

        prior_ab = (
            0.30 * form_rate
            + 0.26 * matched_split
            + 0.24 * side_pitcher_hr
            + 0.20 * LEAGUE_AB_RATE
        )
        prior_ab = max(0.012, min(0.095, prior_ab))

        final_score = (
            0.40 * power_score
            + 0.25 * hand_score
            + 0.20 * pitcher_score
            + 0.10 * pitch_score
            + 0.05 * context_score
            + bvp_score
        )
        final_score = max(-1.9, min(2.1, final_score))

        n_pa = 3.9
        form_sample = min(1.0, max(0.0, (n_batted - 50.0) / 220.0))
        score_slope = 0.16 + 0.04 * form_sample
        display_ab = prior_ab * math.exp(final_score * score_slope)

        # Keep the ML model as a light ranking stabilizer, but not the main price setter.
        prob_raw_clamped = max(0.006, min(0.13, prob_raw))
        display_ab = 0.82 * display_ab + 0.18 * prob_raw_clamped

        base_barrel = _safe_float(hr.get("h_barrel_pct")) or 0.0
        base_exit_velo = _safe_float(hr.get("h_exit_velo")) or 0.0
        base_hard_hit = _safe_float(hr.get("h_hard_hit_pct")) or 0.0

        elite_power = (
            n_batted >= 180
            and form_barrel >= 0.09
            and form_exit_velo >= 87.0
            and form_hard_hit >= 0.34
        )
        upper_tier_power = (
            n_batted >= 120
            and (
                (form_barrel >= 0.07 and form_exit_velo >= 86.0 and form_hard_hit >= 0.31)
                or form_contact >= 0.19
            )
        )

        # Slate-wide pricing has still been too conservative. Give credible bats
        # a modest lift before the weak-profile caps kick in.
        calibration_mult = 1.08
        if upper_tier_power:
            calibration_mult = 1.15
        elif n_batted >= 90 and (
            form_barrel >= 0.06
            or form_exit_velo >= 86.0
            or form_hard_hit >= 0.30
        ):
            calibration_mult = 1.11
        if matched_split >= base_rate * 1.10 and side_pitcher_hr >= LEAGUE_AB_RATE * 1.05:
            calibration_mult += 0.03
        display_ab *= calibration_mult

        # Tiny-sample recent heaters should not dominate the board unless there is
        # real longer-sample power underneath.
        early_2026 = season_n_batted < 18
        tiny_recent = recent_n_batted < 8
        weak_baseline = (
            (_safe_float(hr.get("h_barrel_pct")) or 0.0) < 0.075
            and (_safe_float(hr.get("h_exit_velo")) or 0.0) < 86.0
            and (_safe_float(hr.get("h_hard_hit_pct")) or 0.0) < 0.33
        )
        if early_2026 and tiny_recent and weak_baseline and not elite_power:
            display_ab = min(display_ab, 0.055)
        elif (season_n_batted < 30 or recent_n_batted < 12) and weak_baseline and not upper_tier_power:
            display_ab = min(display_ab, 0.065)

        # Hard credibility cap for fringe bats. If the longer-sample profile is
        # weak, matchup noise should not elevate them into star territory.
        fringe_baseline = (
            base_barrel < 0.060
            and base_exit_velo < 85.5
            and base_hard_hit < 0.30
        )
        poor_baseline = (
            base_barrel < 0.075
            and base_exit_velo < 86.5
            and base_hard_hit < 0.33
        )
        if fringe_baseline and season_n_batted < 80 and not elite_power:
            display_ab = min(display_ab, 0.040)
        elif poor_baseline and season_n_batted < 60 and not upper_tier_power:
            display_ab = min(display_ab, 0.050)
        if base_exit_velo < 86.0 and base_hard_hit < 0.28 and not elite_power:
            display_ab = min(display_ab, 0.048)
        if base_exit_velo < 84.5 and base_hard_hit < 0.24 and not upper_tier_power:
            display_ab = min(display_ab, 0.040)
        if base_barrel < 0.05 and base_exit_velo < 84.5 and base_hard_hit < 0.28:
            display_ab = min(display_ab, 0.032)

        # Gentle guardrails only.
        if form_barrel < 0.05 and form_exit_velo < 85.0 and form_hard_hit < 0.30:
            display_ab = min(display_ab, 0.045)
        if n_batted < 120 and form_barrel < 0.07 and form_exit_velo < 86.0:
            display_ab = min(display_ab, 0.050)

        display_ab = max(0.010, min(0.135, display_ab))
        prob = (1.0 - (1.0 - display_ab) ** n_pa) * 100
        if prob > 30.0:
            prob = 30.0 + (prob - 30.0) * 0.50
        prob = max(3.0, min(30.0, prob))

        if early_2026 and tiny_recent and weak_baseline and not elite_power:
            prob = min(prob, 18.0)
        elif (season_n_batted < 30 or recent_n_batted < 12) and weak_baseline and not upper_tier_power:
            prob = min(prob, 21.0)
        if fringe_baseline and season_n_batted < 80 and not elite_power:
            prob = min(prob, 12.0)
        elif poor_baseline and season_n_batted < 60 and not upper_tier_power:
            prob = min(prob, 15.0)
        if base_exit_velo < 86.0 and base_hard_hit < 0.28 and not elite_power:
            prob = min(prob, 14.0)
        if base_exit_velo < 84.5 and base_hard_hit < 0.24 and not upper_tier_power:
            prob = min(prob, 11.0)
        if base_barrel < 0.05 and base_exit_velo < 84.5 and base_hard_hit < 0.28:
            prob = min(prob, 9.0)
 
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
    if len(h_row) > 0:
        hr = h_row.iloc[0]
        for note in _power_foundation_reasons(hr):
            if note not in reasons:
                reasons.append(note)
            if len(reasons) == 2:
                break

        if pitcher_found and len(reasons) < 2:
            pr = p_row.iloc[0]
            for note in _handedness_reasons(hr, pr, pitcher_hand, batter_side_suffix):
                if note not in reasons:
                    reasons.append(note)
                if len(reasons) == 2:
                    break

        if pitcher_found and len(reasons) < 2:
            matchup_notes = _matchup_reasons(h_row.iloc[0], p_row.iloc[0], batter_side_suffix)
            for note in matchup_notes:
                if note not in reasons:
                    reasons.append(note)
                if len(reasons) == 2:
                    break

    if len(reasons) < 2:
        for feat, z, val in pos_scored[:5]:
            text, _, _ = _reason_text(feat, val, z, pitcher_hand)
            if text and text not in reasons:
                reasons.append(text)
            if len(reasons) == 2:
                break

    if len(reasons) < 3:
        bvp_note = _bvp_reason(batter_id, pitcher_name)
        if bvp_note and bvp_note not in reasons:
            reasons.append(bvp_note)

    platoon_note, platoon_favorable = _platoon_note(h_row, pitcher_hand)
    if platoon_note and platoon_note not in reasons:
        reasons.append(("⚡ " if platoon_favorable else "❄️ ") + platoon_note)

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
        if pitcher_found and len(h_row) > 0 and len(reasons) < 3:
            matchup_notes = _matchup_reasons(h_row.iloc[0], p_row.iloc[0], batter_side_suffix)
            for note in matchup_notes:
                if note not in reasons:
                    reasons.append(note)
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
 
    prob *= MODEL_CALIBRATION["factor"]
    prob = max(2.0, min(30.0, prob))
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
def _mlb_games_today():
    """Number of MLB games scheduled today (ET), from the FREE MLB Stats
    API — costs zero Odds API credits. Returns None if the check fails
    (caller should then proceed rather than silently skip a slate)."""
    today = datetime.now(ET).strftime("%Y-%m-%d")
    try:
        data = requests.get(
            f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={today}",
            timeout=10,
        ).json()
        return int(data.get("totalGames", 0))
    except Exception as e:
        print(f"  MLB schedule check failed ({e}) — proceeding with odds fetch.")
        return None


def _read_quota_headers(resp):
    """(used, remaining) from The Odds API response headers, or Nones."""
    def _num(h):
        v = resp.headers.get(h)
        try:
            return int(float(v))
        except (TypeError, ValueError):
            return None
    return _num("x-requests-used"), _num("x-requests-remaining")


def _log_odds_quota(used, remaining, prop_calls, note=""):
    """Print and persist Odds API credit burn so the 20k/month budget is
    visible run over run (odds_quota_log.csv, appended)."""
    print(f"  Odds API quota: used={used if used is not None else '?'} "
          f"remaining={remaining if remaining is not None else '?'} "
          f"(prop calls this run: {prop_calls}{'; ' + note if note else ''})")
    try:
        new_file = not ODDS_QUOTA_LOG.exists()
        with open(ODDS_QUOTA_LOG, "a", newline="") as f:
            w = csv.writer(f)
            if new_file:
                w.writerow(["timestamp_et", "requests_used", "requests_remaining",
                            "prop_calls_this_run", "note"])
            w.writerow([datetime.now(ET).strftime("%Y-%m-%d %H:%M:%S"),
                        used, remaining, prop_calls, note])
    except Exception as e:
        print(f"  (quota log write failed: {e})")


def fetch_odds():
    """Fetch live 0.5+ HR props, preferring DraftKings when available.

    Credit frugality (20,000 credits/month plan; per-event prop calls are
    the expensive endpoint at ~1 credit x regions x markets each):
      1. FREE MLB Stats API schedule check first — no-game day means no
         Odds API calls at all.
      2. ONE region (us), ONE market (batter_home_runs), and only events
         that commence today (ET) and are still pregame.
      3. x-requests-used / x-requests-remaining logged every run (console
         + odds_quota_log.csv).
      4. If remaining credits fall below ODDS_API_MIN_REMAINING, skip or
         stop instead of burning the reserve.

    Then: keep one 0.5+ HR price per player per book and resolve each
    player to DraftKings if posted, otherwise the highest-priority live
    book in BOOK_PRIORITY.
    """
    print("Fetching HR odds from live sportsbooks...")
    now_utc = datetime.now(timezone.utc)
    today_et = datetime.now(ET).date()
    raw_all  = {}   # {ckey: {book_key: {"player", "book_odds", "book_implied", "book"}}}
    seen_books = set()
    books_with_hr_market = set()
    draftkings_seen = False
    quota_used = quota_remaining = None
    prop_calls = 0

    # Free pre-check: skip the paid API entirely on no-game days.
    n_games = _mlb_games_today()
    if n_games == 0:
        print("  No MLB games today (MLB Stats API) — skipping odds fetch, 0 credits used.")
        return {}

    try:
        ev_resp = requests.get(
            f"https://api.the-odds-api.com/v4/sports/baseball_mlb/events"
            f"?apiKey={ODDS_API_KEY}", timeout=15
        )
        quota_used, quota_remaining = _read_quota_headers(ev_resp)
        if ev_resp.status_code != 200:
            print(f"  Events error: {ev_resp.status_code}")
            _log_odds_quota(quota_used, quota_remaining, 0, "events call failed")
            return {}

        if quota_remaining is not None and quota_remaining < ODDS_API_MIN_REMAINING:
            print(f"  Only {quota_remaining} Odds API credits left "
                  f"(< ODDS_API_MIN_REMAINING={ODDS_API_MIN_REMAINING}) — "
                  f"skipping prop pulls to protect the reserve.")
            _log_odds_quota(quota_used, quota_remaining, 0, "below reserve floor, skipped")
            return {}

        events = ev_resp.json()
        if not isinstance(events, list):
            return {}

        # ── Filter to TODAY'S pregame events only (per-event prop calls
        # cost credits; tomorrow's slate can wait until tomorrow) ──────
        pregame = []
        for ev in events:
            ct = ev.get("commence_time", "")
            try:
                commence_dt = datetime.fromisoformat(ct.replace("Z", "+00:00"))
                if commence_dt.astimezone(ET).date() != today_et:
                    continue
                # Allow up to 30 min after scheduled start (slight delay tolerance)
                if commence_dt > now_utc - timedelta(minutes=30):
                    pregame.append(ev)
            except Exception:
                pregame.append(ev)  # unknown time — include anyway
        print(f"  {len(events)} events listed, {len(pregame)} pregame today (ET)")

        for i, event in enumerate(pregame):
            eid = event.get("id", "")
            if not eid:
                continue
            if i > 0:
                time.sleep(1.2)
            if quota_remaining is not None and quota_remaining < ODDS_API_MIN_REMAINING:
                print(f"  Stopping at {prop_calls} prop calls — remaining credits "
                      f"({quota_remaining}) hit the reserve floor.")
                break

            pr = requests.get(
                f"https://api.the-odds-api.com/v4/sports/baseball_mlb/events/{eid}/odds"
                f"?apiKey={ODDS_API_KEY}&regions=us&markets=batter_home_runs"
                f"&oddsFormat=american", timeout=15
            )
            prop_calls += 1
            u, r = _read_quota_headers(pr)
            if u is not None:
                quota_used = u
            if r is not None:
                quota_remaining = r
            if pr.status_code == 429:
                print(f"  Rate limited — {len(raw_all)} players so far.")
                break
            if pr.status_code != 200:
                continue

            for bm in pr.json().get("bookmakers", []):
                book_key = bm.get("key", "")
                if not book_key:
                    continue
                seen_books.add(book_key)
                for market in bm.get("markets", []):
                    if market.get("key") != "batter_home_runs":
                        continue
                    books_with_hr_market.add(book_key)
                    if book_key == "draftkings":
                        draftkings_seen = True
                    for outcome in market.get("outcomes", []):
                        outcome_name = str(outcome.get("name","")).strip().lower()
                        point = outcome.get("point")
                        if outcome_name in ["no", "under"]:
                            continue
                        if outcome_name not in ["yes", "over"] and outcome.get("description"):
                            continue
                        if point not in (None, 0.5):
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
                            if not existing or abs((point or 0.5) - 0.5) < 1e-9:
                                raw_all[ckey][book_key] = {
                                    "player":       player.strip(),
                                    "book_odds":    odds_val,
                                    "book_implied": round(implied, 2),
                                    "book":         book_key,
                                }
                        except:
                            pass
    except Exception as e:
        print(f"  Odds error: {e}")

    _log_odds_quota(quota_used, quota_remaining, prop_calls)

    # ── Resolve one line per player: DraftKings first, then best live fallback ──
    results = {}
    for ckey, books in raw_all.items():
        chosen = None
        for preferred_book in BOOK_PRIORITY:
            if preferred_book in books:
                chosen = books[preferred_book]
                break
        if chosen is None and books:
            chosen = next(iter(books.values()))
        if chosen:
            results[ckey] = {**chosen, "n_books": len(books)}

    if draftkings_seen:
        dk_count = sum(1 for r in results.values() if r.get("book") == "draftkings")
        print(f"  {dk_count} DraftKings HR props selected from {len(results)} total live props")
    elif books_with_hr_market:
        book_list = ", ".join(BOOK_NAMES.get(k, k) for k in sorted(books_with_hr_market, key=book_rank))
        print(f"  DraftKings HR props not present in API response today; using live fallback books: {book_list}")
        print(f"  {len(results)} total live HR props selected")
    elif seen_books:
        print("  No batter_home_runs market returned by the API for today's MLB events.")
    else:
        print("  No sportsbook data returned for today's MLB events.")
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


def _extract_confirmed_lineup(game, side):
    team_block = game.get("teams", {}).get(side, {})
    sources = []
    if isinstance(game.get("lineups"), dict):
        sources.append(game["lineups"].get(side))
    for key in ["lineup", "players", "batters"]:
        if isinstance(team_block.get(key), list):
            sources.append(team_block.get(key))
    lineup = next((src for src in sources if isinstance(src, list) and src), [])
    parsed = []
    for idx, player in enumerate(lineup, start=1):
        if not isinstance(player, dict):
            continue
        person = player.get("person") or player.get("player") or {}
        pid = person.get("id") or player.get("id")
        name = person.get("fullName") or player.get("fullName") or player.get("name")
        raw_order = player.get("battingOrder") or player.get("order") or idx
        try:
            order = int(str(raw_order)[:1])
        except Exception:
            order = idx
        if pid and name:
            parsed.append({"id": pid, "name": name, "order": order})
    parsed.sort(key=lambda x: x.get("order", 99))
    return parsed[:9]
 
def fetch_games():
    today = datetime.now(ET).strftime("%Y-%m-%d")
    print(f"Fetching games for {today}...")
    games = {}
    try:
        sched = requests.get(
            f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={today}"
            f"&hydrate=probablePitcher,team,lineups", timeout=15
        ).json()
        for de in sched.get("dates", []):
            for game in de.get("games", []):
                home    = game["teams"]["home"]["team"].get("abbreviation","")
                away    = game["teams"]["away"]["team"].get("abbreviation","")
                home_id = game["teams"]["home"]["team"]["id"]
                away_id = game["teams"]["away"]["team"]["id"]
                game_pk = game.get("gamePk")
                game_num = game.get("gameNumber")
                doubleheader = str(game.get("doubleHeader", "N")).upper()
                home_p  = game["teams"]["home"].get("probablePitcher", {})
                away_p  = game["teams"]["away"].get("probablePitcher", {})
                try:
                    dt_utc   = datetime.fromisoformat(game.get("gameDate","").replace("Z","+00:00"))
                    time_str = dt_utc.astimezone(ET).strftime("%-I:%M %p ET")
                except:
                    time_str = "TBD"
 
                home_p_hand = get_pitcher_hand(home_p.get("id"))
                away_p_hand = get_pitcher_hand(away_p.get("id"))
                home_lineup = _extract_confirmed_lineup(game, "home")
                away_lineup = _extract_confirmed_lineup(game, "away")
 
                matchup = f"{away}@{home}"
                display_label = matchup
                if doubleheader in {"Y", "S"} and game_num:
                    display_label = f"{matchup} (G{game_num})"
                game_key = str(game_pk) if game_pk else (f"{matchup}-{game_num}" if game_num else matchup)

                games[game_key] = {
                    "key": game_key,
                    "display_label": display_label,
                    "home": home, "away": away,
                    "home_id": home_id, "away_id": away_id,
                    "time": time_str,
                    "home_pitcher":      home_p.get("fullName","TBD"),
                    "away_pitcher":      away_p.get("fullName","TBD"),
                    "home_pitcher_hand": home_p_hand,
                    "away_pitcher_hand": away_p_hand,
                    "home_lineup":       home_lineup,
                    "away_lineup":       away_lineup,
                    "home_lineup_confirmed": bool(home_lineup),
                    "away_lineup_confirmed": bool(away_lineup),
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
        game_weather_stack = compute_weather_stack(home, gdata.get("weather", {}))
        for batters, opp_pitcher, opp_hand, team, opp_team in [
            # Away batters face the HOME team's starter + bullpen
            (
                gdata["away_lineup"] if gdata.get("away_lineup_confirmed") else get_roster(gdata["away_id"]),
                gdata["home_pitcher"], gdata["home_pitcher_hand"], away, home
            ),
            # Home batters face the AWAY team's starter + bullpen
            (
                gdata["home_lineup"] if gdata.get("home_lineup_confirmed") else get_roster(gdata["home_id"]),
                gdata["away_pitcher"], gdata["away_pitcher_hand"], home, away
            ),
        ]:
            for b in batters:
                bid, name = b["id"], b["name"]
                batting_order = int(b.get("order", 0) or 0)
                h_prof    = hitter[hitter["batter"] == bid]
                in_model  = len(h_prof) > 0
                players_tried += 1
                model_prob = None
                reasons    = []
 
                if in_model and opp_pitcher != "TBD":
                    try:
                        model_prob, reasons = predict_with_reasons(bid, opp_pitcher, home, opp_hand, opp_team)
                        # Slot multiplier only applies to the legacy price:
                        # the structural v2 price below already carries the
                        # lineup slot inside E[PA].
                        if model_prob is not None and batting_order and STRUCTURAL_V2 is None:
                            model_prob *= expected_pa_multiplier(batting_order)
                        players_scored += 1
                    except Exception as e:
                        print(f"  Error {name}: {e}")
                    if STRUCTURAL_V2 is not None:
                        # structural_v2 sets the price; predict_with_reasons
                        # above is kept only for its display reasons.
                        try:
                            slot = batting_order if 1 <= batting_order <= 9 else None
                            model_prob = 100.0 * STRUCTURAL_V2.predict_v2(
                                bid,
                                opposing_pitcher=STRUCTURAL_V2.lookup_pitcher_id(opp_pitcher),
                                pitcher_hand=opp_hand, slot=slot, park=home)
                        except Exception as e:
                            print(f"  Structural v2 error {name}: {e}")
 
                od           = match_odds(name)
                book_odds    = od["book_odds"]    if od else None
                book_implied = od["book_implied"] if od else None
                book_name    = od["book"]         if od else None
                n_books      = od.get("n_books", 1) if od else 0

                value, edge, surface_pick = classify_edge(model_prob, book_implied, book_odds)
                trap_flag = detect_trap_flag(name, datetime.now(ET).strftime("%Y-%m-%d"), book_odds)

                platoon_mismatch = False
                platoon_note = None
                platoon_favorable = False
                if len(h_prof):
                    matched_split = _safe_float(h_prof.iloc[0].get("h_hr_vs_rhp" if opp_hand == "R" else "h_hr_vs_lhp"))
                    off_split = _safe_float(h_prof.iloc[0].get("h_hr_vs_lhp" if opp_hand == "R" else "h_hr_vs_rhp"))
                    base_split = _safe_float(h_prof.iloc[0].get("h_hr_rate"))
                    platoon_note, platoon_favorable = _platoon_note(h_prof, opp_hand)
                    if matched_split is not None and off_split is not None and base_split is not None:
                        platoon_mismatch = (
                            matched_split < off_split * 0.78
                            and matched_split < base_split * 0.90
                        )
 
                rec = {
                    "player":       name,
                    "batter_id":    bid,
                    "team":         team,
                    "game":         gkey,
                    "game_label":   gdata.get("display_label", f"{gdata['away']} @ {home}"),
                    "time":         gdata["time"],
                    "batting_order": batting_order,
                    "pitcher":      opp_pitcher,
                    "pitcher_hand": opp_hand,
                    "home_team":    home,
                    "weather_stack_score": game_weather_stack["score"],
                    "book_odds":    book_odds,
                    "book_implied": book_implied,
                    "book_name":    book_name,
                    "n_books":      n_books,
                    "model_prob":   model_prob,
                    "edge":         edge,
                    "value":        value,
                    "surface_pick": surface_pick,
                    "trap_flag":    trap_flag,
                    "reasons":      reasons,
                    "h_n_batted":   float(h_prof.iloc[0].get("h_n_batted", 0)) if len(h_prof) else 0.0,
                    "h_hr_contact_score": float((_derive_matchup_feature("h_form_hr_contact_score", h_prof.iloc[0], pd.Series(dtype=float), "", "R") if len(h_prof) else 0.0) or 0.0),
                    "h_barrel_pct": float((_derive_matchup_feature("h_form_barrel_pct", h_prof.iloc[0], pd.Series(dtype=float), "", "R") if len(h_prof) else 0.0) or 0.0),
                    "h_exit_velo": float((_derive_matchup_feature("h_form_exit_velo", h_prof.iloc[0], pd.Series(dtype=float), "", "R") if len(h_prof) else 0.0) or 0.0),
                    "h_hard_hit_pct": float((_derive_matchup_feature("h_form_hard_hit_pct", h_prof.iloc[0], pd.Series(dtype=float), "", "R") if len(h_prof) else 0.0) or 0.0),
                    "pitcher_found": len(_resolve_pitcher_row(opp_pitcher, allow_fuzzy=False)) > 0,
                    "platoon_mismatch": platoon_mismatch,
                    "platoon_note": platoon_note,
                    "platoon_favorable": platoon_favorable,
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

def update_picks_history(all_preds):
    today = datetime.now(ET).strftime("%Y-%m-%d")
    min_model = max(MIN_TRACK_MODEL_PROB, PROFIT_POLICY.get("min_model_prob", MIN_TRACK_MODEL_PROB))
    min_edge = PROFIT_POLICY.get("min_edge", MIN_TRACK_EDGE)
    max_odds = PROFIT_POLICY.get("max_odds", MAX_VALUE_AMERICAN_ODDS)
    max_daily = max(1, int(PROFIT_POLICY.get("max_picks_per_day", DEFAULT_MAX_TRACKED_PER_DAY)))

    # expected_roi and kelly_fraction are imported from betting_math

    def estimate_bankroll(existing_rows):
        bankroll = BASE_BANKROLL
        for row in existing_rows:
            result = (row.get("result") or "").strip()
            if result not in {"HR", "No HR"}:
                continue
            try:
                bankroll += float(row.get("pnl") or 0.0)
            except Exception:
                pass
        return max(1000.0, bankroll)

    def recommended_stake(model_prob_pct, american_odds, n_books, bankroll):
        f_star = kelly_fraction(model_prob_pct, american_odds)
        if f_star <= 0:
            return 0.0, 0.0
        f = f_star * KELLY_FRACTION
        if (n_books or 0) < 2:
            f *= 0.65
        raw_stake = bankroll * f
        capped = min(raw_stake, bankroll * MAX_STAKE_PCT_BANKROLL, MAX_STAKE)
        if capped < MIN_STAKE:
            return 0.0, f_star
        return round(capped, 2), f_star

    def is_bettable(rec):
        if rec.get("book_odds") is None or rec.get("model_prob") is None or rec.get("edge") is None:
            return False
        if rec.get("model_prob", 0.0) < min_model:
            return False
        if rec.get("edge", -999.0) < min_edge:
            return False
        if rec.get("pitcher_found") is False:
            return False
        if rec.get("platoon_mismatch"):
            return False
        if rec.get("book_implied") is None or rec.get("book_implied", 0.0) < 6.0:
            return False
        if rec.get("book_odds") is not None and rec.get("book_odds") > max_odds:
            return False
        if expected_roi(rec.get("model_prob"), rec.get("book_odds")) <= 0:
            return False
        return True

    candidates = [r for r in all_preds if is_bettable(r)]
    best_by_player = {}
    for rec in candidates:
        pkey = rec.get("batter_id") or rec.get("player")
        prev = best_by_player.get(pkey)
        rec_score = (
            expected_roi(rec.get("model_prob"), rec.get("book_odds")),
            rec.get("edge") or -999,
            rec.get("model_prob") or 0,
        )
        if prev is None:
            best_by_player[pkey] = rec
            continue
        prev_score = (
            expected_roi(prev.get("model_prob"), prev.get("book_odds")),
            prev.get("edge") or -999,
            prev.get("model_prob") or 0,
        )
        if rec_score > prev_score:
            best_by_player[pkey] = rec
    candidates = list(best_by_player.values())

    if not candidates:
        print("No qualified +EV picks available to track today.")
        return
    base_fieldnames = [
        "date", "pick_type", "rank", "player", "batter_id", "team", "game", "pitcher",
        "sportsbook", "book_odds", "model_prob", "book_implied", "edge",
        "expected_roi", "kelly_fraction", "stake", "result", "pnl"
    ]

    rows = []
    fieldnames = list(base_fieldnames)
    if PICKS_HISTORY.exists():
        with open(PICKS_HISTORY, newline="") as f:
            rows = list(csv.DictReader(f))
            if rows:
                for key in rows[0].keys():
                    if key not in fieldnames:
                        fieldnames.append(key)

    bankroll_now = estimate_bankroll(rows)

    by_key = {}
    for row in rows:
        key = (row.get("date"), row.get("pick_type"), row.get("rank", ""))
        by_key[key] = row

    prob_group = sorted(
        candidates,
        key=lambda r: (
            expected_roi(r.get("model_prob"), r.get("book_odds")),
            r.get("edge") or -999,
            r.get("model_prob") or 0,
        ),
        reverse=True,
    )[:max_daily]

    value_group = []
    if ENABLE_VALUE_TRACKING:
        value_group = [
            r for r in candidates
            if r.get("edge") is not None and r.get("edge") >= max(MIN_VALUE_EDGE, min_edge)
            and r.get("book_odds") is not None and r.get("book_odds") <= min(MAX_VALUE_AMERICAN_ODDS, 500)
            and expected_roi(r.get("model_prob"), r.get("book_odds")) > 0.02
        ]
        value_group = sorted(
            value_group,
            key=lambda r: expected_roi(r.get("model_prob"), r.get("book_odds")),
            reverse=True,
        )[:max_daily]

    tracked_groups = [
        ("highest_probability_top10", prob_group),
        ("best_value_top10", value_group),
    ]

    for pick_type, records in tracked_groups:
        written_rank = 0
        for idx, rec in enumerate(records, start=1):
            exp_roi = expected_roi(rec.get("model_prob"), rec.get("book_odds"))
            stake_amt, kelly_f = recommended_stake(
                rec.get("model_prob"),
                rec.get("book_odds"),
                rec.get("n_books", 1),
                bankroll_now,
            )
            if stake_amt <= 0:
                continue

            written_rank += 1
            rank = str(written_rank)
            key = (today, pick_type, rank)
            existing = by_key.get(key, {})
            updated = {
                "date": today,
                "pick_type": pick_type,
                "rank": rank,
                "player": rec["player"],
                "batter_id": rec["batter_id"],
                "team": rec["team"],
                "game": rec["game_label"],
                "pitcher": rec["pitcher"],
                "sportsbook": BOOK_NAMES.get(rec.get("book_name"), rec.get("book_name", "")),
                "book_odds": rec["book_odds"],
                "model_prob": f'{rec["model_prob"]:.2f}' if rec.get("model_prob") is not None else "",
                "book_implied": f'{rec["book_implied"]:.2f}' if rec.get("book_implied") is not None else "",
                "edge": f'{rec["edge"]:.2f}' if rec.get("edge") is not None else "",
                "expected_roi": f"{exp_roi*100:.2f}",
                "kelly_fraction": f"{kelly_f:.4f}",
                "stake": f"{stake_amt:.2f}",
                "result": existing.get("result", ""),
                "pnl": existing.get("pnl", ""),
            }
            for extra_key in fieldnames:
                if extra_key not in updated:
                    updated[extra_key] = existing.get(extra_key, "")
            by_key[key] = updated

    ordered_rows = sorted(by_key.values(), key=lambda r: (r.get("date", ""), r.get("pick_type", ""), int(r.get("rank", "0") or 0)))
    with open(PICKS_HISTORY, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(ordered_rows)
    print(f"Tracked {sum(len(g) for _, g in tracked_groups)} picks for {today} in picks_history.csv")

def tracked_pick_summary_html():
    if not PICKS_HISTORY.exists():
        return ""

    try:
        with open(PICKS_HISTORY, newline="") as f:
            rows = list(csv.DictReader(f))
    except Exception:
        return ""

    if not rows:
        return ""

    today = datetime.now(ET).strftime("%Y-%m-%d")

    def summarize(subset):
        wins = sum(1 for r in subset if r.get("result") == "HR")
        total = len(subset)
        stake_total = 0.0
        pnl_total = 0.0
        for row in subset:
            try:
                stake_total += float(row.get("stake", TRACKED_STAKE) or TRACKED_STAKE)
            except Exception:
                stake_total += TRACKED_STAKE
            try:
                pnl_total += float(row.get("pnl", 0) or 0)
            except Exception:
                pass
        roi = (pnl_total / stake_total * 100) if stake_total else 0.0
        return wins, total, pnl_total, roi

    cards = []
    for pick_type, label in [
        ("highest_probability_top10", "Highest Probability Top 10"),
        ("best_value_top10", "Best Value Top 10"),
    ]:
        resolved = [r for r in rows if r.get("pick_type") == pick_type and r.get("result")]
        wins, total, pnl_total, roi = summarize(resolved) if resolved else (0, 0, 0.0, 0.0)
        recent = []
        for row in resolved:
            try:
                if row.get("date") and (datetime.now(ET).date() - datetime.strptime(row["date"], "%Y-%m-%d").date()).days <= 6:
                    recent.append(row)
            except Exception:
                continue
        r_wins, r_total, _, r_roi = summarize(recent) if recent else (0, 0, 0.0, 0.0)
        today_picks = [
            r for r in rows
            if r.get("pick_type") == pick_type and r.get("date") == today
        ]
        today_picks = sorted(today_picks, key=lambda r: int(r.get("rank", "0") or 0))
        current_line = (
            "Current tracked list: " + ", ".join(f'#{r.get("rank")} {r.get("player","—")}' for r in today_picks[:3]) +
            (" ..." if len(today_picks) > 3 else "")
        ) if today_picks else "Current tracked list: no picks saved yet"

        settled_dates = sorted({r.get("date") for r in resolved if r.get("date")})
        latest_settled = settled_dates[-1] if settled_dates else ""
        latest_subset = [r for r in resolved if r.get("date") == latest_settled] if latest_settled else []
        l_wins, l_total, l_pnl, l_roi = summarize(latest_subset) if latest_subset else (0, 0, 0.0, 0.0)
        latest_line = (
            f"Latest settled slate ({latest_settled}): {l_wins}/{l_total} · ROI {l_roi:+.1f}% · Profit ${l_pnl:+.0f}"
            if latest_subset else
            "Latest settled slate: none yet"
        )
        record_display = f"{wins}/{total}" if total else "Pending"
        meta_display = (
            f"Hit rate {(wins / total * 100 if total else 0):.1f}% · ROI {roi:+.1f}%"
            if total else
            "Waiting for settled results"
        )
        cards.append(
            f'<div class="track-card">'
            f'<div class="track-title">{label}</div>'
            f'<div class="track-record">{record_display}</div>'
            f'<div class="track-meta">{meta_display}</div>'
            f'<div class="track-submeta">Last 7 days: {r_wins}/{r_total} · ROI {r_roi:+.1f}% · Profit ${pnl_total:+.0f}</div>'
            f'<div class="track-submeta">{latest_line}</div>'
            f'<div class="track-pick">{current_line}</div>'
            f'</div>'
        )

    if not cards:
        return ""
    return (
        '<section class="tracker-section">'
        '<h2>📈 Tracking Results</h2>'
        '<div class="tracker-strip">' + "".join(cards) + '</div>'
        '</section>'
    )
 
# ── HTML generation ────────────────────────────────────────────
def generate_html(all_preds, games):
    now_et   = datetime.now(ET)
    date_big = now_et.strftime("%A, %B %d, %Y")
    updated  = now_et.strftime("%I:%M %p ET")
    tracker_summary = tracked_pick_summary_html()
 
    def odds_fmt(o):
        if o is None: return "—"
        return f"+{o}" if o > 0 else str(o)
 
    def edge_badge(e):
        if e is None:
            return '<span class="badge grey">Model Only</span>'
        if e >= STRONG_EDGE_MIN:
            return f'<span class="badge red">Strong (+{e:.1f}%)</span>'
        if e >= SURFACE_EDGE_MIN:
            return f'<span class="badge green">Moderate (+{e:.1f}%)</span>'
        if e >= 0:
            return f'<span class="badge yellow">Monitor (+{e:.1f}%)</span>'
        return f'<span class="badge dark">Avoid ({e:.1f}%)</span>'
 
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
 
        pitcher_in_model = bool(r.get("pitcher_found"))
        pitcher_note = "" if pitcher_in_model else \
            '<p class="note">⚠️ Pitcher not in model — using hitter data only</p>'
        book_lbl = BOOK_NAMES.get(r.get("book"), r.get("book", "Sportsbook").title())
        edge_str = (f"+{r['edge']:.1f}%" if r["edge"] and r["edge"] > 0
                    else f"{r['edge']:.1f}%" if r["edge"] is not None else "—")
        confidence = r.get("value", "Model Only")
        ph_badge = (f'<span class="ph-badge lhp">vs LHP</span>'
                    if r.get("pitcher_hand") == "L"
                    else f'<span class="ph-badge rhp">vs RHP</span>')
        platoon_badge = ""
        if r.get("platoon_note"):
            badge_text = "Platoon edge" if r.get("platoon_favorable") else "Platoon risk"
            badge_cls = "green" if r.get("platoon_favorable") else "dark"
            platoon_badge = f'<span class="badge {badge_cls}">{badge_text}</span>'
        weather_score = r.get("weather_stack_score")
        weather_badge = f'<span class="badge grey">Weather {weather_score}/5</span>' if weather_score is not None else ""
        trap_html = f'<p class="note">⚠️ Trap flag: {r["trap_flag"]}</p>' if r.get("trap_flag") else ""
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
    {platoon_badge}
    {weather_badge}
    {edge_badge(r['edge'])}
  </div>
  <div class="card-body">
    {odds_source}
    <div class="stats-row" style="margin-top:10px">
      <div class="stat"><label>Model Prob</label>{prob_bar(r['model_prob'])}</div>
      <div class="stat"><label>Implied %</label><span>{f"{r['book_implied']:.1f}%" if r['book_implied'] else "—"}</span></div>
      <div class="stat"><label>Edge</label><span class="edge-val {"edge-pos" if (r["edge"] or 0) > 0 else "edge-neg"}">{edge_str}</span></div>
      <div class="stat"><label>Tier</label><span>{confidence}</span></div>
    </div>
    <p class="matchup">vs <strong>{r['pitcher']}</strong> · {r['game_label']} · {r['time']}</p>
    {pitcher_note}
    {trap_html}
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
 
    # Tab 2: Best Value — only show DK edges that clear the new threshold.
    def value_score(r):
        model_prob = r.get("model_prob") or 0
        book_implied = r.get("book_implied") or 0
        edge = r.get("edge") or -999
        odds = r.get("book_odds") or 9999
        contact = r.get("h_hr_contact_score") or 0
        barrel = r.get("h_barrel_pct") or 0
        hard_hit = r.get("h_hard_hit_pct") or 0
        n_batted = r.get("h_n_batted") or 0
        reasons_txt = " ".join(r.get("reasons", [])).lower()

        side_bonus = 0.0
        if "favorable side" in reasons_txt or "pitcher is especially vulnerable to this side" in reasons_txt:
            side_bonus += 1.0
        elif "weaker side" in reasons_txt or r.get("platoon_mismatch"):
            side_bonus -= 1.5

        pitch_bonus = 0.0
        if "crushes" in reasons_txt or "strong contact quality vs" in reasons_txt:
            pitch_bonus += 0.8
        elif "weaker-than-usual hr profile vs" in reasons_txt or "below average" in reasons_txt:
            pitch_bonus -= 0.6

        quality_bonus = 0.0
        if contact >= 0.20:
            quality_bonus += 1.1
        elif contact >= 0.17:
            quality_bonus += 0.6
        if barrel >= 0.09:
            quality_bonus += 0.7
        elif barrel >= 0.07:
            quality_bonus += 0.3
        if hard_hit >= 0.36:
            quality_bonus += 0.5

        sample_bonus = 0.0
        if n_batted >= 300:
            sample_bonus += 0.35
        elif n_batted >= 180:
            sample_bonus += 0.15

        # Stars with very short odds should not automatically dominate value.
        star_penalty = 0.0
        if odds <= 260 and model_prob >= 17:
            star_penalty -= 1.4
        elif odds <= 320 and model_prob >= 15:
            star_penalty -= 0.8

        # Give a boost to mid-tier plus-money plays where the relative mispricing matters.
        tier_bonus = 0.0
        if 330 <= odds <= 700:
            tier_bonus += 0.9
        elif 701 <= odds <= 900:
            tier_bonus += 0.45

        rel_edge = edge / max(book_implied, 1)
        return (
            rel_edge * 4.5
            + edge * 0.40
            + quality_bonus
            + side_bonus
            + pitch_bonus
            + sample_bonus
            + tier_bonus
            + star_penalty
        )

    top_edge = [
        r for r in all_preds
        if r["edge"] is not None
        and r["book_implied"] is not None and r["book_implied"] > 0
        and r["book_odds"] is not None
        and r["value"] in {"Strong", "Moderate"}
    ]
    top_edge.sort(key=value_score, reverse=True)

    monitor = [
        r for r in all_preds
        if r["edge"] is not None and r["book_odds"] is not None and r["value"] == "Monitor"
    ]
    monitor.sort(key=value_score, reverse=True)

    edge_display = top_edge[:10]
    if len(edge_display) < 5:
        edge_display.extend(monitor[: max(0, 5 - len(edge_display))])

    top_edge_html = "\n".join(player_card(r, i+1) for i, r in enumerate(edge_display[:10])) \
                    if edge_display else '<p class="empty">No Strong or Moderate DK value picks right now.</p>'
 
    top_picks_tabs = f"""
  <div class="tab-bar top-tab-bar">
    <button class="top-tab-btn active" onclick="showTopTab('prob')">🎯 Highest Probability</button>
    <button class="top-tab-btn" onclick="showTopTab('edge')">💰 DK Edge Board <span class="tab-hint">(Strong 10+ · Moderate 5-9 · Monitor 2-4)</span></button>
  </div>
  <div id="tab-prob" class="top-tab-panel active">{top_prob_html}</div>
  <div id="tab-edge" class="top-tab-panel">{top_edge_html}</div>"""
 
    # ── Helper: pitcher stat box ───────────────────────────────
    def pitcher_box(pname, hand, team_color="rhp"):
        if not pname or pname == "TBD":
            return f'<div class="pitcher-box"><div class="pb-name">TBD <span class="ph-badge {team_color}">?HP</span></div><div class="pb-stats"><span class="no-data">Pitcher TBD</span></div></div>'
        p_row = _resolve_pitcher_row(pname, allow_fuzzy=True)
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
 
            def _rec_stat(v, fmt, mult=1):
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    return "<td class='ln-stat'>—</td>"
                return f"<td class='ln-stat'>{fmt.format(float(v) * mult)}</td>"
            stat_cells = (
                _rec_stat(r.get("h_barrel_pct"), "{:.0f}%", 100)
                + _rec_stat(r.get("h_exit_velo"), "{:.0f}")
                + _rec_stat(r.get("h_hard_hit_pct"), "{:.0f}%", 100)
            )
 
            # Top reason (first bullet, strip emoji prefix for compactness)
            top_reason = ""
            if r["reasons"]:
                ordered_reasons = sorted(r["reasons"], key=_reason_priority)
                raw = ordered_reasons[0]
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
  <th>Player</th><th>Prob</th><th>Book Odds</th><th>Edge</th>
  <th>Barrel%</th><th>Exit Velo</th><th>Hard Hit%</th><th>Top Factor</th>
</tr></thead>
<tbody>{rows}</tbody>
</table>"""
 
    tab_buttons = ""
    tab_panels  = ""
    compass     = ["N","NE","E","SE","S","SW","W","NW"]
    for i, (gkey, gdata) in enumerate(games.items()):
        w       = gdata.get("weather", {})
        weather_stack = compute_weather_stack(gdata["home"], w)
        dir_str = compass[int((w.get("wind_dir",0)+22.5)/45)%8] if w.get("wind_dir") else "—"
        weather_str = (f'🌡 {w.get("temp_f",0):.0f}°F &nbsp;·&nbsp; '
                       f'💨 {w.get("wind_speed",0):.0f} mph {dir_str} &nbsp;·&nbsp; '
                       f'💧 {w.get("humidity",0):.0f}% humidity &nbsp;·&nbsp; '
                       f'Weather stack {weather_stack["score"]}/5') if w else ""
 
        away_players = sorted([r for r in gdata["players"] if r["team"] == gdata["away"] and r["model_prob"]],
                              key=lambda r: r["model_prob"] or 0, reverse=True)
        home_players = sorted([r for r in gdata["players"] if r["team"] == gdata["home"] and r["model_prob"]],
                              key=lambda r: r["model_prob"] or 0, reverse=True)
 
        # Show the pitcher each lineup is actually facing, not that team's own starter.
        away_pb = pitcher_box(gdata["home_pitcher"], gdata["home_pitcher_hand"])
        home_pb = pitcher_box(gdata["away_pitcher"], gdata["away_pitcher_hand"])
 
        active = " active" if i == 0 else ""
        tab_buttons += (f'<button class="tab-btn{active}" onclick="showTab(\'{gkey}\')">'
                        f'{gdata.get("display_label", gkey)}<br><small>{gdata["time"]}</small></button>\n')
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
<title>Weight Room Hero Sim — {date_big}</title>
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
  .header{{background:#ffffff;padding:28px 20px 24px;text-align:center;border-bottom:1px solid #e2e2e6}}
  .brand-kicker{{font-size:11px;font-weight:800;color:#4f86f7;letter-spacing:.14em;text-transform:uppercase;margin-bottom:8px}}
  .header .date{{font-size:clamp(30px,4.8vw,56px);font-weight:900;color:#0a0a0c;letter-spacing:-1.4px}}
  .header .subtitle{{color:#5a5a6e;margin-top:8px;font-size:12px;letter-spacing:.06em;text-transform:uppercase;font-weight:600}}
  .header .games-date{{color:#5a5a6e;margin-top:6px;font-size:14px;font-weight:700;letter-spacing:-.02em}}
  .header-links{{display:flex;justify-content:center;gap:10px;flex-wrap:wrap;margin-top:14px}}
  .header-link{{display:inline-flex;align-items:center;gap:6px;padding:7px 12px;border:1px solid #d7d7df;border-radius:999px;color:#20202a;text-decoration:none;font-size:12px;font-weight:700;letter-spacing:.01em;transition:border-color .15s,color .15s,transform .15s}}
  .header-link:hover{{border-color:#4f86f7;color:#4f86f7;transform:translateY(-1px)}}
 
  /* ── Layout ── */
  .container{{max-width:1440px;margin:0 auto;padding:24px 24px}}
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
  .edge-pos{{color:var(--green) !important;font-weight:700}}
  .edge-neg{{color:var(--red) !important;font-weight:700}}
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
  .badge.dark{{background:rgba(240,79,88,.12);color:var(--red)}}
 
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
  .tab-hint{{font-size:11px;color:var(--soft);font-weight:600}}
  .top-tab-panel{{display:none}}.top-tab-panel.active{{display:block}}
  .tracker-section{{margin-top:32px}}
  .tracker-strip{{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px;margin-top:14px}}
  .track-card{{background:linear-gradient(180deg,rgba(79,134,247,.08),rgba(255,255,255,.02));border:1px solid var(--border);border-radius:12px;padding:14px 16px}}
  .track-title{{font-size:11px;text-transform:uppercase;letter-spacing:.09em;color:var(--muted);font-weight:700;margin-bottom:8px}}
  .track-record{{font-size:28px;font-weight:900;letter-spacing:-.8px;color:var(--text);line-height:1}}
  .track-meta{{margin-top:8px;font-size:13px;color:#b8b8c8;font-weight:600}}
  .track-submeta{{margin-top:4px;font-size:12px;color:#9a9aad;font-weight:600}}
  .track-pick{{margin-top:10px;font-size:12px;color:#e1e1ea;font-weight:700;line-height:1.4}}
 
  /* ── Game matchup layout ── */
  .game-weather{{font-size:11px;color:var(--muted);text-align:center;padding:6px 0 16px;letter-spacing:.03em}}
  .matchup-grid{{display:grid;grid-template-columns:minmax(0,1fr) 24px minmax(0,1fr);gap:16px;align-items:start}}
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
  .lineup-wrap{{overflow-x:auto;border:1px solid var(--border);border-radius:12px;background:linear-gradient(180deg,rgba(255,255,255,.02),rgba(255,255,255,.01))}}
  .lineup-tbl{{width:100%;border-collapse:collapse;font-size:12px;table-layout:fixed;min-width:0}}
  .lineup-tbl thead th{{padding:10px 14px;text-align:left;font-size:10px;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);border-bottom:1px solid var(--border);white-space:nowrap;font-weight:700;background:rgba(255,255,255,.02)}}
  .ln-row{{border-bottom:1px solid var(--border);transition:background .1s}}
  .ln-row:hover{{background:rgba(255,255,255,.025)}}
  .ln-row td{{padding:12px 14px;vertical-align:middle}}
  .lineup-tbl th:nth-child(1), .lineup-tbl td:nth-child(1){{width:22%}}
  .lineup-tbl th:nth-child(2), .lineup-tbl td:nth-child(2){{width:11%}}
  .lineup-tbl th:nth-child(3), .lineup-tbl td:nth-child(3){{width:12%}}
  .lineup-tbl th:nth-child(4), .lineup-tbl td:nth-child(4){{width:11%}}
  .lineup-tbl th:nth-child(5), .lineup-tbl td:nth-child(5),
  .lineup-tbl th:nth-child(6), .lineup-tbl td:nth-child(6),
  .lineup-tbl th:nth-child(7), .lineup-tbl td:nth-child(7){{width:11%}}
  .lineup-tbl th:nth-child(8), .lineup-tbl td:nth-child(8){{width:22%}}
  .ln-name{{font-weight:800;white-space:nowrap;font-size:14px;color:var(--text);letter-spacing:-.2px}}
  .ln-prob{{font-weight:900;font-size:15px;white-space:nowrap;letter-spacing:-.4px}}
  .lineup-tbl th:nth-child(1){{padding-right:24px}}
  .lineup-tbl td:nth-child(1){{padding-right:24px}}
  .lineup-tbl th:nth-child(2){{padding-left:20px}}
  .lineup-tbl td:nth-child(2){{padding-left:20px}}
  .ln-odds{{color:var(--green);font-weight:800;font-size:14px}}
  .ln-edge-pos{{color:var(--green) !important;font-weight:800;font-size:13px}}
  .ln-edge-neg{{color:var(--red) !important;font-size:13px;font-weight:700}}
  .ln-stat{{color:#d7d7e5;font-size:13px;font-weight:700}}
  .ln-reason{{color:#a9a9bb;font-size:12px;line-height:1.45;overflow-wrap:anywhere}}
  .lineup-tbl tbody td:nth-child(2),
  .lineup-tbl tbody td:nth-child(3),
  .lineup-tbl tbody td:nth-child(4){{background:rgba(255,255,255,.018)}}
  .lineup-tbl tbody td:nth-child(2){{box-shadow:inset 0 0 0 1px rgba(255,255,255,.03)}}
  .lineup-tbl tbody td:nth-child(5),
  .lineup-tbl tbody td:nth-child(6),
  .lineup-tbl tbody td:nth-child(7){{background:rgba(79,134,247,.05)}}
  .lineup-tbl tbody td:nth-child(2), .lineup-tbl tbody td:nth-child(3), .lineup-tbl tbody td:nth-child(4),
  .lineup-tbl tbody td:nth-child(5), .lineup-tbl tbody td:nth-child(6), .lineup-tbl tbody td:nth-child(7){{text-align:center}}
  .no-data{{color:var(--muted);font-style:italic;font-size:12px}}
  .empty{{color:var(--muted);padding:20px 0;text-align:center;font-size:13px}}
  .footer{{text-align:center;color:var(--muted);font-size:11px;padding:24px 20px;border-top:1px solid var(--border);margin-top:40px;letter-spacing:.03em}}
  @media(max-width:1100px){{.container{{padding:20px 16px}}.matchup-grid{{grid-template-columns:1fr}}.vs-divider{{display:none}}}}
  @media(max-width:700px){{.stats-row{{gap:16px}}.card-header{{flex-direction:column;align-items:flex-start}}.top-tab-bar{{width:100%}}.lineup-tbl{{font-size:11px;min-width:760px}}.lineup-tbl thead th{{padding:8px 10px}}.ln-row td{{padding:10px 10px}}}}
</style>
</head>
<body>
<div class="header">
  <div class="brand-kicker">Weight Room Hero Sim</div>
  <div class="date">Weight Room Hero Sim</div>
  <div class="games-date">{date_big}</div>
  <div class="subtitle">MLB Home Run Model &nbsp;·&nbsp; Updated {updated}</div>
  <div class="header-links">
    <a class="header-link" href="https://www.linkedin.com/in/justinloo12/" target="_blank" rel="noopener noreferrer">LinkedIn · @justinloo12</a>
    <a class="header-link" href="https://www.instagram.com/justinloo12/" target="_blank" rel="noopener noreferrer">Instagram · @justinloo12</a>
  </div>
</div>
<div class="container">
  <h2>🏆 Top Picks Today</h2>
  {top_picks_tabs}
  <h2>📅 Today's Games</h2>
  {'<p class="empty">No games scheduled today.</p>' if not games else ''}
  <div class="tab-bar">{tab_buttons}</div>
  {tab_panels}
  {tracker_summary}
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
    update_picks_history(all_preds)
    html = generate_html(all_preds, games)
    with open("index.html", "w") as f:
        f.write(html)
    print("\nindex.html saved — open it in your browser!")
