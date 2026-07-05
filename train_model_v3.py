"""GBM v3 — the ML rematch, done right this time.

The original 85-feature GBM was trained at the BATTED-BALL level (P(HR |
batted ball)) and then contorted into per-game probabilities at deployment.
On a strict holdout it lost to the naive base rate. v3 asks the fair
question: if the ML is trained on the deployment question itself, with the
structural model's own inputs plus the rich profile features, all built
leakage-safe, does it beat the formula?

Design:

* UNIT: player-game. One row per (game, batter with >= 1 PA); label = hit
  >= 1 HR that game. Identical population to evaluate_model.py's
  player-game frame, so every number is directly comparable.

* FEATURES (all knowable pre-game, all as-of the game date):
  - Structural inputs (s_*): shrunken batter HR/PA, opposing-starter
    factor, platoon factor, park factor by handedness, slot-aware E[PA],
    the lineup slot itself, and structural v2's own probability (as
    log-odds) — the formula is handed to the ML as a feature.
  - Profile features: the same ~85 hitter/pitcher/matchup features as the
    original GBM (model_features.csv), assembled from profiles built ONLY
    from data before each segment cutoff (use_season_stats=False, so the
    season-level p_xera_gap leak stays excluded).
  - Weather: actual historical values (same optimistic stand-in as the
    main evaluation, applied to every model equally).
  Exclusions (cannot be made point-in-time, documented): p_xera_gap /
  season xERA aggregates; everything else in model_features.csv is
  rebuilt as-of each cutoff.

* TIME DISCIPLINE:
  - Training rows come from walk-forward segments inside the training
    window (2025-05-01 .. 2026-03-22); each segment is scored with
    profiles/structural tables built strictly BEFORE the segment start.
    The first month of 2025 is burn-in (profiles only, no training rows).
  - Model selection AND probability calibration (isotonic vs sigmoid)
    use the LAST TWO training segments (2025-09-15 .. 2025-12-01 and
    2026-03-01 .. 2026-03-22) as a time-ordered validation slice. No
    random CV across time anywhere.
  - The holdout (2026-03-23 .. 2026-07-02) reuses evaluate_model.py's
    exact walk-forward segments; the tree weights are frozen.

Outputs: appends a "gbm_v3" block to eval_results.json, writes
hr_model_v3.pkl and holdout_predictions_v3.csv (per-row predictions for
error analysis; gitignored).
"""

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

import build_features as bf
from evaluate_model import (
    HOLDOUT_PATH, ONE_SIDED_MARGIN, SEGMENTS, TRAIN_CUTOFF, TRAIN_PATH,
    build_player_games, fetch_holdout_weather, load_pitches,
    load_train_weather, metric_row,
)
from structural_model import (
    MAX_LINEUP_SLOT, P_PA_CAP, PITCHER_FACTOR_CLIP, PLATOON_FACTOR_CLIP,
    PLATOON_PSEUDO_PA, SLOT_EPA_WEIGHT, STARTER_PA_SHARE, StructuralModelV2,
    game_hr_probability, infer_game_slots, pa_table_from_statcast,
    shrunk_rate,
)

warnings.filterwarnings("ignore", category=FutureWarning)

# Walk-forward segments INSIDE the training window: rows in [start, end)
# are scored with profiles built from data < start. 2025-04 is burn-in.
TRAIN_SEGMENTS = [
    (pd.Timestamp("2025-05-01"), pd.Timestamp("2025-06-15")),
    (pd.Timestamp("2025-06-15"), pd.Timestamp("2025-08-01")),
    (pd.Timestamp("2025-08-01"), pd.Timestamp("2025-09-15")),
    (pd.Timestamp("2025-09-15"), pd.Timestamp("2025-12-01")),
    (pd.Timestamp("2026-03-01"), pd.Timestamp("2026-03-23")),
]
# The last N training segments form the time-ordered validation slice used
# for model selection and probability calibration (never for tree fitting).
# Two segments (2025-09-15..12-01 + 2026-03) give ~7.5k rows / ~700
# positives across two different calendar contexts, instead of a thin
# March-only slice whose low HR rate would skew the calibration mapping.
N_VALIDATION_SEGMENTS = 2

# Candidate HGB configs, selected by validation log-loss (time-ordered
# slice, never a random split). Kept deliberately small and shallow-ish:
# 27k positive labels do not support a huge hyperparameter search.
CANDIDATES = [
    {"max_iter": 300, "learning_rate": 0.05, "max_leaf_nodes": 31, "min_samples_leaf": 60},
    {"max_iter": 150, "learning_rate": 0.10, "max_leaf_nodes": 31, "min_samples_leaf": 60},
    {"max_iter": 300, "learning_rate": 0.05, "max_leaf_nodes": 15, "min_samples_leaf": 100},
]

STRUCT_FEATURES = [
    "s_batter_rate", "s_pitcher_factor", "s_platoon_factor",
    "s_park_factor", "s_epa", "s_slot", "s_struct_v2_logit",
]

MODEL_PATH = "hr_model_v3.pkl"
PRED_PATH = "holdout_predictions_v3.csv"
EPS = 1e-6


# ── Structural feature frame (vectorized twin of StructuralModelV2) ──────

def structural_feature_frame(struct, pg):
    """Compute the structural model's components as ML features, vectorized.

    `pg` needs columns: batter, opp_starter_id, opp_starter_hand, slot
    (NaN for unknown/sub), home_team, stand. Returns a DataFrame with
    STRUCT_FEATURES plus s_struct_v2_prob, aligned to pg's index.

    Deterministic and exactly consistent with StructuralModelV2.predict_v2
    (asserted in tests): the per-PA probability is the product of the
    components, capped at P_PA_CAP, and the game probability is
    1 - (1 - p)^E[PA].
    """
    prior_mean_b = struct.alpha_b / (struct.alpha_b + struct.beta_b)

    # Batter shrunk rate.
    bt = struct.batter_totals
    rate_s = shrunk_rate(bt["hr"], bt["pa"], struct.alpha_b, struct.beta_b)
    s_batter = pg["batter"].map(rate_s).fillna(prior_mean_b)

    # Pitcher factor (shrunk, clipped, blended toward 1.0 for the bullpen
    # share) — same math as structural_model.pitcher_factor.
    pt = struct.pitcher_totals
    shrunk_p = shrunk_rate(pt["hr"], pt["pa"], struct.alpha_p, struct.beta_p)
    raw = (shrunk_p / struct.league_rate).clip(*PITCHER_FACTOR_CLIP)
    pf_s = STARTER_PA_SHARE * raw + (1.0 - STARTER_PA_SHARE)
    s_pitcher = pg["opp_starter_id"].map(pf_s).fillna(1.0)

    # Platoon factor: batter's vs-hand rate shrunk toward his own overall
    # shrunk rate with PLATOON_PSEUDO_PA pseudo-observations.
    bh = struct.batter_by_hand.reset_index()
    bh["overall"] = bh["batter"].map(rate_s).fillna(prior_mean_b)
    bh["rate_hand"] = (bh["hr"] + PLATOON_PSEUDO_PA * bh["overall"]) / (bh["pa"] + PLATOON_PSEUDO_PA)
    bh["factor"] = (bh["rate_hand"] / bh["overall"]).clip(*PLATOON_FACTOR_CLIP)
    factor_map = bh.set_index(["batter", "p_throws"])["factor"]
    keys = pd.MultiIndex.from_arrays([pg["batter"], pg["opp_starter_hand"]])
    s_platoon = pd.Series(factor_map.reindex(keys).values, index=pg.index).fillna(1.0)
    # Unknown hand -> neutral (predict_v2 only applies the split for R/L).
    s_platoon = s_platoon.where(pg["opp_starter_hand"].isin(["R", "L"]), 1.0)

    # Park factor by (park, batter side); falls back to the batter's modal
    # side from the pre-cutoff data, exactly like predict_v2(stand=None).
    stand = pg["stand"].where(pg["stand"].notna(),
                              pg["batter"].map(struct.batter_stand))
    s_park = pd.Series(
        [struct.park_factors.get((t, s), 1.0) for t, s in zip(pg["home_team"], stand)],
        index=pg.index, dtype=float)

    # Slot-aware E[PA]: blend league slot mean with trailing regressed
    # PA/game; unknown slot (NaN, i.e. a sub) falls back to trailing only.
    games = struct.batter_games
    total_pa = bt["pa"]
    trailing_tbl = ((total_pa + 10.0 * struct.league_pa_per_game)
                    / (games.reindex(total_pa.index).fillna(0.0) + 10.0))
    trailing = pg["batter"].map(trailing_tbl).fillna(struct.league_pa_per_game)
    slot_mean = pg["slot"].map(struct.pa_by_slot)
    s_epa = np.where(slot_mean.notna(),
                     SLOT_EPA_WEIGHT * slot_mean + (1.0 - SLOT_EPA_WEIGHT) * trailing,
                     trailing)

    p_pa = np.minimum(s_batter * s_pitcher * s_platoon * s_park, P_PA_CAP)
    prob = game_hr_probability(p_pa.values, s_epa)
    prob_c = np.clip(prob, EPS, 1 - EPS)

    return pd.DataFrame({
        "s_batter_rate": s_batter.values,
        "s_pitcher_factor": s_pitcher.values,
        "s_platoon_factor": s_platoon.values,
        "s_park_factor": s_park.values,
        "s_epa": s_epa,
        "s_slot": pg["slot"].values,
        "s_struct_v2_logit": np.log(prob_c / (1 - prob_c)),
        "s_struct_v2_prob": prob,
    }, index=pg.index)


# ── Calibration (fit on a time-ordered validation slice) ─────────────────

def fit_calibrators(p_val, y_val):
    """Fit isotonic and Platt-sigmoid calibrators on validation scores.

    Returns {"isotonic": iso, "sigmoid": lr} — apply with
    apply_calibrator(). Both map raw model probabilities to calibrated
    probabilities; the caller picks one by validation log-loss.
    """
    iso = IsotonicRegression(out_of_bounds="clip", y_min=EPS, y_max=1 - EPS)
    iso.fit(p_val, y_val)
    logit = np.log(np.clip(p_val, EPS, 1 - EPS) / (1 - np.clip(p_val, EPS, 1 - EPS)))
    lr = LogisticRegression(C=1e6, solver="lbfgs")
    lr.fit(logit.reshape(-1, 1), y_val)
    return {"isotonic": iso, "sigmoid": lr}


def apply_calibrator(cal, method, p_raw):
    p_raw = np.asarray(p_raw, dtype=float)
    if method == "isotonic":
        return np.clip(cal["isotonic"].predict(p_raw), EPS, 1 - EPS)
    pc = np.clip(p_raw, EPS, 1 - EPS)
    logit = np.log(pc / (1 - pc))
    return cal["sigmoid"].predict_proba(logit.reshape(-1, 1))[:, 1]


# ── Assembly helpers ──────────────────────────────────────────────────────

def _profile_bundle(pre_pitches):
    """Profiles + structural tables from strictly pre-cutoff pitches."""
    all_pre, batted_pre = bf.prepare_pitch_data(pre_pitches)
    hitter, pitcher, matchup, _ = bf.build_profiles(
        all_pre, batted_pre, use_season_stats=False, verbose=False)
    pa_pre = pa_table_from_statcast(all_pre)
    struct = StructuralModelV2(pa_pre)
    pg_base_rate = float(pa_pre.groupby(["game_pk", "batter"])["is_hr"].max().mean())
    return hitter, pitcher, matchup, struct, pg_base_rate


def assemble_segment(pg, hitter, pitcher, matchup, struct, features,
                     team_categories, weather):
    """Feature matrix for one segment of player-games.

    Returns (X, struct_frame): X has the profile features + STRUCT_FEATURES
    (NaN preserved — HistGradientBoosting handles missing natively),
    struct_frame carries s_struct_v2_prob for downstream comparison.
    """
    rows = pg.rename(columns={
        "opp_starter_name": "player_name", "opp_starter_hand": "p_throws"})[
        ["batter", "player_name", "stand", "p_throws", "home_team", "game_date"]]
    frame = bf.assemble_matchup_frame(
        rows, hitter, pitcher, matchup, team_categories=team_categories)
    frame = frame.merge(weather, on=["game_date", "home_team"], how="left")
    # assemble_matchup_frame can emit a column twice (profile merge +
    # derived-column concat, e.g. p_hr_vulnerability_score); keep the first
    # copy and reindex to a canonical order so every segment — train,
    # validation, holdout — produces the identical matrix layout.
    frame = frame.loc[:, ~frame.columns.duplicated()]
    sf = structural_feature_frame(struct, pg.reset_index(drop=True))
    X = pd.concat([frame.reindex(columns=features).reset_index(drop=True),
                   sf[STRUCT_FEATURES].reset_index(drop=True)], axis=1)
    return X, sf


def attach_slots(pg, pitches):
    """Merge each player-game's inferred lineup slot (subs -> NaN)."""
    slot_map = infer_game_slots(pa_table_from_statcast(pitches))
    pg = pg.merge(slot_map, on=["game_pk", "batter"], how="left")
    pg.loc[pg["slot"] > MAX_LINEUP_SLOT, "slot"] = np.nan
    return pg


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print("Loading data...")
    train_raw = load_pitches(TRAIN_PATH)
    train_raw = train_raw[train_raw["game_date"] < TRAIN_CUTOFF]
    holdout_raw = load_pitches(HOLDOUT_PATH)
    holdout_raw = holdout_raw[holdout_raw["game_date"] >= TRAIN_CUTOFF]
    team_categories = sorted(train_raw["home_team"].dropna().unique())
    profile_features = pd.read_csv("model_features.csv")["feature"].tolist()

    train_weather = load_train_weather()
    holdout_games = holdout_raw[["game_date", "home_team"]].drop_duplicates()
    test_weather = fetch_holdout_weather(holdout_games)

    # ── Training rows, walk-forward inside the train window ─────────
    X_parts, y_parts, seg_ids = [], [], []
    for i, (cutoff, seg_end) in enumerate(TRAIN_SEGMENTS):
        seg_t = time.time()
        pre = train_raw[train_raw["game_date"] < cutoff]
        seg_pitches = train_raw[(train_raw["game_date"] >= cutoff)
                                & (train_raw["game_date"] < seg_end)]
        if seg_pitches.empty:
            continue
        hitter, pitcher, matchup, struct, _ = _profile_bundle(pre)
        pg = attach_slots(build_player_games(seg_pitches), seg_pitches)
        X, _ = assemble_segment(pg, hitter, pitcher, matchup, struct,
                                profile_features, team_categories, train_weather)
        X_parts.append(X)
        y_parts.append(pg["is_hr_game"].values)
        seg_ids.append(np.full(len(X), i))
        print(f"  train seg {cutoff:%Y-%m-%d}..{seg_end:%Y-%m-%d}: "
              f"{len(X):,} player-games, {pg['is_hr_game'].sum():,} HR-games "
              f"({time.time()-seg_t:.0f}s)")

    X_all = pd.concat(X_parts, ignore_index=True)
    y_all = np.concatenate(y_parts)
    seg_all = np.concatenate(seg_ids)
    n_val_seg = len(TRAIN_SEGMENTS) - N_VALIDATION_SEGMENTS
    fit_mask = seg_all < n_val_seg
    X_fit, y_fit = X_all[fit_mask], y_all[fit_mask]
    X_val, y_val = X_all[~fit_mask], y_all[~fit_mask]
    feature_list = list(X_all.columns)
    print(f"\n  fit rows: {len(X_fit):,} ({int(y_fit.sum()):,} pos) | "
          f"validation rows: {len(X_val):,} ({int(y_val.sum()):,} pos)")

    # ── Model selection on the time-ordered validation slice ────────
    best = None
    for params in CANDIDATES:
        clf = HistGradientBoostingClassifier(
            random_state=42, early_stopping=False, **params)
        clf.fit(X_fit, y_fit)
        p_val = clf.predict_proba(X_val)[:, 1]
        ll = log_loss(y_val, np.clip(p_val, EPS, 1 - EPS))
        print(f"  candidate {params}: val log-loss {ll:.5f}")
        if best is None or ll < best["ll"]:
            best = {"clf": clf, "params": params, "ll": ll, "p_val": p_val}
    clf = best["clf"]

    # ── Calibration on the same time-ordered slice ───────────────────
    cal = fit_calibrators(best["p_val"], y_val)
    method_ll = {m: log_loss(y_val, apply_calibrator(cal, m, best["p_val"]))
                 for m in ["isotonic", "sigmoid"]}
    cal_method = min(method_ll, key=method_ll.get)
    print(f"  calibration: {method_ll} -> chose {cal_method}")

    # ── Holdout scoring: identical walk-forward segments ─────────────
    master = pd.concat([train_raw, holdout_raw], ignore_index=True)
    del train_raw
    test_pg_all = attach_slots(build_player_games(holdout_raw), holdout_raw)

    pg_frames = []
    for cutoff, seg_end in SEGMENTS:
        seg_t = time.time()
        pre = master[master["game_date"] < cutoff]
        hitter, pitcher, matchup, struct, pg_base = _profile_bundle(pre)
        pg = test_pg_all[(test_pg_all["game_date"] >= cutoff)
                         & (test_pg_all["game_date"] < seg_end)].copy()
        if pg.empty:
            continue
        X, sf = assemble_segment(pg, hitter, pitcher, matchup, struct,
                                 profile_features, team_categories, test_weather)
        p_raw = clf.predict_proba(X.reindex(columns=feature_list))[:, 1]
        pg["gbm_v3_raw"] = p_raw
        pg["gbm_v3"] = apply_calibrator(cal, cal_method, p_raw)
        pg["gbm_v3_alt"] = apply_calibrator(
            cal, "sigmoid" if cal_method == "isotonic" else "isotonic", p_raw)
        pg["structural_v2"] = sf["s_struct_v2_prob"].values
        pg["s_pitcher_factor"] = sf["s_pitcher_factor"].values
        pg["structural_v1"] = [
            struct.predict(b, opposing_pitcher=pid, pitcher_hand=hand)
            for b, pid, hand in zip(pg["batter"], pg["opp_starter_id"],
                                    pg["opp_starter_hand"])]
        pg["base"] = pg_base
        pg_frames.append(pg)
        print(f"  holdout seg {cutoff:%Y-%m-%d}..{seg_end:%Y-%m-%d}: "
              f"{len(pg):,} player-games ({time.time()-seg_t:.0f}s)")

    pg_all = pd.concat(pg_frames, ignore_index=True)

    # ── Picks-subset comparison (same join as evaluate_model.py) ─────
    picks = pd.read_csv("picks_history.csv", parse_dates=["date"])
    picks = picks.dropna(subset=["book_implied", "batter_id"])
    picks = picks.groupby(["date", "batter_id"], as_index=False).agg(
        book_implied=("book_implied", "mean"))
    picks["book_implied"] = picks["book_implied"] / 100.0
    merged = picks.merge(pg_all, left_on=["date", "batter_id"],
                         right_on=["game_date", "batter"], how="inner")
    merged["devig"] = merged["book_implied"] / ONE_SIDED_MARGIN

    # ── Results ───────────────────────────────────────────────────────
    y = pg_all["is_hr_game"]
    alt = "sigmoid" if cal_method == "isotonic" else "isotonic"
    block = {
        "config": {
            "unit": "player-game (one row per batter-game, label = homered y/n)",
            "features": f"{len(profile_features)} profile features (leakage-safe, "
                        f"use_season_stats=False) + {len(STRUCT_FEATURES)} structural "
                        "inputs incl. structural v2 log-odds",
            "excluded": "p_xera_gap / season-level xERA aggregates (not point-in-time)",
            "train_segments": [f"{c:%Y-%m-%d}..{e:%Y-%m-%d}" for c, e in TRAIN_SEGMENTS],
            "validation_slice": "last two train segments (2025-09-15..2025-12-01 + "
                                "2026-03-01..2026-03-22), time-ordered",
            "model": f"HistGradientBoostingClassifier {best['params']}",
            "calibration": f"{cal_method} (chosen by validation log-loss "
                           f"{method_ll[cal_method]:.5f} vs {method_ll[alt]:.5f})",
        },
        "validation": {
            "n": int(len(y_val)), "positives": int(y_val.sum()),
            "raw_log_loss": round(best["ll"], 5),
            "calibrated_log_loss": {m: round(v, 5) for m, v in method_ll.items()},
        },
        "player_game_level": {
            "gbm_v3": metric_row(y, pg_all["gbm_v3"]),
            f"gbm_v3_{alt}_alt": metric_row(y, pg_all["gbm_v3_alt"]),
            "gbm_v3_raw_uncalibrated": metric_row(y, pg_all["gbm_v3_raw"]),
            "structural_v2_recomputed": metric_row(y, pg_all["structural_v2"]),
            "base_rate": metric_row(y, pg_all["base"]),
        },
        "picks_subset (n=%d)" % len(merged): {
            "market_implied_devig": metric_row(merged["is_hr_game"], merged["devig"]),
            "gbm_v3": metric_row(merged["is_hr_game"], merged["gbm_v3"]),
            "structural_v2": metric_row(merged["is_hr_game"], merged["structural_v2"]),
        } if len(merged) else {},
    }

    results = json.loads(Path("eval_results.json").read_text())
    results["gbm_v3"] = block
    Path("eval_results.json").write_text(json.dumps(results, indent=2))
    print("\n" + json.dumps(block, indent=2))

    # ── Persist artifacts ─────────────────────────────────────────────
    import pickle
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"format": "hr_model_v3", "clf": clf, "calibrators": cal,
                     "cal_method": cal_method, "features": feature_list,
                     "params": best["params"]}, f)
    keep = ["game_pk", "game_date", "batter", "stand", "home_team", "slot",
            "opp_starter_id", "opp_starter_hand", "n_pa", "is_hr_game",
            "base", "structural_v1", "structural_v2", "s_pitcher_factor",
            "gbm_v3_raw", "gbm_v3", "gbm_v3_alt"]
    pg_all[keep].to_csv(PRED_PATH, index=False)
    print(f"\nSaved: eval_results.json (+gbm_v3), {MODEL_PATH}, {PRED_PATH}")
    print(f"Total runtime: {time.time()-t0:.0f}s")


def build_scoring_pack(out_path="scoring_pack_v3.pkl.gz", verbose=True):
    """Freeze the deployment feature tables for live v3 scoring.

    Profiles are built from ALL Statcast data on disk (train + holdout,
    use_season_stats=False — same feature definitions the model was
    trained on) and gzipped so predict_today.py can assemble v3 features
    in CI without the multi-GB raw data. Deployment artifact: frozen at
    registration time per PROTOCOL.json; never score historical games
    with it.
    """
    import gzip
    import pickle

    frames = [load_pitches(p) for p in (TRAIN_PATH, HOLDOUT_PATH) if Path(p).exists()]
    master = pd.concat(frames, ignore_index=True)
    all_p, batted = bf.prepare_pitch_data(master)
    hitter, pitcher, matchup, _ = bf.build_profiles(
        all_p, batted, use_season_stats=False, verbose=False)
    names = (all_p.dropna(subset=["player_name"])
             .groupby("pitcher")["player_name"].first().to_dict())
    pack = {
        "format": "v3_scoring_pack",
        "built_through": f"{master['game_date'].max():%Y-%m-%d}",
        "hitter": hitter, "pitcher": pitcher, "matchup": matchup,
        "team_categories": sorted(master["home_team"].dropna().unique()),
        "pitcher_name_by_id": names,
        "features": pd.read_csv("model_features.csv")["feature"].tolist(),
    }
    with gzip.open(out_path, "wb") as f:
        pickle.dump(pack, f)
    if verbose:
        print(f"  scoring pack through {pack['built_through']} -> {out_path} "
              f"({Path(out_path).stat().st_size/1e6:.1f} MB)")
    return pack


if __name__ == "__main__":
    import sys
    if sys.argv[1:2] == ["pack"]:
        build_scoring_pack()
    else:
        main()
