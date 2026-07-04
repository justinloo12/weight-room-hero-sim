"""Honest time-based holdout evaluation of the HR model.

Design (leakage rules spelled out):

* TRAIN window: every pitch on disk in homerun_data_all.csv, i.e.
  2025-03-27 .. 2026-03-22 (2025 season + spring/early 2026). The GBM is
  trained ONCE on this window, with profile features built only from this
  window — identical architecture to train_model.py.

* TEST window: 2026-03-23 .. 2026-07-02 (holdout_2026.csv, pulled fresh
  from Statcast). Neither the shipped hr_model.pkl (built 2026-04-15 from
  data ending 2026-03-22) nor the GBM re-trained here has seen any of it.

* Walk-forward profile refreshes: the test window is split into segments
  (see SEGMENTS). For each segment, hitter/pitcher/matchup profiles and the
  structural model's tables are rebuilt from all data BEFORE the segment
  start. So a June game is scored with profiles through May 31 — past data
  only, never same-day or future data. The GBM's weights are NOT refit.

* Known deviations from the production build (documented, not silent):
  - p_xera_gap (season ERA/xERA gap from pybaseball) is EXCLUDED
    (use_season_stats=False) in both train and test feature builds:
    full-season 2026 pitching stats would leak future games into the
    p_hr_vulnerability_score composite.
  - Weather for test games is the actual historical value from the
    Open-Meteo archive (7pm ET, same convention as build_features.py).
    In production you'd have a forecast; actual weather is a close,
    slightly optimistic stand-in for every model equally.

Evaluation units:

* Batted-ball level (the GBM's native unit): P(HR | batted ball), vs the
  actual pitcher of each batted ball.
* Player-game level (the betting unit): P(player hits >= 1 HR in game),
  one row per (game, batter with >= 1 PA), features vs the opposing
  STARTER. GBM per-game prob = 1 - (1 - p_bb)^E[batted balls per game],
  with E[BB/game] taken from pre-cutoff data (shrunk to league mean, 10
  pseudo-games) — the same conversion family the dashboard uses (it uses
  1-(1-p)^3.9). The structural model natively predicts this unit.

Market comparison: picks_history.csv rows inside the test window, joined
to actual outcomes from Statcast. book_implied is the Yes-price implied
probability and contains one-sided vig; we also report a de-vig
approximation (divide by 1.06, i.e. an assumed ~6% one-sided margin).
There is no No-price in the log, so a proper two-way de-vig is impossible
— treat the raw implied number as an upper bound on the market's true
probability. Caveat: the picks are model-selected top-10 lists, not a
random sample of the market.

Outputs: eval_results.json, calibration_curve.png, and a console report.
"""

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss

import build_features as bf
from structural_model import (StructuralModelV2, infer_game_slots,
                              pa_table_from_statcast, game_hr_probability,
                              MAX_LINEUP_SLOT)

warnings.filterwarnings("ignore", category=FutureWarning)

TRAIN_PATH = "homerun_data_all.csv"
HOLDOUT_PATH = "holdout_2026.csv"
WEATHER_CACHE = "holdout_weather.csv"
TRAIN_CUTOFF = pd.Timestamp("2026-03-23")

# Walk-forward segments: (profile cutoff, segment end). Games in
# [cutoff, end) are scored with profiles built from data < cutoff.
SEGMENTS = [
    (pd.Timestamp("2026-03-23"), pd.Timestamp("2026-05-01")),
    (pd.Timestamp("2026-05-01"), pd.Timestamp("2026-06-01")),
    (pd.Timestamp("2026-06-01"), pd.Timestamp("2026-07-03")),
]

# Columns actually used anywhere in the feature/eval pipeline — loading
# only these keeps the 1.7GB of CSVs manageable in memory.
USECOLS = {
    "game_date", "game_pk", "at_bat_number", "pitch_number", "events",
    "description", "batter", "pitcher", "player_name", "stand", "p_throws",
    "home_team", "away_team", "inning_topbot", "launch_speed", "launch_angle",
    "launch_speed_angle", "bb_type", "hc_x", "plate_x", "plate_z", "sz_bot",
    "sz_top", "pfx_x", "release_pos_x", "release_pos_z", "release_extension",
    "release_spin_rate", "pitch_type", "delta_run_exp",
    "estimated_ba_using_speedangle", "estimated_slg_using_speedangle",
    "estimated_woba_using_speedangle",
}

# E[batted balls per game] regression (mirrors structural E[PA] logic).
EBB_PRIOR_GAMES = 10.0

# De-vig approximation for the one-sided Yes price (assumed ~6% margin).
ONE_SIDED_MARGIN = 1.06

# Betting-sim thresholds: bet only when v2 prob exceeds the de-vig market
# prob by at least this many probability points (flat 1-unit stakes).
BET_EDGE_THRESHOLDS = [0.01, 0.02, 0.03, 0.05]

EPS = 1e-6


def load_pitches(path):
    df = pd.read_csv(path, usecols=lambda c: c in USECOLS, low_memory=False)
    df["game_date"] = pd.to_datetime(df["game_date"])
    return df


# ── Weather ───────────────────────────────────────────────────────────────

def fetch_holdout_weather(games, cache_path=WEATHER_CACHE):
    """Historical 7pm-ET weather for (game_date, home_team) pairs, one
    Open-Meteo archive range request per ballpark, cached to CSV."""
    games = games.drop_duplicates().copy()
    if Path(cache_path).exists():
        cache = pd.read_csv(cache_path, parse_dates=["game_date"])
    else:
        cache = pd.DataFrame(columns=["game_date", "home_team", *bf.WEATHER_FIELDS])
        cache["game_date"] = pd.to_datetime(cache["game_date"])

    have = set(zip(cache["game_date"], cache["home_team"]))
    missing = games[[not ((d, t) in have) for d, t in zip(games["game_date"], games["home_team"])]]

    new_rows = []
    for team, grp in missing.groupby("home_team"):
        coords = bf.ballpark_coords.get(team)
        if coords is None:
            continue
        lat, lon = coords
        start, end = grp["game_date"].min(), grp["game_date"].max()
        url = (
            f"https://archive-api.open-meteo.com/v1/archive"
            f"?latitude={lat}&longitude={lon}"
            f"&start_date={start:%Y-%m-%d}&end_date={end:%Y-%m-%d}"
            f"&hourly=temperature_2m,relativehumidity_2m,windspeed_10m,winddirection_10m"
            f"&temperature_unit=fahrenheit&windspeed_unit=mph&timezone=America/New_York"
        )
        try:
            d = requests.get(url, timeout=30).json()["hourly"]
        except Exception as e:
            print(f"  weather fetch failed for {team}: {e}")
            continue
        times = pd.to_datetime(d["time"])
        idx_by_date = {}
        for i, t in enumerate(times):
            if t.hour == 19:
                idx_by_date[t.normalize()] = i
        for gd in grp["game_date"]:
            i = idx_by_date.get(gd)
            if i is None:
                continue
            new_rows.append({
                "game_date": gd, "home_team": team,
                "temp_f": d["temperature_2m"][i],
                "humidity": d["relativehumidity_2m"][i],
                "wind_speed": d["windspeed_10m"][i],
                "wind_dir": d["winddirection_10m"][i],
            })
    if new_rows:
        cache = pd.concat([cache, pd.DataFrame(new_rows)], ignore_index=True)
        cache.to_csv(cache_path, index=False)
    return cache


def load_train_weather():
    """Weather for the training window, from the enriched CSV's cache."""
    path = Path("homerun_data_enriched.csv")
    if not path.exists():
        return pd.DataFrame(columns=["game_date", "home_team", *bf.WEATHER_FIELDS])
    cached = pd.read_csv(
        path,
        usecols=lambda c: c in {"game_date", "home_team", *bf.WEATHER_FIELDS},
        on_bad_lines="skip",
        low_memory=False,
    )
    cached["game_date"] = pd.to_datetime(cached["game_date"], errors="coerce", format="mixed")
    return cached.dropna(subset=["game_date", "home_team"]).drop_duplicates(["game_date", "home_team"])


# ── Player-game construction ──────────────────────────────────────────────

def build_player_games(pitches):
    """One row per (game_pk, batter with >= 1 PA): outcome + pre-game-known
    context (home team, opposing starter and his hand, batter side)."""
    first = pitches.sort_values(["game_pk", "at_bat_number", "pitch_number"])
    # Starters: pitcher of the first Top-inning pitch is the HOME starter,
    # first Bot-inning pitch the AWAY starter.
    starters = {}
    for side, col in [("Top", "home_starter"), ("Bot", "away_starter")]:
        s = first[first["inning_topbot"] == side].groupby("game_pk").head(1)
        starters[col] = s.set_index("game_pk")[["pitcher", "player_name", "p_throws"]]

    pa = pa_table_from_statcast(pitches)
    # Batter's side of the inning tells us his team half; modal stand.
    tb = pitches.groupby(["game_pk", "batter"]).agg(
        top_share=("inning_topbot", lambda s: float((s == "Top").mean())),
        stand=("stand", lambda s: s.mode().iat[0]),
        game_date=("game_date", "first"),
        home_team=("home_team", "first"),
    ).reset_index()
    outcomes = pa.groupby(["game_pk", "batter"]).agg(
        is_hr_game=("is_hr", "max"), n_pa=("is_hr", "count")
    ).reset_index()
    pg = tb.merge(outcomes, on=["game_pk", "batter"], how="inner")
    pg["is_away"] = pg["top_share"] >= 0.5

    hs = starters["home_starter"].rename(columns={
        "pitcher": "hs_id", "player_name": "hs_name", "p_throws": "hs_hand"})
    as_ = starters["away_starter"].rename(columns={
        "pitcher": "as_id", "player_name": "as_name", "p_throws": "as_hand"})
    pg = pg.merge(hs, on="game_pk", how="left").merge(as_, on="game_pk", how="left")
    # Away batters face the home starter and vice versa.
    pg["opp_starter_id"] = np.where(pg["is_away"], pg["hs_id"], pg["as_id"])
    pg["opp_starter_name"] = np.where(pg["is_away"], pg["hs_name"], pg["as_name"])
    pg["opp_starter_hand"] = np.where(pg["is_away"], pg["hs_hand"], pg["as_hand"])
    return pg[["game_pk", "game_date", "batter", "stand", "home_team",
               "opp_starter_id", "opp_starter_name", "opp_starter_hand",
               "is_hr_game", "n_pa"]]


def expected_bb_table(all_pitches, batted):
    """Per-batter expected batted balls per game from pre-cutoff data,
    regressed to the league mean with EBB_PRIOR_GAMES pseudo-games."""
    bb = batted.groupby("batter").size().rename("bb")
    games = all_pitches.groupby("batter")["game_pk"].nunique().rename("games")
    tbl = pd.concat([bb, games], axis=1).fillna(0.0)
    league = tbl["bb"].sum() / max(tbl["games"].sum(), 1.0)
    tbl["ebb"] = (tbl["bb"] + EBB_PRIOR_GAMES * league) / (tbl["games"] + EBB_PRIOR_GAMES)
    return tbl["ebb"], league


# ── Metrics ───────────────────────────────────────────────────────────────

def metric_row(y, p):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    y = np.asarray(y)
    if len(np.unique(p)) > 1 and 0 < y.mean() < 1:
        auc = roc_auc_score(y, p)
    else:
        auc = 0.5  # constant predictor
    return {
        "n": int(len(y)),
        "positives": int(y.sum()),
        "auc": round(float(auc), 4),
        "brier": round(float(brier_score_loss(y, p)), 5),
        "log_loss": round(float(log_loss(y, p)), 5),
        "mean_pred": round(float(p.mean()), 4),
        "actual_rate": round(float(y.mean()), 4),
    }


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print("Loading data...")
    train_raw = load_pitches(TRAIN_PATH)
    train_raw = train_raw[train_raw["game_date"] < TRAIN_CUTOFF]
    holdout_raw = load_pitches(HOLDOUT_PATH)
    holdout_raw = holdout_raw[holdout_raw["game_date"] >= TRAIN_CUTOFF]
    print(f"  train pitches: {len(train_raw):,} ({train_raw['game_date'].min():%Y-%m-%d} .. {train_raw['game_date'].max():%Y-%m-%d})")
    print(f"  holdout pitches: {len(holdout_raw):,} ({holdout_raw['game_date'].min():%Y-%m-%d} .. {holdout_raw['game_date'].max():%Y-%m-%d})")

    features = pd.read_csv("model_features.csv")["feature"].tolist()
    team_categories = sorted(train_raw["home_team"].dropna().unique())

    # ── Train the GBM exactly as train_model.py does, minus leakage ──
    print("\nBuilding training features (as of cutoff)...")
    final_train, _, _, _ = bf.build_features(
        train_raw, use_season_stats=False, team_categories=team_categories, verbose=False)
    train_weather = load_train_weather()
    final_train = final_train.merge(train_weather, on=["game_date", "home_team"], how="left")
    features = [f for f in features if f in final_train.columns]
    print(f"  {len(final_train):,} batted balls, {len(features)} features")

    X_train = final_train[features].fillna(0)
    y_train = final_train["is_homerun"]
    bb_base_rate = float(y_train.mean())

    base_model = GradientBoostingClassifier(
        n_estimators=180, max_depth=4, learning_rate=0.05,
        min_samples_leaf=30, subsample=0.8, random_state=42,
    )
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", CalibratedClassifierCV(base_model, method="sigmoid", cv=3)),
    ])
    print("Training GBM on pre-cutoff data...")
    pipeline.fit(X_train, y_train)
    print(f"  done ({time.time()-t0:.0f}s elapsed)")

    # Weather for the holdout window.
    print("\nFetching holdout weather (Open-Meteo archive)...")
    holdout_games = holdout_raw[["game_date", "home_team"]].drop_duplicates()
    test_weather = fetch_holdout_weather(holdout_games)
    print(f"  weather rows: {len(test_weather):,}")

    master = pd.concat([train_raw, holdout_raw], ignore_index=True)
    del train_raw

    # Player-game frame for the whole test window (outcomes + context).
    test_pg_all = build_player_games(holdout_raw)

    # Batting-order slot for each test player-game, inferred from the PA
    # ordering of that game. This stands in for the posted pre-game lineup
    # (identical information for starters); slots > 9 are mid-game subs
    # whose slot would NOT be on a posted lineup, so they become unknown.
    slot_map = infer_game_slots(pa_table_from_statcast(holdout_raw))
    test_pg_all = test_pg_all.merge(slot_map, on=["game_pk", "batter"], how="left")
    test_pg_all.loc[test_pg_all["slot"] > MAX_LINEUP_SLOT, "slot"] = np.nan
    print(f"  player-games in holdout: {len(test_pg_all):,} "
          f"({test_pg_all['slot'].notna().mean():.0%} with a starting-lineup slot)")

    # ── Walk-forward scoring ─────────────────────────────────────────
    bb_frames = []      # batted-ball level predictions
    pg_frames = []      # player-game level predictions

    for cutoff, seg_end in SEGMENTS:
        seg_t = time.time()
        print(f"\nSegment {cutoff:%Y-%m-%d} .. {seg_end:%Y-%m-%d} (profiles from data < {cutoff:%Y-%m-%d})")
        pre = master[master["game_date"] < cutoff]
        all_pre, batted_pre = bf.prepare_pitch_data(pre)
        hitter, pitcher, matchup, _ = bf.build_profiles(
            all_pre, batted_pre, use_season_stats=False, verbose=False)

        # Structural model tables from strictly pre-cutoff PAs (v2 tables —
        # slot PA means and park factors — come from the same pre-cutoff
        # window; v1 predictions are unchanged by the subclass).
        struct = StructuralModelV2(pa_table_from_statcast(all_pre))
        ebb, league_bb_pg = expected_bb_table(all_pre, batted_pre)

        # Pre-cutoff player-game base rate (league P(HR in game | >=1 PA)).
        pre_pg = pa_table_from_statcast(all_pre).groupby(["game_pk", "batter"])["is_hr"].max()
        pg_base_rate = float(pre_pg.mean())

        seg_pitches = holdout_raw[(holdout_raw["game_date"] >= cutoff)
                                  & (holdout_raw["game_date"] < seg_end)]
        if seg_pitches.empty:
            continue

        # --- batted-ball level (actual pitcher) ---
        _, seg_batted = bf.prepare_pitch_data(seg_pitches)
        seg_bb = bf.assemble_matchup_frame(
            seg_batted, hitter, pitcher, matchup, team_categories=team_categories)
        seg_bb = seg_bb.merge(test_weather, on=["game_date", "home_team"], how="left")
        p_bb = pipeline.predict_proba(seg_bb[features].fillna(0))[:, 1]
        bb_frames.append(pd.DataFrame({
            "game_date": seg_bb["game_date"], "y": seg_bb["is_homerun"].values,
            "gbm": p_bb, "base": bb_base_rate,
        }))

        # --- player-game level (vs opposing starter) ---
        pg = test_pg_all[(test_pg_all["game_date"] >= cutoff)
                         & (test_pg_all["game_date"] < seg_end)].copy()
        rows = pg.rename(columns={
            "opp_starter_name": "player_name", "opp_starter_hand": "p_throws"})[
            ["batter", "player_name", "stand", "p_throws", "home_team", "game_date"]]
        frame = bf.assemble_matchup_frame(
            rows, hitter, pitcher, matchup, team_categories=team_categories)
        frame = frame.merge(test_weather, on=["game_date", "home_team"], how="left")
        p_bb_game = pipeline.predict_proba(frame[features].fillna(0))[:, 1]
        n_bb = pg["batter"].map(ebb).fillna(league_bb_pg).values
        pg["gbm"] = game_hr_probability(p_bb_game, n_bb)

        pg["structural"] = [
            struct.predict(b, opposing_pitcher=pid, pitcher_hand=hand)
            for b, pid, hand in zip(pg["batter"], pg["opp_starter_id"], pg["opp_starter_hand"])
        ]

        # Ablation ladder: v1 + slots, v1 + parks, v1 + both (= v2).
        def _score_variant(use_slots, use_parks):
            return [
                struct.predict_v2(
                    b, opposing_pitcher=pid, pitcher_hand=hand,
                    slot=int(s) if pd.notna(s) else None,
                    park=park, stand=st,
                    use_slots=use_slots, use_parks=use_parks)
                for b, pid, hand, s, park, st in zip(
                    pg["batter"], pg["opp_starter_id"], pg["opp_starter_hand"],
                    pg["slot"], pg["home_team"], pg["stand"])
            ]
        pg["structural_slots"] = _score_variant(True, False)
        pg["structural_parks"] = _score_variant(False, True)
        pg["structural_v2"] = _score_variant(True, True)
        pg["base"] = pg_base_rate
        pg_frames.append(pg)
        print(f"  scored {len(seg_bb):,} batted balls, {len(pg):,} player-games ({time.time()-seg_t:.0f}s)")

    bb_all = pd.concat(bb_frames, ignore_index=True)
    pg_all = pd.concat(pg_frames, ignore_index=True)

    # ── Market comparison on overlapping picks ───────────────────────
    picks = pd.read_csv("picks_history.csv", parse_dates=["date"])
    picks = picks.dropna(subset=["book_implied", "batter_id"])
    # Decimal odds for the betting sim (mean across books when a player
    # was logged more than once on a date).
    picks["decimal_odds"] = picks["book_odds"].apply(
        lambda o: (1 + o / 100.0 if o > 0 else 1 + 100.0 / abs(o)) if pd.notna(o) else np.nan)
    picks = picks.groupby(["date", "batter_id"], as_index=False).agg(
        book_implied=("book_implied", "mean"), decimal_odds=("decimal_odds", "mean"))
    picks["book_implied"] = picks["book_implied"] / 100.0
    merged = picks.merge(
        pg_all, left_on=["date", "batter_id"], right_on=["game_date", "batter"], how="inner")
    merged["devig"] = merged["book_implied"] / ONE_SIDED_MARGIN

    # ── Betting simulation on the picks subset ───────────────────────
    # Flat 1-unit bet whenever the v2 probability exceeds the de-vig
    # market probability by >= threshold, settled at the recorded odds.
    betting_sim = {"description": (
        "Flat 1u on Yes when structural_v2 prob - devig market prob >= threshold, "
        "settled at recorded book odds. CAVEAT: n=%d model-selected picks, not a "
        "random market sample; treat as illustrative only." % len(merged))}
    bettable = merged.dropna(subset=["decimal_odds"])
    for thr in BET_EDGE_THRESHOLDS:
        sel = bettable[bettable["structural_v2"] - bettable["devig"] >= thr]
        n = len(sel)
        wins = int(sel["is_hr_game"].sum())
        pnl = float((sel["is_hr_game"] * (sel["decimal_odds"] - 1.0)
                     - (1 - sel["is_hr_game"])).sum())
        betting_sim[f"edge>={thr:.2f}"] = {
            "bets": n, "wins": wins, "losses": n - wins,
            "pnl_units": round(pnl, 2),
            "roi": round(pnl / n, 4) if n else None,
        }

    # ── Metrics ──────────────────────────────────────────────────────
    results = {
        "config": {
            "train_window": "2025-03-27..2026-03-22",
            "test_window": f"{TRAIN_CUTOFF:%Y-%m-%d}..{pg_all['game_date'].max():%Y-%m-%d}",
            "segments": [f"{c:%Y-%m-%d}..{e:%Y-%m-%d}" for c, e in SEGMENTS],
            "n_features": len(features),
            "excluded_features": "p_xera_gap term of p_hr_vulnerability_score (season-level stats would leak future games)",
            "devig_assumption": f"one-sided margin {ONE_SIDED_MARGIN}",
        },
        "batted_ball_level": {
            "gbm": metric_row(bb_all["y"], bb_all["gbm"]),
            "base_rate": metric_row(bb_all["y"], bb_all["base"]),
        },
        "player_game_level": {
            "gbm": metric_row(pg_all["is_hr_game"], pg_all["gbm"]),
            "structural": metric_row(pg_all["is_hr_game"], pg_all["structural"]),
            "base_rate": metric_row(pg_all["is_hr_game"], pg_all["base"]),
        },
        "picks_subset (n=%d)" % len(merged): {
            "market_implied_raw": metric_row(merged["is_hr_game"], merged["book_implied"]),
            "market_implied_devig": metric_row(merged["is_hr_game"], merged["devig"]),
            "gbm": metric_row(merged["is_hr_game"], merged["gbm"]),
            "structural": metric_row(merged["is_hr_game"], merged["structural"]),
            "base_rate": metric_row(merged["is_hr_game"], merged["base"]),
        } if len(merged) > 0 else {},
        "structural_v2": {
            "config": {
                "slot_epa": "E[PA] = 0.7 * league PA/game for the batter's slot that game "
                            "+ 0.3 * trailing regressed PA/game; slot constants from the "
                            "pre-cutoff window only; subs (slot>9) fall back to v1 E[PA]",
                "park_factors": "per-park HR/PA by batter hand, shrunk toward 1.0 with "
                                "2000 pseudo-PA, clipped to [0.80, 1.25], pre-cutoff only",
            },
            "player_game_ablation": {
                "v1_structural": metric_row(pg_all["is_hr_game"], pg_all["structural"]),
                "v1_plus_slots": metric_row(pg_all["is_hr_game"], pg_all["structural_slots"]),
                "v1_plus_parks": metric_row(pg_all["is_hr_game"], pg_all["structural_parks"]),
                "v2_slots_plus_parks": metric_row(pg_all["is_hr_game"], pg_all["structural_v2"]),
            },
            "picks_subset_ablation (n=%d)" % len(merged): {
                "market_implied_devig": metric_row(merged["is_hr_game"], merged["devig"]),
                "v1_structural": metric_row(merged["is_hr_game"], merged["structural"]),
                "v1_plus_slots": metric_row(merged["is_hr_game"], merged["structural_slots"]),
                "v1_plus_parks": metric_row(merged["is_hr_game"], merged["structural_parks"]),
                "v2_slots_plus_parks": metric_row(merged["is_hr_game"], merged["structural_v2"]),
            } if len(merged) > 0 else {},
            "betting_simulation": betting_sim,
        },
    }

    with open("eval_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n" + json.dumps(results, indent=2))

    # ── Calibration plot ─────────────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    frac, mean_pred = calibration_curve(bb_all["y"], bb_all["gbm"], n_bins=10, strategy="quantile")
    ax.plot(mean_pred, frac, "o-", label="GBM")
    lim = max(mean_pred.max(), frac.max()) * 1.1
    ax.plot([0, lim], [0, lim], "k--", lw=1, label="perfect")
    ax.set_title("Batted-ball level: P(HR | batted ball)")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed HR frequency")
    ax.legend()

    ax = axes[1]
    for name, col, style in [("GBM", "gbm", "o-"), ("Structural v1", "structural", "s-"),
                             ("Structural v2 (slots+parks)", "structural_v2", "^-")]:
        frac, mean_pred = calibration_curve(pg_all["is_hr_game"], pg_all[col], n_bins=10, strategy="quantile")
        ax.plot(mean_pred, frac, style, label=name)
    lim = 0.15
    ax.plot([0, lim], [0, lim], "k--", lw=1, label="perfect")
    ax.set_title("Player-game level: P(>=1 HR in game)")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed HR-game frequency")
    ax.legend()

    fig.suptitle(f"Holdout calibration, {TRAIN_CUTOFF:%Y-%m-%d} .. {pg_all['game_date'].max():%Y-%m-%d} (walk-forward, leakage-free)")
    fig.tight_layout()
    fig.savefig("calibration_curve.png", dpi=140)
    print("\nSaved: eval_results.json, calibration_curve.png")
    print(f"Total runtime: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
