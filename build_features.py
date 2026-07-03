import pandas as pd
import numpy as np
import requests
from pathlib import Path
from datetime import datetime

try:
    from pybaseball import pitching_stats
except Exception:
    pitching_stats = None

PITCH_LABELS = [
    ("FF", "4seam"),
    ("SI", "sinker"),
    ("SL", "slider"),
    ("ST", "sweeper"),
    ("CH", "change"),
    ("CU", "curve"),
    ("FC", "cutter"),
]

FEATURE_TOGGLES = {
    "pitcher_vulnerability": True,
    "platoon_splits": True,
    "pitcher_batter_history": True,
}

PA_EVENT_WEIGHTS = {
    "walk": 0.69,
    "intent_walk": 0.00,
    "hit_by_pitch": 0.72,
    "single": 0.88,
    "double": 1.24,
    "triple": 1.56,
    "home_run": 1.95,
}

OFFICIAL_AB_EVENTS = {
    "single", "double", "triple", "home_run", "field_out", "force_out",
    "grounded_into_double_play", "fielders_choice", "fielders_choice_out",
    "double_play", "triple_play", "strikeout", "strikeout_double_play",
    "other_out", "reached_on_error", "field_error",
}

HIT_EVENTS = {"single", "double", "triple", "home_run"}

STRIKEOUT_EVENTS = {"strikeout", "strikeout_double_play"}

GROUND_BALL_TYPES = {"ground_ball"}

SECONDARY_PITCH_TYPES = {"SL", "ST", "CH"}


def hitter_contact_score(df_like, ev_col):
    ev_bonus = ((df_like[ev_col] - 88.0) / 8.0).clip(lower=0)
    return (
        df_like["barrel_pct"] * 0.35
        + df_like["sweet_spot_pct"] * 0.25
        + df_like["hard_hit_pct"] * 0.20
        + df_like["pull_air_pct"] * 0.20
        + ev_bonus * 0.10
    )


def hitter_lift_score(df_like):
    launch_window = ((16.0 - (df_like["launch_angle"] - 18.0).abs()) / 16.0).clip(lower=0)
    return (
        df_like["barrel_pct"] * 0.45
        + df_like["sweet_spot_pct"] * 0.30
        + df_like["pull_air_pct"] * 0.15
        + launch_window * 0.10
    )


def pitcher_contact_risk(df_like, ev_col):
    ev_risk = ((df_like[ev_col] - 88.0) / 8.0).clip(lower=0)
    return (
        df_like["barrel_pct_allowed"] * 0.30
        + df_like["hard_hit_pct_allowed"] * 0.20
        + df_like["sweet_spot_pct_allowed"] * 0.20
        + df_like["pull_air_pct_allowed"] * 0.15
        + df_like["hr_rate_allowed"] * 0.15
        + ev_risk * 0.10
    )


def pitcher_lift_risk(df_like):
    launch_window = ((16.0 - (df_like["launch_angle_allowed"] - 18.0).abs()) / 16.0).clip(lower=0)
    return (
        df_like["sweet_spot_pct_allowed"] * 0.35
        + df_like["pull_air_pct_allowed"] * 0.25
        + launch_window * 0.20
        + df_like["barrel_pct_allowed"] * 0.20
    )


def aggregate_hitter_profile(batted_df, prefix):
    if batted_df.empty:
        return pd.DataFrame(columns=["batter"])
    out = batted_df.groupby("batter").agg(
        exit_velo=("launch_speed", "mean"),
        barrel_pct=("barrel", "mean"),
        launch_angle=("launch_angle", "mean"),
        sweet_spot_pct=("sweet_spot", "mean"),
        hard_hit_pct=("hard_hit", "mean"),
        pull_air_pct=("pull_air", "mean"),
        hr_rate=("is_homerun", "mean"),
        n_batted=("launch_speed", "count"),
    ).reset_index()
    scores = out.rename(columns={
        "barrel_pct": "barrel_pct",
        "sweet_spot_pct": "sweet_spot_pct",
        "hard_hit_pct": "hard_hit_pct",
        "pull_air_pct": "pull_air_pct",
        "launch_angle": "launch_angle",
        "exit_velo": "exit_velo",
    })
    out["hr_contact_score"] = hitter_contact_score(scores, "exit_velo")
    out["lifted_power_score"] = hitter_lift_score(scores)
    rename_map = {"batter": "batter"}
    for col in out.columns:
        if col != "batter":
            rename_map[col] = f"{prefix}{col}"
    return out.rename(columns=rename_map)


def aggregate_pitcher_profile(batted_df, prefix):
    if batted_df.empty:
        return pd.DataFrame(columns=["player_name"])
    out = batted_df.groupby("player_name").agg(
        exit_velo_allowed=("launch_speed", "mean"),
        hard_hit_pct_allowed=("hard_hit", "mean"),
        barrel_pct_allowed=("barrel", "mean"),
        launch_angle_allowed=("launch_angle", "mean"),
        sweet_spot_pct_allowed=("sweet_spot", "mean"),
        pull_air_pct_allowed=("pull_air", "mean"),
        hr_rate_allowed=("is_homerun", "mean"),
        n_faced=("launch_speed", "count"),
    ).reset_index()
    scores = out.rename(columns={
        "barrel_pct_allowed": "barrel_pct_allowed",
        "hard_hit_pct_allowed": "hard_hit_pct_allowed",
        "sweet_spot_pct_allowed": "sweet_spot_pct_allowed",
        "pull_air_pct_allowed": "pull_air_pct_allowed",
        "hr_rate_allowed": "hr_rate_allowed",
        "exit_velo_allowed": "exit_velo_allowed",
        "launch_angle_allowed": "launch_angle_allowed",
    })
    out["hr_contact_risk"] = pitcher_contact_risk(scores, "exit_velo_allowed")
    out["lift_damage_risk"] = pitcher_lift_risk(scores)
    rename_map = {"player_name": "player_name"}
    for col in out.columns:
        if col != "player_name":
            rename_map[col] = f"{prefix}{col}"
    return out.rename(columns=rename_map)


def normalize_name(name):
    if not isinstance(name, str):
        return ""
    return " ".join(name.replace(",", " ").split()).strip().lower()


def build_pa_df(all_pitches: pd.DataFrame) -> pd.DataFrame:
    pa_cols = [c for c in ["game_pk", "at_bat_number", "pitch_number", "pitcher", "batter", "player_name"] if c in all_pitches.columns]
    if not {"game_pk", "at_bat_number", "batter"}.issubset(all_pitches.columns):
        return pd.DataFrame()

    sort_cols = [c for c in ["game_pk", "at_bat_number", "pitch_number"] if c in all_pitches.columns]
    pa_df = all_pitches.sort_values(sort_cols).groupby(["game_pk", "at_bat_number", "batter"], as_index=False).tail(1).copy()
    pa_df["is_pa"] = 1
    pa_df["is_ab"] = pa_df["events"].isin(OFFICIAL_AB_EVENTS).astype(int)
    pa_df["is_hit"] = pa_df["events"].isin(HIT_EVENTS).astype(int)
    pa_df["is_hr"] = (pa_df["events"] == "home_run").astype(int)
    pa_df["is_so"] = pa_df["events"].isin(STRIKEOUT_EVENTS).astype(int)
    pa_df["tb"] = np.select(
        [
            pa_df["events"] == "single",
            pa_df["events"] == "double",
            pa_df["events"] == "triple",
            pa_df["events"] == "home_run",
        ],
        [1, 2, 3, 4],
        default=0,
    )
    pa_df["woba_num"] = pa_df["events"].map(PA_EVENT_WEIGHTS).fillna(0.0)
    return pa_df


def add_handedness_split_features(hitter_df: pd.DataFrame, pa_df: pd.DataFrame, batted_df: pd.DataFrame) -> pd.DataFrame:
    if pa_df.empty or "p_throws" not in pa_df.columns:
        return hitter_df

    for hand, label in [("R", "rhp"), ("L", "lhp")]:
        pa_split = pa_df[pa_df["p_throws"] == hand].groupby("batter").agg(
            pa=("is_pa", "sum"),
            ab=("is_ab", "sum"),
            hits=("is_hit", "sum"),
            hr=("is_hr", "sum"),
            so=("is_so", "sum"),
            total_bases=("tb", "sum"),
            woba_num=("woba_num", "sum"),
        ).reset_index()
        pa_split[f"h_woba_vs_{label}"] = pa_split["woba_num"] / pa_split["pa"].replace(0, np.nan)
        batting_avg = pa_split["hits"] / pa_split["ab"].replace(0, np.nan)
        slug = pa_split["total_bases"] / pa_split["ab"].replace(0, np.nan)
        pa_split[f"h_iso_vs_{label}"] = slug - batting_avg
        pa_split[f"h_hr_pa_vs_{label}"] = pa_split["hr"] / pa_split["pa"].replace(0, np.nan)
        pa_split[f"h_k_pct_vs_{label}"] = pa_split["so"] / pa_split["pa"].replace(0, np.nan)
        hitter_df = hitter_df.merge(
            pa_split[["batter", f"h_woba_vs_{label}", f"h_iso_vs_{label}", f"h_hr_pa_vs_{label}", f"h_k_pct_vs_{label}"]],
            on="batter",
            how="left",
        )

    return hitter_df


def add_pitcher_vulnerability_features(pitcher_df: pd.DataFrame, all_pitches_df: pd.DataFrame, batted_df: pd.DataFrame, pa_df: pd.DataFrame) -> pd.DataFrame:
    if all_pitches_df.empty:
        return pitcher_df

    # Pitch-type usage and per-pitch contact quality.
    for pt, label in PITCH_LABELS:
        pt_all = all_pitches_df[all_pitches_df["pitch_type"] == pt]
        if len(pt_all) == 0:
            continue

        usage = pt_all.groupby("player_name").size().reset_index(name=f"p_{label}_pitch_count")
        usage[f"p_{label}_usage_pct"] = usage[f"p_{label}_pitch_count"] / all_pitches_df.groupby("player_name").size().reindex(usage["player_name"]).to_numpy()
        pitcher_df = pitcher_df.merge(usage[["player_name", f"p_{label}_usage_pct"]], on="player_name", how="left")

        pt_batted = batted_df[batted_df["pitch_type"] == pt]
        if len(pt_batted) == 0:
            continue
        agg_map = {
            f"p_{label}_hard_hit_pct_allowed": ("hard_hit", "mean"),
            f"p_{label}_rv100": ("delta_run_exp", "mean"),
        }
        if "estimated_ba_using_speedangle" in pt_batted.columns:
            agg_map[f"p_{label}_xba_allowed"] = ("estimated_ba_using_speedangle", "mean")
        if "estimated_slg_using_speedangle" in pt_batted.columns:
            agg_map[f"p_{label}_xslg_allowed"] = ("estimated_slg_using_speedangle", "mean")

        pt_stats = pt_batted.groupby("player_name").agg(**agg_map).reset_index()
        if f"p_{label}_rv100" in pt_stats.columns:
            pt_stats[f"p_{label}_rv100"] = pt_stats[f"p_{label}_rv100"] * 100.0
        pitcher_df = pitcher_df.merge(pt_stats, on="player_name", how="left")

    # Extension percentile.
    if "release_extension" in all_pitches_df.columns:
        ext = all_pitches_df.groupby("player_name")["release_extension"].mean().reset_index(name="p_release_extension")
        ext["p_extension_pctile"] = ext["p_release_extension"].rank(pct=True) * 100.0
        ext["p_low_extension_flag"] = (ext["p_extension_pctile"] < 20).astype(int)
        pitcher_df = pitcher_df.merge(ext, on="player_name", how="left")

    # Ground-ball rate.
    if "bb_type" in batted_df.columns:
        gb = batted_df.groupby("player_name").agg(
            p_gb_pct=("bb_type", lambda s: float(np.mean(s.isin(GROUND_BALL_TYPES))))
        ).reset_index()
        gb["p_gb_pctile"] = gb["p_gb_pct"].rank(pct=True) * 100.0
        gb["p_flyball_risk_flag"] = (gb["p_gb_pctile"] < 40).astype(int)
        pitcher_df = pitcher_df.merge(gb, on="player_name", how="left")

    # Put-away secondary usage.
    if "pitch_type" in all_pitches_df.columns:
        sec = all_pitches_df.assign(is_secondary=all_pitches_df["pitch_type"].isin(SECONDARY_PITCH_TYPES).astype(int))
        sec = sec.groupby("player_name").agg(
            p_secondary_usage_pct=("is_secondary", "mean")
        ).reset_index()
        sec["p_secondary_usage_risk_flag"] = (sec["p_secondary_usage_pct"] < 0.18).astype(int)
        pitcher_df = pitcher_df.merge(sec, on="player_name", how="left")

    # Optional seasonal ERA/xERA gap from pybaseball.
    if pitching_stats is not None:
        season_frames = []
        for year in [2025, datetime.now().year]:
            try:
                season = pitching_stats(year, qual=0)
                season["norm_name"] = season["Name"].map(normalize_name)
                available = {"norm_name"}
                for col in ["ERA", "xERA", "GB%", "GB/FB"]:
                    if col in season.columns:
                        available.add(col)
                season_frames.append(season[list(available)].drop_duplicates("norm_name"))
            except Exception:
                continue
        if season_frames:
            season = pd.concat(season_frames, ignore_index=True).drop_duplicates("norm_name", keep="last")
            season["p_xera_gap"] = season.get("xERA", np.nan) - season.get("ERA", np.nan)
            merge_df = pitcher_df.assign(norm_name=pitcher_df["player_name"].map(normalize_name))
            merge_df = merge_df.merge(season[["norm_name", "p_xera_gap"]], on="norm_name", how="left")
            pitcher_df = merge_df.drop(columns=["norm_name"])

    # Composite pitcher HR-risk score.
    risk_terms = []
    for label in ["4seam", "sinker"]:
        rv = pitcher_df.get(f"p_{label}_rv100")
        hh = pitcher_df.get(f"p_{label}_hard_hit_pct_allowed")
        xba = pitcher_df.get(f"p_{label}_xba_allowed")
        xslg = pitcher_df.get(f"p_{label}_xslg_allowed")
        if rv is not None:
            risk_terms.append(((rv.fillna(0.0) - 0.0) / 4.0) * 0.20)
        if hh is not None:
            risk_terms.append(((hh.fillna(0.34) - 0.34) / 0.10) * 0.15)
        if xba is not None:
            risk_terms.append(((xba.fillna(0.240) - 0.240) / 0.040) * 0.10)
        if xslg is not None:
            risk_terms.append(((xslg.fillna(0.390) - 0.390) / 0.080) * 0.15)

    if "p_low_extension_flag" in pitcher_df.columns:
        risk_terms.append(pitcher_df["p_low_extension_flag"].fillna(0.0) * 0.30)
    if "p_xera_gap" in pitcher_df.columns:
        risk_terms.append(pitcher_df["p_xera_gap"].fillna(0.0).clip(lower=-1.0, upper=2.5) * 0.22)
    if "p_flyball_risk_flag" in pitcher_df.columns:
        risk_terms.append(pitcher_df["p_flyball_risk_flag"].fillna(0.0) * 0.25)
    if "p_secondary_usage_risk_flag" in pitcher_df.columns:
        risk_terms.append(pitcher_df["p_secondary_usage_risk_flag"].fillna(0.0) * 0.22)
    if risk_terms:
        pitcher_df["p_hr_vulnerability_score"] = np.sum(risk_terms, axis=0)

    return pitcher_df

print("Loading raw data...")
df = pd.read_csv("homerun_data_all.csv").copy()
print("Columns available:", list(df.columns))
df["game_date"] = pd.to_datetime(df["game_date"])
df["is_homerun"] = (df["events"] == "home_run").astype(int)

all_pitches = df.copy()
batted = df[df["launch_speed"].notna() & df["launch_angle"].notna()].copy()

# ── Barrel flag ───────────────────────────────────────────────
# Newer Statcast uses launch_speed_angle (6 = barrel), older data has "barrel" column
if "barrel" not in batted.columns:
    if "launch_speed_angle" in batted.columns:
        batted["barrel"] = (batted["launch_speed_angle"] == 6).astype(int)
    else:
        batted["barrel"] = 0
else:
    batted["barrel"] = batted["barrel"].fillna(0).astype(int)

# Same for all_pitches
if "barrel" not in all_pitches.columns:
    if "launch_speed_angle" in all_pitches.columns:
        all_pitches["barrel"] = (all_pitches["launch_speed_angle"] == 6).astype(int)
    else:
        all_pitches["barrel"] = 0
else:
    all_pitches["barrel"] = all_pitches["barrel"].fillna(0).astype(int)

# ── Helper flags (all column-safe) ───────────────────────────
batted["hard_hit"] = (batted["launch_speed"] >= 95).astype(int)
batted["sweet_spot"] = batted["launch_angle"].between(8, 32, inclusive="both").astype(int)

# Air ball (fly ball or line drive)
if "bb_type" in batted.columns:
    batted["air_ball"] = batted["bb_type"].isin(["fly_ball", "line_drive"]).astype(int)
else:
    batted["air_ball"] = (batted["launch_angle"] > 10).astype(int)

# Pull direction
if "hc_x" in batted.columns and "stand" in batted.columns:
    batted["pull"] = 0
    mask_r = batted["stand"] == "R"
    mask_l = batted["stand"] == "L"
    batted.loc[mask_r, "pull"] = (batted.loc[mask_r, "hc_x"] < 128).astype(int)
    batted.loc[mask_l, "pull"] = (batted.loc[mask_l, "hc_x"] > 128).astype(int)
    batted["pull"] = batted["pull"].fillna(0).astype(int)
else:
    batted["pull"] = 0

batted["pull_air"] = ((batted["pull"] == 1) & (batted["air_ball"] == 1)).astype(int)

# In-zone flag
if all(c in all_pitches.columns for c in ["plate_x", "plate_z", "sz_bot", "sz_top"]):
    all_pitches["in_zone"] = (
        all_pitches["plate_x"].between(-0.83, 0.83) &
        (all_pitches["plate_z"] >= all_pitches["sz_bot"]) &
        (all_pitches["plate_z"] <= all_pitches["sz_top"])
    ).astype(int).fillna(0)
else:
    all_pitches["in_zone"] = 0

# Zone contact (swing + contact on in-zone pitches)
if "description" in all_pitches.columns:
    all_pitches["zone_contact"] = (
        (all_pitches["in_zone"] == 1) &
        all_pitches["description"].isin(["hit_into_play", "foul", "foul_tip"])
    ).astype(int)
else:
    all_pitches["zone_contact"] = 0

# Spin into barrel (horizontal movement toward hitter's barrel side)
if "pfx_x" in all_pitches.columns and "stand" in all_pitches.columns:
    all_pitches["spin_into_barrel"] = np.nan
    mask_r = all_pitches["stand"] == "R"
    mask_l = all_pitches["stand"] == "L"
    all_pitches.loc[mask_r, "spin_into_barrel"] = (all_pitches.loc[mask_r, "pfx_x"] < 0).astype(float)
    all_pitches.loc[mask_l, "spin_into_barrel"] = (all_pitches.loc[mask_l, "pfx_x"] > 0).astype(float)
else:
    all_pitches["spin_into_barrel"] = np.nan

# Arm angle from release point geometry
if "release_pos_z" in all_pitches.columns and "release_pos_x" in all_pitches.columns:
    all_pitches["arm_angle"] = np.degrees(np.arctan2(
        all_pitches["release_pos_z"].fillna(0),
        all_pitches["release_pos_x"].abs().fillna(1)
    ))
else:
    all_pitches["arm_angle"] = np.nan

pa_df = build_pa_df(all_pitches)

latest_game_date = df["game_date"].max()
season_2026_mask = df["game_date"] >= pd.Timestamp("2026-01-01")
recent_2026_start = latest_game_date - pd.Timedelta(days=45) if pd.notna(latest_game_date) else pd.Timestamp("2026-01-01")
recent_2026_mask = season_2026_mask & (df["game_date"] >= recent_2026_start)

batted_2026 = batted[season_2026_mask.loc[batted.index]].copy()
batted_recent = batted[recent_2026_mask.loc[batted.index]].copy()

# ── Hitter profiles ───────────────────────────────────────────
print("Building hitter profiles...")

hitter = batted.groupby("batter").agg(
    h_exit_velo    = ("launch_speed", "mean"),
    h_barrel_pct   = ("barrel",       "mean"),
    h_launch_angle = ("launch_angle", "mean"),
    h_sweet_spot_pct = ("sweet_spot", "mean"),
    h_hard_hit_pct = ("hard_hit",     "mean"),
    h_pull_air_pct = ("pull_air",     "mean"),
    h_hr_rate      = ("is_homerun",   "mean"),
    h_n_batted     = ("launch_speed", "count"),
).reset_index()

_hitter_scores = hitter.rename(columns={
    "h_barrel_pct": "barrel_pct",
    "h_sweet_spot_pct": "sweet_spot_pct",
    "h_hard_hit_pct": "hard_hit_pct",
    "h_pull_air_pct": "pull_air_pct",
    "h_launch_angle": "launch_angle",
    "h_exit_velo": "exit_velo",
})
hitter["h_hr_contact_score"] = hitter_contact_score(_hitter_scores, "exit_velo")
hitter["h_lifted_power_score"] = hitter_lift_score(_hitter_scores)

hitter = hitter.merge(aggregate_hitter_profile(batted_2026, "h_2026_"), on="batter", how="left")
hitter = hitter.merge(aggregate_hitter_profile(batted_recent, "h_recent_"), on="batter", how="left")

if "stand" in all_pitches.columns:
    batter_hand = all_pitches.groupby("batter").agg(
        h_batter_right=("stand", lambda s: float((s == "R").mean() >= 0.5))
    ).reset_index()
    hitter = hitter.merge(batter_hand, on="batter", how="left")

# Zone contact %
zc = all_pitches[all_pitches["in_zone"] == 1].groupby("batter").agg(
    zone_pitches  = ("in_zone",      "count"),
    zone_contacts = ("zone_contact", "sum"),
).reset_index()
zc["h_zone_contact_pct"] = zc["zone_contacts"] / zc["zone_pitches"]
hitter = hitter.merge(zc[["batter", "h_zone_contact_pct"]], on="batter", how="left")

# Performance vs each pitch type
for pt, label in [("FF","4seam"), ("SI","sinker"), ("SL","slider"),
                   ("CH","change"), ("CU","curve"),  ("FC","cutter")]:
    if "pitch_type" not in batted.columns:
        break
    sub = batted[batted["pitch_type"] == pt]
    if len(sub) == 0:
        continue
    stats = sub.groupby("batter").agg(
        hr_rate   = ("is_homerun",   "mean"),
        exit_velo = ("launch_speed", "mean"),
        pa        = ("is_homerun",   "count"),
    ).reset_index().rename(columns={
        "hr_rate":   f"h_hr_vs_{label}",
        "exit_velo": f"h_ev_vs_{label}",
        "pa":        f"h_pa_vs_{label}",
    })
    hitter = hitter.merge(stats, on="batter", how="left")

# Platoon splits (vs RHP and LHP)
for hand, label in [("R", "rhp"), ("L", "lhp")]:
    if "p_throws" not in batted.columns:
        break
    platoon = batted[batted["p_throws"] == hand].groupby("batter").agg(
        hr_rate   = ("is_homerun",   "mean"),
        exit_velo = ("launch_speed", "mean"),
        barrel_pct = ("barrel", "mean"),
        hard_hit_pct = ("hard_hit", "mean"),
        launch_angle = ("launch_angle", "mean"),
        sweet_spot_pct = ("sweet_spot", "mean"),
        pull_air_pct = ("pull_air", "mean"),
        n_batted = ("launch_speed", "count"),
    ).reset_index().rename(columns={
        "hr_rate":   f"h_hr_vs_{label}",
        "exit_velo": f"h_ev_vs_{label}",
        "barrel_pct": f"h_barrel_pct_vs_{label}",
        "hard_hit_pct": f"h_hard_hit_pct_vs_{label}",
        "launch_angle": f"h_launch_angle_vs_{label}",
        "sweet_spot_pct": f"h_sweet_spot_pct_vs_{label}",
        "pull_air_pct": f"h_pull_air_pct_vs_{label}",
        "n_batted": f"h_n_batted_vs_{label}",
    })
    hitter = hitter.merge(platoon, on="batter", how="left")

    if "estimated_woba_using_speedangle" in batted.columns:
        xw = batted[batted["p_throws"] == hand].groupby("batter")["estimated_woba_using_speedangle"].mean().reset_index().rename(
            columns={"estimated_woba_using_speedangle": f"h_xwoba_vs_{label}"}
        )
        hitter = hitter.merge(xw, on="batter", how="left")

for label in ["rhp", "lhp"]:
    required = [
        f"h_barrel_pct_vs_{label}",
        f"h_sweet_spot_pct_vs_{label}",
        f"h_hard_hit_pct_vs_{label}",
        f"h_pull_air_pct_vs_{label}",
        f"h_ev_vs_{label}",
        f"h_launch_angle_vs_{label}",
    ]
    if all(col in hitter.columns for col in required):
        split_scores = hitter.rename(columns={
            f"h_barrel_pct_vs_{label}": "barrel_pct",
            f"h_sweet_spot_pct_vs_{label}": "sweet_spot_pct",
            f"h_hard_hit_pct_vs_{label}": "hard_hit_pct",
            f"h_pull_air_pct_vs_{label}": "pull_air_pct",
            f"h_ev_vs_{label}": "exit_velo",
            f"h_launch_angle_vs_{label}": "launch_angle",
        })
        hitter[f"h_hr_contact_score_vs_{label}"] = hitter_contact_score(split_scores, "exit_velo")
        hitter[f"h_lifted_power_score_vs_{label}"] = hitter_lift_score(split_scores)

# xwOBA vs each pitch type (quality of contact metric)
for pt, label in [("FF","4seam"), ("SI","sinker"), ("SL","slider"),
                   ("CH","change"), ("CU","curve"),  ("FC","cutter")]:
    if "pitch_type" not in batted.columns:
        break
    if "estimated_woba_using_speedangle" not in batted.columns:
        break
    sub = batted[batted["pitch_type"] == pt]
    if len(sub) == 0:
        continue
    rv = sub.groupby("batter")["estimated_woba_using_speedangle"].mean().reset_index().rename(
        columns={"estimated_woba_using_speedangle": f"h_xwoba_vs_{label}"})
    hitter = hitter.merge(rv, on="batter", how="left")

if FEATURE_TOGGLES["platoon_splits"]:
    hitter = add_handedness_split_features(hitter, pa_df, batted)

print(f"  Hitter profiles: {len(hitter)} players, {len(hitter.columns)} features")

# ── Pitcher profiles ──────────────────────────────────────────
print("Building pitcher profiles...")

pitcher = batted.groupby("player_name").agg(
    p_exit_velo_allowed    = ("launch_speed", "mean"),
    p_hard_hit_pct_allowed = ("hard_hit",     "mean"),
    p_barrel_pct_allowed   = ("barrel",       "mean"),
    p_launch_angle_allowed = ("launch_angle", "mean"),
    p_sweet_spot_pct_allowed = ("sweet_spot", "mean"),
    p_pull_air_pct_allowed = ("pull_air",     "mean"),
    p_hr_rate_allowed      = ("is_homerun",   "mean"),
    p_n_faced              = ("launch_speed", "count"),
).reset_index()

_pitcher_scores = pitcher.rename(columns={
    "p_barrel_pct_allowed": "barrel_pct_allowed",
    "p_hard_hit_pct_allowed": "hard_hit_pct_allowed",
    "p_sweet_spot_pct_allowed": "sweet_spot_pct_allowed",
    "p_pull_air_pct_allowed": "pull_air_pct_allowed",
    "p_hr_rate_allowed": "hr_rate_allowed",
    "p_exit_velo_allowed": "exit_velo_allowed",
    "p_launch_angle_allowed": "launch_angle_allowed",
})
pitcher["p_hr_contact_risk"] = pitcher_contact_risk(_pitcher_scores, "exit_velo_allowed")
pitcher["p_lift_damage_risk"] = pitcher_lift_risk(_pitcher_scores)

pitcher = pitcher.merge(aggregate_pitcher_profile(batted_2026, "p_2026_"), on="player_name", how="left")
pitcher = pitcher.merge(aggregate_pitcher_profile(batted_recent, "p_recent_"), on="player_name", how="left")

# Arm angle
arm = all_pitches.groupby("player_name")["arm_angle"].mean().reset_index().rename(
    columns={"arm_angle": "p_arm_angle"})
pitcher = pitcher.merge(arm, on="player_name", how="left")

# In-zone %
izp = all_pitches.groupby("player_name")["in_zone"].mean().reset_index().rename(
    columns={"in_zone": "p_in_zone_pct"})
pitcher = pitcher.merge(izp, on="player_name", how="left")

# Spin into barrel %
sib = all_pitches.groupby("player_name")["spin_into_barrel"].mean().reset_index().rename(
    columns={"spin_into_barrel": "p_spin_into_barrel_pct"})
pitcher = pitcher.merge(sib, on="player_name", how="left")

# Spin rates + pitch usage vs LHH/RHH per pitch type
total_vs = {}
for hand in ["R", "L"]:
    if "stand" in all_pitches.columns:
        total_vs[hand] = all_pitches[all_pitches["stand"] == hand]\
            .groupby("player_name").size().reset_index(name=f"total_{hand}")

for pt, label in [("FF","4seam"), ("SI","sinker"), ("SL","slider"),
                   ("CH","change"), ("CU","curve"),  ("FC","cutter")]:
    if "pitch_type" not in all_pitches.columns:
        break
    sub = all_pitches[all_pitches["pitch_type"] == pt]
    if len(sub) == 0:
        continue

    # Spin rate
    if "release_spin_rate" in all_pitches.columns:
        spin = sub.groupby("player_name")["release_spin_rate"].mean().reset_index().rename(
            columns={"release_spin_rate": f"p_spin_{label}"})
        pitcher = pitcher.merge(spin, on="player_name", how="left")

    # Usage vs RHH and LHH
    for hand, hlabel in [("R", "rhh"), ("L", "lhh")]:
        if hand not in total_vs or "stand" not in all_pitches.columns:
            continue
        usage = sub[sub["stand"] == hand].groupby("player_name").size().reset_index(name="pitch_n")
        usage = usage.merge(total_vs[hand], on="player_name", how="left")
        usage[f"p_{label}_usage_{hlabel}"] = usage["pitch_n"] / usage[f"total_{hand}"]
        pitcher = pitcher.merge(
            usage[["player_name", f"p_{label}_usage_{hlabel}"]],
            on="player_name", how="left"
        )

for pt, label in [("FF","4seam"), ("SI","sinker"), ("SL","slider"),
                  ("CH","change"), ("CU","curve"), ("FC","cutter")]:
    if "pitch_type" not in batted.columns:
        break
    sub = batted[batted["pitch_type"] == pt]
    if len(sub) == 0:
        continue
    pitch_contact = sub.groupby("player_name").agg(
        ev_allowed=("launch_speed", "mean"),
        barrel_allowed=("barrel", "mean"),
        launch_angle_allowed=("launch_angle", "mean"),
        sweet_spot_allowed=("sweet_spot", "mean"),
    ).reset_index().rename(columns={
        "ev_allowed": f"p_ev_allowed_{label}",
        "barrel_allowed": f"p_barrel_pct_allowed_{label}",
        "launch_angle_allowed": f"p_launch_angle_allowed_{label}",
        "sweet_spot_allowed": f"p_sweet_spot_pct_allowed_{label}",
    })
    pitcher = pitcher.merge(pitch_contact, on="player_name", how="left")

for hand, label in [("R", "rhh"), ("L", "lhh")]:
    if "stand" not in batted.columns:
        break
    side = batted[batted["stand"] == hand].groupby("player_name").agg(
        hr_rate_allowed=("is_homerun", "mean"),
        exit_velo_allowed=("launch_speed", "mean"),
        barrel_pct_allowed=("barrel", "mean"),
        hard_hit_pct_allowed=("hard_hit", "mean"),
        launch_angle_allowed=("launch_angle", "mean"),
        sweet_spot_pct_allowed=("sweet_spot", "mean"),
        pull_air_pct_allowed=("pull_air", "mean"),
        n_faced=("launch_speed", "count"),
    ).reset_index().rename(columns={
        "hr_rate_allowed": f"p_hr_rate_allowed_{label}",
        "exit_velo_allowed": f"p_exit_velo_allowed_{label}",
        "barrel_pct_allowed": f"p_barrel_pct_allowed_{label}",
        "hard_hit_pct_allowed": f"p_hard_hit_pct_allowed_{label}",
        "launch_angle_allowed": f"p_launch_angle_allowed_{label}",
        "sweet_spot_pct_allowed": f"p_sweet_spot_pct_allowed_{label}",
        "pull_air_pct_allowed": f"p_pull_air_pct_allowed_{label}",
        "n_faced": f"p_n_faced_{label}",
    })
    pitcher = pitcher.merge(side, on="player_name", how="left")

    required = [
        f"p_barrel_pct_allowed_{label}",
        f"p_hard_hit_pct_allowed_{label}",
        f"p_sweet_spot_pct_allowed_{label}",
        f"p_pull_air_pct_allowed_{label}",
        f"p_hr_rate_allowed_{label}",
        f"p_exit_velo_allowed_{label}",
        f"p_launch_angle_allowed_{label}",
    ]
    if all(col in pitcher.columns for col in required):
        side_scores = pitcher.rename(columns={
            f"p_barrel_pct_allowed_{label}": "barrel_pct_allowed",
            f"p_hard_hit_pct_allowed_{label}": "hard_hit_pct_allowed",
            f"p_sweet_spot_pct_allowed_{label}": "sweet_spot_pct_allowed",
            f"p_pull_air_pct_allowed_{label}": "pull_air_pct_allowed",
            f"p_hr_rate_allowed_{label}": "hr_rate_allowed",
            f"p_exit_velo_allowed_{label}": "exit_velo_allowed",
            f"p_launch_angle_allowed_{label}": "launch_angle_allowed",
        })
        pitcher[f"p_hr_contact_risk_{label}"] = pitcher_contact_risk(side_scores, "exit_velo_allowed")
        pitcher[f"p_lift_damage_risk_{label}"] = pitcher_lift_risk(side_scores)

if FEATURE_TOGGLES["pitcher_vulnerability"]:
    pitcher = add_pitcher_vulnerability_features(pitcher, all_pitches, batted, pa_df)

print(f"  Pitcher profiles: {len(pitcher)} players, {len(pitcher.columns)} features")

# ── Matchup history ───────────────────────────────────────────
print("Building matchup history...")

matchup = batted.groupby(["batter", "player_name"]).agg(
    m_hr_rate   = ("is_homerun",   "mean"),
    m_exit_velo = ("launch_speed", "mean"),
    m_pa        = ("is_homerun",   "count"),
).reset_index()

if FEATURE_TOGGLES["pitcher_batter_history"] and not pa_df.empty:
    bvp_pa = pa_df.groupby(["batter", "player_name"]).agg(
        m_bvp_pa=("is_pa", "sum"),
        m_bvp_ab=("is_ab", "sum"),
        m_bvp_hits=("is_hit", "sum"),
        m_bvp_hr=("is_hr", "sum"),
        m_bvp_woba_num=("woba_num", "sum"),
        m_bvp_so=("is_so", "sum"),
        m_bvp_tb=("tb", "sum"),
    ).reset_index()
    bvp_pa["m_bvp_woba"] = bvp_pa["m_bvp_woba_num"] / bvp_pa["m_bvp_pa"].replace(0, np.nan)
    bvp_pa["m_bvp_avg"] = bvp_pa["m_bvp_hits"] / bvp_pa["m_bvp_ab"].replace(0, np.nan)
    bvp_pa["m_bvp_slg"] = bvp_pa["m_bvp_tb"] / bvp_pa["m_bvp_ab"].replace(0, np.nan)
    bvp_pa["m_bvp_iso"] = bvp_pa["m_bvp_slg"] - bvp_pa["m_bvp_avg"]
    bvp_pa["m_bvp_hr_pa"] = bvp_pa["m_bvp_hr"] / bvp_pa["m_bvp_pa"].replace(0, np.nan)
    bvp_pa["m_bvp_k_pct"] = bvp_pa["m_bvp_so"] / bvp_pa["m_bvp_pa"].replace(0, np.nan)
    matchup = matchup.merge(
        bvp_pa[
            [
                "batter", "player_name", "m_bvp_pa", "m_bvp_woba", "m_bvp_iso",
                "m_bvp_hr_pa", "m_bvp_k_pct", "m_bvp_hr",
            ]
        ],
        on=["batter", "player_name"],
        how="left",
    )

    bvp_agg = {
        "m_bvp_barrel_pct": ("barrel", "mean"),
        "m_bvp_hard_hit_pct": ("hard_hit", "mean"),
    }
    if "estimated_woba_using_speedangle" in batted.columns:
        bvp_agg["m_bvp_xwoba"] = ("estimated_woba_using_speedangle", "mean")
    bvp_batted = batted.groupby(["batter", "player_name"]).agg(**bvp_agg).reset_index()
    matchup = matchup.merge(bvp_batted, on=["batter", "player_name"], how="left")

# ── Merge all features onto at-bat level ──────────────────────
print("Merging features...")

final = batted.merge(hitter,  on="batter",                  how="left")
final = final.merge(pitcher,  on="player_name",             how="left")
final = final.merge(matchup,  on=["batter", "player_name"], how="left")
batter_right_series = (final["stand"] == "R").astype(int)
pitcher_right_series = (final["p_throws"] == "R").astype(int)

def col_or_nan(df, col):
    return df[col] if col in df.columns else np.nan


def blend_series(base, season, recent, season_weight=0.35, recent_weight=0.35, season_n=None, recent_n=None,
                 season_scale=120.0, recent_scale=45.0):
    base_weight = 1.0
    sw = season_weight
    rw = recent_weight
    if season_n is not None:
        sw = season_weight * np.clip(season_n / season_scale, 0.0, 1.0)
    if recent_n is not None:
        rw = recent_weight * np.clip(recent_n / recent_scale, 0.0, 1.0)
    total = base_weight + sw + rw
    return (base * base_weight + season * sw + recent * rw) / total

extra_cols = {
    "ballpark_code": final["home_team"].astype("category").cat.codes,
    "is_coors": (final["home_team"] == "COL").astype(int),
    "batter_right": batter_right_series,
    "pitcher_right": pitcher_right_series,
    "m_sweet_spot_contact_edge": final["h_sweet_spot_pct"] * final["p_sweet_spot_pct_allowed"],
    "m_zone_attack_edge": final["h_zone_contact_pct"] * final["p_in_zone_pct"],
    "m_barrel_matchup_score": final["h_barrel_pct"] * final["p_barrel_pct_allowed"],
    "m_lift_matchup_score": (
        final["h_sweet_spot_pct"] * final["p_sweet_spot_pct_allowed"]
        + final["h_pull_air_pct"] * final["p_pull_air_pct_allowed"]
    ) / 2.0,
    "m_hr_contact_matchup": final["h_hr_contact_score"] * final["p_hr_contact_risk"],
    "m_lifted_power_matchup": final["h_lifted_power_score"] * final["p_lift_damage_risk"],
    "h_hr_rate_vs_hand": np.where(final["p_throws"] == "R", col_or_nan(final, "h_hr_vs_rhp"), col_or_nan(final, "h_hr_vs_lhp")),
    "h_hr_pa_vs_hand": np.where(final["p_throws"] == "R", col_or_nan(final, "h_hr_pa_vs_rhp"), col_or_nan(final, "h_hr_pa_vs_lhp")),
    "h_woba_vs_hand": np.where(final["p_throws"] == "R", col_or_nan(final, "h_woba_vs_rhp"), col_or_nan(final, "h_woba_vs_lhp")),
    "h_iso_vs_hand": np.where(final["p_throws"] == "R", col_or_nan(final, "h_iso_vs_rhp"), col_or_nan(final, "h_iso_vs_lhp")),
    "h_k_pct_vs_hand": np.where(final["p_throws"] == "R", col_or_nan(final, "h_k_pct_vs_rhp"), col_or_nan(final, "h_k_pct_vs_lhp")),
    "h_xwoba_vs_hand": np.where(final["p_throws"] == "R", col_or_nan(final, "h_xwoba_vs_rhp"), col_or_nan(final, "h_xwoba_vs_lhp")),
    "h_barrel_pct_vs_hand": np.where(final["p_throws"] == "R", col_or_nan(final, "h_barrel_pct_vs_rhp"), col_or_nan(final, "h_barrel_pct_vs_lhp")),
    "h_hard_hit_pct_vs_hand": np.where(final["p_throws"] == "R", col_or_nan(final, "h_hard_hit_pct_vs_rhp"), col_or_nan(final, "h_hard_hit_pct_vs_lhp")),
    "h_launch_angle_vs_hand": np.where(final["p_throws"] == "R", col_or_nan(final, "h_launch_angle_vs_rhp"), col_or_nan(final, "h_launch_angle_vs_lhp")),
    "h_sweet_spot_pct_vs_hand": np.where(final["p_throws"] == "R", col_or_nan(final, "h_sweet_spot_pct_vs_rhp"), col_or_nan(final, "h_sweet_spot_pct_vs_lhp")),
    "h_pull_air_pct_vs_hand": np.where(final["p_throws"] == "R", col_or_nan(final, "h_pull_air_pct_vs_rhp"), col_or_nan(final, "h_pull_air_pct_vs_lhp")),
    "h_hr_contact_score_vs_hand": np.where(final["p_throws"] == "R", col_or_nan(final, "h_hr_contact_score_vs_rhp"), col_or_nan(final, "h_hr_contact_score_vs_lhp")),
    "h_lifted_power_score_vs_hand": np.where(final["p_throws"] == "R", col_or_nan(final, "h_lifted_power_score_vs_rhp"), col_or_nan(final, "h_lifted_power_score_vs_lhp")),
    "p_hr_rate_allowed_vs_side": np.where(batter_right_series == 1, col_or_nan(final, "p_hr_rate_allowed_rhh"), col_or_nan(final, "p_hr_rate_allowed_lhh")),
    "p_exit_velo_allowed_vs_side": np.where(batter_right_series == 1, col_or_nan(final, "p_exit_velo_allowed_rhh"), col_or_nan(final, "p_exit_velo_allowed_lhh")),
    "p_barrel_pct_allowed_vs_side": np.where(batter_right_series == 1, col_or_nan(final, "p_barrel_pct_allowed_rhh"), col_or_nan(final, "p_barrel_pct_allowed_lhh")),
    "p_hard_hit_pct_allowed_vs_side": np.where(batter_right_series == 1, col_or_nan(final, "p_hard_hit_pct_allowed_rhh"), col_or_nan(final, "p_hard_hit_pct_allowed_lhh")),
    "p_launch_angle_allowed_vs_side": np.where(batter_right_series == 1, col_or_nan(final, "p_launch_angle_allowed_rhh"), col_or_nan(final, "p_launch_angle_allowed_lhh")),
    "p_sweet_spot_pct_allowed_vs_side": np.where(batter_right_series == 1, col_or_nan(final, "p_sweet_spot_pct_allowed_rhh"), col_or_nan(final, "p_sweet_spot_pct_allowed_lhh")),
    "p_pull_air_pct_allowed_vs_side": np.where(batter_right_series == 1, col_or_nan(final, "p_pull_air_pct_allowed_rhh"), col_or_nan(final, "p_pull_air_pct_allowed_lhh")),
    "p_hr_contact_risk_vs_side": np.where(batter_right_series == 1, col_or_nan(final, "p_hr_contact_risk_rhh"), col_or_nan(final, "p_hr_contact_risk_lhh")),
    "p_lift_damage_risk_vs_side": np.where(batter_right_series == 1, col_or_nan(final, "p_lift_damage_risk_rhh"), col_or_nan(final, "p_lift_damage_risk_lhh")),
    "p_hr_vulnerability_score": col_or_nan(final, "p_hr_vulnerability_score"),
}

h_recent_n = col_or_nan(final, "h_recent_n_batted")
h_2026_n = col_or_nan(final, "h_2026_n_batted")
p_recent_n = col_or_nan(final, "p_recent_n_faced")
p_2026_n = col_or_nan(final, "p_2026_n_faced")

for metric in [
    "exit_velo", "barrel_pct", "launch_angle", "sweet_spot_pct",
    "hard_hit_pct", "pull_air_pct", "hr_rate", "hr_contact_score", "lifted_power_score",
]:
    extra_cols[f"h_form_{metric}"] = blend_series(
        col_or_nan(final, f"h_{metric}"),
        col_or_nan(final, f"h_2026_{metric}"),
        col_or_nan(final, f"h_recent_{metric}"),
        season_n=h_2026_n,
        recent_n=h_recent_n,
    )

for metric in [
    "exit_velo_allowed", "hard_hit_pct_allowed", "barrel_pct_allowed", "launch_angle_allowed",
    "sweet_spot_pct_allowed", "pull_air_pct_allowed", "hr_rate_allowed", "hr_contact_risk", "lift_damage_risk",
]:
    extra_cols[f"p_form_{metric}"] = blend_series(
        col_or_nan(final, f"p_{metric}"),
        col_or_nan(final, f"p_2026_{metric}"),
        col_or_nan(final, f"p_recent_{metric}"),
        season_n=p_2026_n,
        recent_n=p_recent_n,
        season_scale=180.0,
        recent_scale=70.0,
    )

extra_cols["m_handed_hr_matchup"] = extra_cols["h_hr_rate_vs_hand"] * extra_cols["p_hr_rate_allowed_vs_side"]
extra_cols["m_handed_contact_matchup"] = extra_cols["h_hr_contact_score_vs_hand"] * extra_cols["p_hr_contact_risk_vs_side"]
extra_cols["m_handed_lift_matchup"] = extra_cols["h_lifted_power_score_vs_hand"] * extra_cols["p_lift_damage_risk_vs_side"]

pitch_hr_terms = []
pitch_contact_terms = []
pitch_ev_terms = []

for label in ["4seam", "sinker", "slider", "change", "curve", "cutter"]:
    usage_r = f"p_{label}_usage_rhh"
    usage_l = f"p_{label}_usage_lhh"
    usage_match_col = f"p_{label}_usage_matchup"
    if usage_r in final.columns and usage_l in final.columns:
        extra_cols[usage_match_col] = np.where(
            extra_cols["batter_right"] == 1,
            final[usage_r],
            final[usage_l],
        )
    if usage_match_col not in extra_cols:
        continue

    usage_match = extra_cols[usage_match_col]
    hr_col = f"h_hr_vs_{label}"
    if hr_col in final.columns:
        extra_cols[f"m_{label}_hr_exposure"] = final[hr_col] * usage_match
        pitch_hr_terms.append(extra_cols[f"m_{label}_hr_exposure"])

    xwoba_col = f"h_xwoba_vs_{label}"
    if xwoba_col in final.columns:
        extra_cols[f"m_{label}_xwoba_exposure"] = final[xwoba_col] * usage_match
        pitch_contact_terms.append(extra_cols[f"m_{label}_xwoba_exposure"])

    ev_col = f"h_ev_vs_{label}"
    p_ev_col = f"p_ev_allowed_{label}"
    if ev_col in final.columns and p_ev_col in final.columns:
        extra_cols[f"m_{label}_ev_delta"] = final[ev_col] - final[p_ev_col]
        pitch_ev_terms.append(extra_cols[f"m_{label}_ev_delta"] * usage_match)

extra_cols["m_pitch_hr_matchup"] = np.sum(pitch_hr_terms, axis=0) if pitch_hr_terms else np.nan
extra_cols["m_pitch_contact_matchup"] = np.sum(pitch_contact_terms, axis=0) if pitch_contact_terms else np.nan
extra_cols["m_pitch_ev_matchup"] = np.sum(pitch_ev_terms, axis=0) if pitch_ev_terms else np.nan
if "m_bvp_pa" in final.columns:
    extra_cols["m_bvp_weight"] = np.where(
        final["m_bvp_pa"] >= 15, 0.20,
        np.where(final["m_bvp_pa"] >= 8, 0.10, 0.0)
    )

final = pd.concat([final, pd.DataFrame(extra_cols, index=final.index)], axis=1)

# ── Weather (historical via Open-Meteo archive) ───────────────
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

WEATHER_FIELDS = ["temp_f", "humidity", "wind_speed", "wind_dir"]

def get_weather(game_date, team):
    coords = ballpark_coords.get(team)
    if not coords:
        return {}
    lat, lon = coords
    date_str = game_date.strftime("%Y-%m-%d")
    url = (
        f"https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={lat}&longitude={lon}"
        f"&start_date={date_str}&end_date={date_str}"
        f"&hourly=temperature_2m,relativehumidity_2m,windspeed_10m,winddirection_10m"
        f"&temperature_unit=fahrenheit&windspeed_unit=mph&timezone=America/New_York"
    )
    try:
        d = requests.get(url, timeout=10).json()["hourly"]
        return {
            "temp_f":    d["temperature_2m"][19],
            "humidity":  d["relativehumidity_2m"][19],
            "wind_speed":d["windspeed_10m"][19],
            "wind_dir":  d["winddirection_10m"][19],
        }
    except:
        return {}

weather_cache = {}
cache_path = Path("homerun_data_enriched.csv")
if cache_path.exists():
    try:
        cached = pd.read_csv(
            cache_path,
            usecols=lambda c: c in {"game_date", "home_team", *WEATHER_FIELDS},
            on_bad_lines="skip",
            low_memory=False,
        )
        cached["game_date"] = pd.to_datetime(cached["game_date"], errors="coerce", format="mixed")
        cached = cached.dropna(subset=["game_date", "home_team"]).drop_duplicates(["game_date", "home_team"])
        for _, row in cached.iterrows():
            weather_cache[(row["game_date"], row["home_team"])] = {
                field: row[field] for field in WEATHER_FIELDS if field in row and not pd.isna(row[field])
            }
        print(f"Loaded cached weather for {len(weather_cache)} game/team pairs from homerun_data_enriched.csv")
    except Exception as e:
        print(f"  Weather cache load failed: {e}")

games_list = final[["game_date", "home_team"]].drop_duplicates()
missing_games = [
    (row["game_date"], row["home_team"])
    for _, row in games_list.iterrows()
    if len(weather_cache.get((row["game_date"], row["home_team"]), {})) < len(WEATHER_FIELDS)
]

if missing_games:
    print(f"Fetching weather data for {len(missing_games)} missing game/team pairs...")
    for i, (game_date, team) in enumerate(missing_games):
        if i % 50 == 0:
            print(f"  Weather: {i}/{len(missing_games)} games...")
        weather_cache[(game_date, team)] = get_weather(game_date, team)
else:
    print("Using cached historical weather for all games.")

weather_rows = [
    {"game_date": game_date, "home_team": team, **{field: payload.get(field, np.nan) for field in WEATHER_FIELDS}}
    for (game_date, team), payload in weather_cache.items()
]
weather_df = pd.DataFrame(weather_rows).drop_duplicates(["game_date", "home_team"])
final = final.drop(columns=[field for field in WEATHER_FIELDS if field in final.columns], errors="ignore")
final = final.merge(weather_df, on=["game_date", "home_team"], how="left")

# ── Save ──────────────────────────────────────────────────────
hitter.to_csv("hitter_profiles.csv",      index=False)
pitcher.to_csv("pitcher_profiles.csv",    index=False)
final.to_csv("homerun_data_enriched.csv", index=False)

print(f"\nDone!")
print(f"  Enriched data: {len(final):,} rows, {len(final.columns)} features")
print(f"  Hitter profiles:  {len(hitter):,} players")
print(f"  Pitcher profiles: {len(pitcher):,} players")
print("  Saved: hitter_profiles.csv, pitcher_profiles.csv, homerun_data_enriched.csv")
