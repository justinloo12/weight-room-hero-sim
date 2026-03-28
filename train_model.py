import pandas as pd
import numpy as np
import requests
from pathlib import Path

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

h_ev_bonus = ((hitter["h_exit_velo"] - 88.0) / 8.0).clip(lower=0)
h_launch_window = ((16.0 - (hitter["h_launch_angle"] - 18.0).abs()) / 16.0).clip(lower=0)
hitter["h_hr_contact_score"] = (
    hitter["h_barrel_pct"] * 0.35
    + hitter["h_sweet_spot_pct"] * 0.25
    + hitter["h_hard_hit_pct"] * 0.20
    + hitter["h_pull_air_pct"] * 0.20
    + h_ev_bonus * 0.10
)
hitter["h_lifted_power_score"] = (
    hitter["h_barrel_pct"] * 0.45
    + hitter["h_sweet_spot_pct"] * 0.30
    + hitter["h_pull_air_pct"] * 0.15
    + h_launch_window * 0.10
)

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
    ).reset_index().rename(columns={
        "hr_rate":   f"h_hr_vs_{label}",
        "exit_velo": f"h_ev_vs_{label}",
    })
    hitter = hitter.merge(platoon, on="batter", how="left")

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

p_ev_risk = ((pitcher["p_exit_velo_allowed"] - 88.0) / 8.0).clip(lower=0)
p_launch_window_allowed = ((16.0 - (pitcher["p_launch_angle_allowed"] - 18.0).abs()) / 16.0).clip(lower=0)
pitcher["p_hr_contact_risk"] = (
    pitcher["p_barrel_pct_allowed"] * 0.30
    + pitcher["p_hard_hit_pct_allowed"] * 0.20
    + pitcher["p_sweet_spot_pct_allowed"] * 0.20
    + pitcher["p_pull_air_pct_allowed"] * 0.15
    + pitcher["p_hr_rate_allowed"] * 0.15
    + p_ev_risk * 0.10
)
pitcher["p_lift_damage_risk"] = (
    pitcher["p_sweet_spot_pct_allowed"] * 0.35
    + pitcher["p_pull_air_pct_allowed"] * 0.25
    + p_launch_window_allowed * 0.20
    + pitcher["p_barrel_pct_allowed"] * 0.20
)

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

print(f"  Pitcher profiles: {len(pitcher)} players, {len(pitcher.columns)} features")

# ── Matchup history ───────────────────────────────────────────
print("Building matchup history...")

matchup = batted.groupby(["batter", "player_name"]).agg(
    m_hr_rate   = ("is_homerun",   "mean"),
    m_exit_velo = ("launch_speed", "mean"),
    m_pa        = ("is_homerun",   "count"),
).reset_index()

# ── Merge all features onto at-bat level ──────────────────────
print("Merging features...")

final = batted.merge(hitter,  on="batter",                  how="left")
final = final.merge(pitcher,  on="player_name",             how="left")
final = final.merge(matchup,  on=["batter", "player_name"], how="left")
extra_cols = {
    "ballpark_code": final["home_team"].astype("category").cat.codes,
    "is_coors": (final["home_team"] == "COL").astype(int),
    "batter_right": (final["stand"] == "R").astype(int),
    "pitcher_right": (final["p_throws"] == "R").astype(int),
    "m_sweet_spot_contact_edge": final["h_sweet_spot_pct"] * final["p_sweet_spot_pct_allowed"],
    "m_zone_attack_edge": final["h_zone_contact_pct"] * final["p_in_zone_pct"],
    "m_barrel_matchup_score": final["h_barrel_pct"] * final["p_barrel_pct_allowed"],
    "m_lift_matchup_score": (
        final["h_sweet_spot_pct"] * final["p_sweet_spot_pct_allowed"]
        + final["h_pull_air_pct"] * final["p_pull_air_pct_allowed"]
    ) / 2.0,
    "m_hr_contact_matchup": final["h_hr_contact_score"] * final["p_hr_contact_risk"],
    "m_lifted_power_matchup": final["h_lifted_power_score"] * final["p_lift_damage_risk"],
}

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

    xwoba_col = f"h_xwoba_vs_{label}"
    if xwoba_col in final.columns:
        extra_cols[f"m_{label}_xwoba_exposure"] = final[xwoba_col] * usage_match

    ev_col = f"h_ev_vs_{label}"
    p_ev_col = f"p_ev_allowed_{label}"
    if ev_col in final.columns and p_ev_col in final.columns:
        extra_cols[f"m_{label}_ev_delta"] = final[ev_col] - final[p_ev_col]

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
        cached = pd.read_csv(cache_path, usecols=["game_date", "home_team"] + WEATHER_FIELDS)
        cached["game_date"] = pd.to_datetime(cached["game_date"])
        cached = cached.dropna(subset=["game_date", "home_team"]).drop_duplicates(["game_date", "home_team"])
        for _, row in cached.iterrows():
            weather_cache[(row["game_date"], row["home_team"])] = {
