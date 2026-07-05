"""Log today's frozen-model HR probabilities — a forecasting scorecard.

Pre-registered protocol (PROTOCOL.json): every day, for every batter in a
POSTED starting lineup of a not-yet-started MLB game, emit the frozen
model's P(>= 1 HR) and append it to predictions_log.csv. The next morning
grade_predictions.py grades every row against what actually happened and
rebuilds SCOREBOARD.md.

The frozen model is GBM v3 (hr_model_v3.pkl), the holdout winner — see
EVALUATION.md. Its features come from three frozen artifacts:

* hr_model_v3.pkl          — HistGradientBoosting weights + isotonic map
* scoring_pack_v3.pkl.gz   — hitter/pitcher/matchup profile tables built
                             through 2026-07-02 (train_model_v3.py pack)
* structural_v2.pkl        — structural tables for the s_* features

Rules that keep the scorecard honest:

* Data sources: the FREE MLB Stats API (statsapi.mlb.com) for schedule,
  probables and lineups, plus the free Open-Meteo forecast for game-time
  weather. No odds API, no credits, no stakes — this is a forecast log,
  not betting.
* Only games in "Preview" state are logged: every prediction is strictly
  pre-game. Lineups not yet posted are skipped; a later run the same day
  picks them up when they appear.
* First write wins: once a (date, batter) prediction is logged it is
  never modified. Re-runs are idempotent and can only ADD new rows
  (append_predictions enforces this).
* Changing the model or artifacts requires a dated amendment note in
  PROTOCOL.json.
"""

import gzip
import json
import pickle
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

ET = ZoneInfo("America/New_York")
LOG_PATH = Path("predictions_log.csv")
PROTOCOL_PATH = Path("PROTOCOL.json")
MODEL_ID = "gbm_v3"
MODEL_ARTIFACT = "hr_model_v3.pkl"
PACK_ARTIFACT = "scoring_pack_v3.pkl.gz"
STRUCT_ARTIFACT = "structural_v2.pkl"

SCHEDULE_URL = ("https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={date}"
                "&hydrate=probablePitcher,team,lineups")
PEOPLE_URL = "https://statsapi.mlb.com/api/v1/people?personIds={ids}"
FORECAST_URL = ("https://api.open-meteo.com/v1/forecast"
                "?latitude={lat}&longitude={lon}&start_date={date}&end_date={date}"
                "&hourly=temperature_2m,relativehumidity_2m,windspeed_10m,winddirection_10m"
                "&temperature_unit=fahrenheit&windspeed_unit=mph&timezone=America/New_York")

LOG_COLUMNS = ["date", "game_pk", "player", "batter_id", "team", "opponent",
               "park", "slot", "opp_starter", "opp_starter_id",
               "opp_starter_hand", "model", "probability", "hr"]

WEATHER_FIELDS = ["temp_f", "humidity", "wind_speed", "wind_dir"]


# ── Pure parsing / assembly (fully unit-testable, no network) ────────────

def parse_games(schedule_json):
    """Extract pre-game matchup structure from a statsapi schedule payload.

    Returns a list of dicts, one per game still in Preview state:
    {game_pk, home, away, home_pitcher, away_pitcher, home_lineup,
     away_lineup} where each pitcher is {"id", "name"} (or None) and each
    lineup is an ordered list of {"id", "name"} (empty if not posted).
    """
    games = []
    for date_block in schedule_json.get("dates", []):
        for g in date_block.get("games", []):
            state = g.get("status", {}).get("abstractGameState", "")
            if state != "Preview":
                continue  # prediction must be strictly pre-game
            lineups = g.get("lineups") or {}
            game = {"game_pk": g.get("gamePk")}
            for side in ("home", "away"):
                team = g.get("teams", {}).get(side, {})
                game[side] = team.get("team", {}).get("abbreviation", "")
                pp = team.get("probablePitcher") or {}
                game[f"{side}_pitcher"] = (
                    {"id": pp.get("id"), "name": pp.get("fullName", "")}
                    if pp.get("id") else None)
                posted = lineups.get(f"{side}Players") or []
                game[f"{side}_lineup"] = [
                    {"id": p.get("id"), "name": p.get("fullName", "")}
                    for p in posted if p.get("id")]
            games.append(game)
    return games


def build_matchup_frame(games, date_str, pitcher_hands):
    """One row per (posted-lineup batter): everything the scorer needs.

    Pure given its inputs. Away batters face the home probable and vice
    versa; the park is always the home team's. Missing probables leave
    opp_starter_id/hand empty (the model degrades to neutral factors).
    """
    rows = []
    for g in games:
        for side, opp_side in (("home", "away"), ("away", "home")):
            lineup = g[f"{side}_lineup"]
            if not lineup:
                continue  # lineup not posted yet — a later run catches it
            opp = g[f"{opp_side}_pitcher"]
            opp_id = opp["id"] if opp else None
            for slot, batter in enumerate(lineup[:9], start=1):
                rows.append({
                    "date": date_str, "game_pk": g["game_pk"],
                    "player": batter["name"], "batter": int(batter["id"]),
                    "team": g[side], "opponent": g[opp_side],
                    "home_team": g["home"], "slot": float(slot),
                    "opp_starter": opp["name"] if opp else "",
                    "opp_starter_id": opp_id if opp_id else np.nan,
                    "opp_starter_hand": pitcher_hands.get(opp_id) if opp_id else None,
                })
    return pd.DataFrame(rows)


def append_predictions(existing, new):
    """Append new prediction rows; first logged (date, batter_id) wins.

    Idempotent: re-appending rows already present adds nothing and never
    alters existing rows (including their graded outcomes). Returns
    (combined_frame, n_added).
    """
    if existing is None or existing.empty:
        return new.copy(), len(new)
    have = set(zip(existing["date"].astype(str), existing["batter_id"].astype(int)))
    keys = list(zip(new["date"].astype(str), new["batter_id"].astype(int)))
    fresh = new[[k not in have for k in keys]]
    combined = pd.concat([existing, fresh], ignore_index=True)
    return combined, len(fresh)


# ── The frozen v3 scorer ──────────────────────────────────────────────────

class V3Scorer:
    """Assembles v3 features from the frozen artifacts and scores them.

    Mirrors train_model_v3.assemble_segment exactly: profile features via
    build_features.assemble_matchup_frame + weather + structural s_*
    columns, reindexed to the frozen feature list, then
    HistGradientBoosting -> isotonic/sigmoid calibration.
    """

    def __init__(self, model_path=MODEL_ARTIFACT, pack_path=PACK_ARTIFACT,
                 struct_path=STRUCT_ARTIFACT):
        from structural_model import StructuralModelV2
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)
        with gzip.open(pack_path, "rb") as f:
            self.pack = pickle.load(f)
        self.struct = StructuralModelV2.load(struct_path)

    def score(self, pg, weather_by_park=None):
        """P(>= 1 HR) for each row of a build_matchup_frame() frame."""
        import build_features as bf
        from train_model_v3 import (STRUCT_FEATURES, apply_calibrator,
                                    structural_feature_frame)

        pg = pg.reset_index(drop=True).copy()
        pg["stand"] = pg["batter"].map(self.struct.batter_stand)
        rows = pd.DataFrame({
            "batter": pg["batter"],
            "player_name": pg["opp_starter_id"].map(self.pack["pitcher_name_by_id"]),
            "stand": pg["stand"],
            "p_throws": pg["opp_starter_hand"],
            "home_team": pg["home_team"],
            "game_date": pd.to_datetime(pg["date"]),
        })
        frame = bf.assemble_matchup_frame(
            rows, self.pack["hitter"], self.pack["pitcher"], self.pack["matchup"],
            team_categories=self.pack["team_categories"])
        frame = frame.loc[:, ~frame.columns.duplicated()]
        weather_by_park = weather_by_park or {}
        for i, field in enumerate(WEATHER_FIELDS):
            frame[field] = [
                (weather_by_park.get(t) or [np.nan] * 4)[i]
                for t in pg["home_team"]]
        sf = structural_feature_frame(self.struct, pg)
        X = pd.concat([frame.reindex(columns=self.pack["features"]).reset_index(drop=True),
                       sf[STRUCT_FEATURES].reset_index(drop=True)], axis=1)
        X = X.reindex(columns=self.model["features"])
        p_raw = self.model["clf"].predict_proba(X)[:, 1]
        return apply_calibrator(self.model["calibrators"],
                                self.model["cal_method"], p_raw)


# ── Network + IO glue ─────────────────────────────────────────────────────

def fetch_pitcher_hands(pitcher_ids):
    """{pitcher_id: 'R'/'L'} for all probables, one batched people call."""
    ids = sorted({int(i) for i in pitcher_ids if i})
    if not ids:
        return {}
    try:
        resp = requests.get(PEOPLE_URL.format(ids=",".join(map(str, ids))), timeout=15)
        resp.raise_for_status()
        people = resp.json().get("people", [])
    except Exception as e:
        print(f"  pitcher-hand lookup failed ({e}); platoon factors neutral today")
        return {}
    return {p["id"]: p.get("pitchHand", {}).get("code")
            for p in people if p.get("pitchHand", {}).get("code") in ("R", "L")}


def fetch_forecast_weather(parks, date_str):
    """{park: [temp_f, humidity, wind_speed, wind_dir]} at 7pm ET from the
    free Open-Meteo forecast API (same convention as the training data).
    Any failure degrades to missing weather (the model handles NaN)."""
    import build_features as bf
    out = {}
    for park in sorted(set(parks)):
        coords = bf.ballpark_coords.get(park)
        if coords is None:
            continue
        try:
            resp = requests.get(FORECAST_URL.format(
                lat=coords[0], lon=coords[1], date=date_str), timeout=15)
            resp.raise_for_status()
            h = resp.json()["hourly"]
            idx = next(i for i, t in enumerate(h["time"]) if t.endswith("T19:00"))
            out[park] = [h["temperature_2m"][idx], h["relativehumidity_2m"][idx],
                         h["windspeed_10m"][idx], h["winddirection_10m"][idx]]
        except Exception as e:
            print(f"  forecast failed for {park}: {e}")
    return out


def load_log():
    if LOG_PATH.exists():
        return pd.read_csv(LOG_PATH, dtype={"hr": "string"}, keep_default_na=False)
    return pd.DataFrame(columns=LOG_COLUMNS)


def to_log_rows(pg, probs, model_id=MODEL_ID):
    """Format scored matchup rows into the predictions_log.csv schema."""
    out = pg.copy()
    out["park"] = out["home_team"]
    out["batter_id"] = out["batter"]
    out["slot"] = out["slot"].astype(int)
    out["opp_starter_id"] = out["opp_starter_id"].map(
        lambda v: int(v) if pd.notna(v) else "")
    out["opp_starter_hand"] = out["opp_starter_hand"].fillna("")
    out["model"] = model_id
    out["probability"] = np.round(np.asarray(probs, dtype=float), 5)
    out["hr"] = ""
    return out[LOG_COLUMNS]


def main():
    if PROTOCOL_PATH.exists():
        protocol = json.loads(PROTOCOL_PATH.read_text())
        assert protocol["model"]["id"] == MODEL_ID, \
            "model drift: predict_today.py disagrees with PROTOCOL.json"

    today = datetime.now(ET).strftime("%Y-%m-%d")
    print(f"Predictions for {today} ({MODEL_ID}, frozen)")

    resp = requests.get(SCHEDULE_URL.format(date=today), timeout=20)
    resp.raise_for_status()
    games = parse_games(resp.json())
    n_posted = sum(bool(g["home_lineup"]) + bool(g["away_lineup"]) for g in games)
    print(f"  {len(games)} pre-game matchups, {n_posted} posted lineups")
    if n_posted == 0:
        print("  nothing to log yet")
        return

    hands = fetch_pitcher_hands(
        [p["id"] for g in games for k in ("home_pitcher", "away_pitcher")
         if (p := g[k])])
    pg = build_matchup_frame(games, today, hands)
    weather = fetch_forecast_weather(pg["home_team"], today)
    probs = V3Scorer().score(pg, weather)
    combined, n_added = append_predictions(load_log(), to_log_rows(pg, probs))
    combined.to_csv(LOG_PATH, index=False)
    print(f"  logged {n_added} new predictions ({len(combined)} total rows)")


if __name__ == "__main__":
    main()
