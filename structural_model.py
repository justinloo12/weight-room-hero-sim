"""Transparent structural baseline for P(player hits >= 1 HR in a game).

The model is a closed-form formula with no fitted black-box parts:

    P(HR in game) = 1 - (1 - p_PA) ** E[PA]

where

    p_PA  = shrunk_batter_HR_rate * pitcher_factor * platoon_factor
    E[PA] = trailing plate appearances per game, regressed to the league mean

Every component:

1. shrunk_batter_HR_rate — the batter's empirical HR/PA shrunk toward the
   league mean via an empirical-Bayes beta-binomial posterior:
       posterior = (hr + alpha) / (pa + alpha + beta)
   alpha/beta are fit from the league-wide distribution of batter HR rates
   by method of moments (see fit_beta_binomial_mom).

2. pitcher_factor — (pitcher's shrunk HR-allowed/PA) / (league HR/PA),
   clipped to PITCHER_FACTOR_CLIP. Because the opposing STARTER only faces
   the batter for part of the game (relievers cover the rest), the factor
   is blended toward 1.0 by STARTER_PA_SHARE.

3. platoon_factor — batter's HR/PA vs the starter's throwing hand, shrunk
   toward the batter's own overall rate with PLATOON_PSEUDO_PA
   pseudo-observations, divided by the overall rate, clipped to
   PLATOON_FACTOR_CLIP.

4. E[PA] — batter's season-to-date PA / games-played, regressed toward the
   league-average PA/game with EPA_PRIOR_GAMES pseudo-games. (No reliable
   lineup-slot history is stored in this repo, so the league prior is a
   single number; slot-specific priors are a documented next step.)

All constants below are hand-set and documented — nothing here is tuned on
holdout data.
"""

import numpy as np
import pandas as pd

# ── Documented constants ──────────────────────────────────────────────────
# Players must have this many PA to be included in the method-of-moments fit
# of the league talent distribution (below it, sampling noise swamps talent).
MIN_PA_FOR_MOM = 100

# Bounds on the fitted prior strength (alpha + beta), i.e. how many
# pseudo-PA the league prior is worth. Guards against degenerate MoM fits.
PRIOR_STRENGTH_BOUNDS = (30.0, 2000.0)

# The opposing starter historically takes roughly 60-70% of a batter's PA;
# the remainder goes to relievers, assumed league-average.
STARTER_PA_SHARE = 0.65

# A truly extreme HR-prone pitcher allows ~2x league rate over a full
# season; anything outside these bounds is small-sample noise.
PITCHER_FACTOR_CLIP = (0.5, 2.0)

# Platoon splits are noisy: shrink the vs-hand rate toward the batter's
# overall rate with this many pseudo-PA before taking the ratio.
PLATOON_PSEUDO_PA = 350.0
PLATOON_FACTOR_CLIP = (0.7, 1.4)

# E[PA]: regress a batter's observed PA/game to the league mean with this
# many pseudo-games (about two weeks of starts).
EPA_PRIOR_GAMES = 10.0

# ── v2 constants (lineup slots + park factors) ────────────────────────────
# Weight on the league PA/game-for-his-slot mean when the batter's slot in
# TODAY'S posted lineup is known; the rest stays on his trailing PA/game
# (which carries player-specific effects like getting pinch-hit for).
SLOT_EPA_WEIGHT = 0.7

# Slots 1-9 are the starting lineup; a first PA ranking beyond 9 means a
# mid-game substitute, whose slot is NOT knowable from a posted lineup.
MAX_LINEUP_SLOT = 9

# Park factors by batter handedness are shrunk toward 1.0 with this many
# pseudo-PA at the league rate. A park/hand cell gets ~2,500-3,500 PA per
# season, so one season of data moves the factor roughly halfway from 1.0
# to its raw value — deliberate heavy regression.
PARK_PSEUDO_PA = 2000.0

# Post-shrinkage guard rails. Even Coors should not exceed this after
# regression on one-plus season of data.
PARK_FACTOR_CLIP = (0.80, 1.25)

# Safety cap on the per-PA probability (the best HR seasons ever peaked
# near 0.10 HR/PA; anything above 0.25 is a bug, not a forecast).
P_PA_CAP = 0.25


# ── Core math ─────────────────────────────────────────────────────────────

def fit_beta_binomial_mom(hr_counts, pa_counts, min_pa=MIN_PA_FOR_MOM):
    """Fit Beta(alpha, beta) prior over batter HR/PA rates by method of moments.

    Uses players with pa >= min_pa. The observed variance of raw rates is
    the sum of true talent variance and binomial sampling noise, so the
    expected sampling variance E[p(1-p)/n] is subtracted before matching
    moments. Returns (alpha, beta).
    """
    hr = np.asarray(hr_counts, dtype=float)
    pa = np.asarray(pa_counts, dtype=float)
    mask = pa >= min_pa
    if mask.sum() < 10:
        # Not enough qualifiers: fall back to a weakly-informative prior
        # centred on the pooled rate, worth ~200 PA.
        pooled = hr.sum() / max(pa.sum(), 1.0)
        return pooled * 200.0, (1.0 - pooled) * 200.0

    rates = hr[mask] / pa[mask]
    m = rates.mean()
    v_obs = rates.var(ddof=1)
    # Expected binomial sampling variance at each player's sample size.
    v_within = np.mean(m * (1.0 - m) / pa[mask])
    v_between = max(v_obs - v_within, 1e-7)

    strength = m * (1.0 - m) / v_between - 1.0
    strength = float(np.clip(strength, *PRIOR_STRENGTH_BOUNDS))
    alpha = m * strength
    beta = (1.0 - m) * strength
    return alpha, beta


def shrunk_rate(hr, pa, alpha, beta):
    """Beta-binomial posterior mean: (hr + alpha) / (pa + alpha + beta).

    Lies strictly between the raw rate hr/pa and the prior mean
    alpha/(alpha+beta), approaching the raw rate as pa grows.
    """
    return (hr + alpha) / (pa + alpha + beta)


def game_hr_probability(p_pa, expected_pa):
    """P(at least one HR in a game) = 1 - (1 - p_PA)^E[PA].

    Handles edge cases: p_pa=0 or expected_pa=0 -> 0; p_pa=1 -> 1
    (for expected_pa > 0).
    """
    p = np.clip(p_pa, 0.0, 1.0)
    n = np.maximum(expected_pa, 0.0)
    return 1.0 - np.power(1.0 - p, n)


def expected_pa_per_game(total_pa, games, league_pa_per_game, prior_games=EPA_PRIOR_GAMES):
    """Trailing PA/game regressed to the league mean.

    (total_pa + prior_games * league_mean) / (games + prior_games):
    a player with zero games gets exactly the league mean; with many games
    the estimate converges to his raw PA/game.
    """
    return (total_pa + prior_games * league_pa_per_game) / (games + prior_games)


def pitcher_factor(hr_allowed, pa_faced, alpha_p, beta_p, league_rate,
                   starter_share=STARTER_PA_SHARE, clip=PITCHER_FACTOR_CLIP):
    """Multiplier for the opposing starter's HR-proneness, shrunk and
    blended toward 1.0 for the bullpen share of the game."""
    if league_rate <= 0:
        return 1.0
    shrunk = shrunk_rate(hr_allowed, pa_faced, alpha_p, beta_p)
    raw_factor = float(np.clip(shrunk / league_rate, *clip))
    return starter_share * raw_factor + (1.0 - starter_share) * 1.0


def platoon_factor(hr_vs_hand, pa_vs_hand, overall_rate,
                   pseudo_pa=PLATOON_PSEUDO_PA, clip=PLATOON_FACTOR_CLIP):
    """Multiplier for the batter's split vs the starter's throwing hand.

    The vs-hand rate is shrunk toward the batter's own overall rate with
    pseudo_pa pseudo-observations, then expressed as a ratio to overall.
    """
    if overall_rate <= 0:
        return 1.0
    rate_hand = (hr_vs_hand + pseudo_pa * overall_rate) / (pa_vs_hand + pseudo_pa)
    return float(np.clip(rate_hand / overall_rate, *clip))


# ── Dataset assembly ──────────────────────────────────────────────────────

class StructuralModel:
    """Lookup tables + fitted priors built from PA-level data up to a cutoff.

    `pa_df` must contain one row per plate appearance with columns:
    batter, pitcher, p_throws, is_hr, game_pk, game_date.
    Only pass data strictly BEFORE the dates you want to predict.
    """

    def __init__(self, pa_df):
        pa_df = pa_df.copy()

        # League per-PA HR rate.
        self.league_rate = float(pa_df["is_hr"].mean())

        # Batter totals (overall and by pitcher hand).
        bat = pa_df.groupby("batter").agg(hr=("is_hr", "sum"), pa=("is_hr", "count"))
        self.batter_totals = bat

        self.batter_by_hand = pa_df.groupby(["batter", "p_throws"]).agg(
            hr=("is_hr", "sum"), pa=("is_hr", "count")
        )

        # Pitcher totals (HR allowed).
        pit = pa_df.groupby("pitcher").agg(hr=("is_hr", "sum"), pa=("is_hr", "count"))
        self.pitcher_totals = pit

        # Empirical-Bayes priors for batters and pitchers.
        self.alpha_b, self.beta_b = fit_beta_binomial_mom(bat["hr"].values, bat["pa"].values)
        self.alpha_p, self.beta_p = fit_beta_binomial_mom(pit["hr"].values, pit["pa"].values)

        # PA per game: per-batter games and total PA.
        per_game = pa_df.groupby(["batter", "game_pk"]).size()
        games = per_game.groupby("batter").size()
        self.batter_games = games
        # League mean PA/game across batter-games.
        self.league_pa_per_game = float(per_game.mean())

    # -- component lookups -------------------------------------------------

    def batter_rate(self, batter):
        if batter in self.batter_totals.index:
            row = self.batter_totals.loc[batter]
            return shrunk_rate(row["hr"], row["pa"], self.alpha_b, self.beta_b)
        return self.alpha_b / (self.alpha_b + self.beta_b)

    def expected_pa(self, batter):
        if batter in self.batter_totals.index and batter in self.batter_games.index:
            total_pa = float(self.batter_totals.loc[batter, "pa"])
            games = float(self.batter_games.loc[batter])
        else:
            total_pa, games = 0.0, 0.0
        return expected_pa_per_game(total_pa, games, self.league_pa_per_game)

    def _p_pa(self, batter, opposing_pitcher=None, pitcher_hand=None):
        """Uncapped per-PA HR probability: shrunk batter rate times pitcher
        and platoon factors. Missing pitcher/hand fall back to 1.0."""
        p = self.batter_rate(batter)

        if opposing_pitcher is not None and opposing_pitcher in self.pitcher_totals.index:
            row = self.pitcher_totals.loc[opposing_pitcher]
            p *= pitcher_factor(row["hr"], row["pa"], self.alpha_p, self.beta_p, self.league_rate)

        if pitcher_hand in ("R", "L"):
            key = (batter, pitcher_hand)
            if key in self.batter_by_hand.index:
                row = self.batter_by_hand.loc[key]
                overall = self.batter_rate(batter)
                p *= platoon_factor(row["hr"], row["pa"], overall)
        return p

    def predict(self, batter, opposing_pitcher=None, pitcher_hand=None):
        """P(batter hits >= 1 HR in the game). Missing pitcher/hand fall
        back to neutral factors of 1.0."""
        p = min(self._p_pa(batter, opposing_pitcher, pitcher_hand), P_PA_CAP)
        return float(game_hr_probability(p, self.expected_pa(batter)))

    def predict_frame(self, rows):
        """Vector-ish convenience: rows is a DataFrame with columns
        batter, opposing_pitcher, pitcher_hand. Returns np.array of probs."""
        return np.array([
            self.predict(r.batter,
                         getattr(r, "opposing_pitcher", None),
                         getattr(r, "pitcher_hand", None))
            for r in rows.itertuples()
        ])


def pa_table_from_statcast(all_pitches):
    """Collapse a raw Statcast pitch-level frame to one row per PA with the
    columns StructuralModel needs. Mirrors build_features.build_pa_df but
    keeps only what the structural model uses. v2 columns (at_bat_number,
    inning_topbot, stand, home_team, player_name) are kept when present so
    StructuralModelV2 can infer lineup slots and park factors."""
    sort_cols = [c for c in ["game_pk", "at_bat_number", "pitch_number"] if c in all_pitches.columns]
    pa = (all_pitches.sort_values(sort_cols)
          .groupby(["game_pk", "at_bat_number", "batter"], as_index=False)
          .tail(1))
    keep = [c for c in ["game_pk", "game_date", "at_bat_number", "inning_topbot",
                        "batter", "pitcher", "player_name", "stand", "p_throws",
                        "home_team", "events"] if c in pa.columns]
    pa = pa[keep].copy()
    pa["is_hr"] = (pa["events"] == "home_run").astype(int)
    return pa


# ── v2: lineup-slot E[PA] and park factors by handedness ─────────────────

def infer_game_slots(pa_df):
    """Infer each batter's batting-order slot per game from PA ordering.

    Within a (game_pk, inning_topbot) team-half, batters first appear in
    batting-order sequence, so the rank of each batter's FIRST plate
    appearance (by at_bat_number) is his lineup slot. Slots above
    MAX_LINEUP_SLOT are mid-game substitutes (pinch hitters, injury subs):
    their entry order is real but NOT knowable from a posted pre-game
    lineup, so callers should treat slot > 9 as unknown.

    Returns a DataFrame with columns game_pk, batter, slot.
    """
    first = (pa_df.sort_values(["game_pk", "at_bat_number"])
             .drop_duplicates(["game_pk", "inning_topbot", "batter"])
             .copy())
    first["slot"] = (first.groupby(["game_pk", "inning_topbot"])["at_bat_number"]
                     .rank(method="first").astype(int))
    return first[["game_pk", "batter", "slot"]]


def pa_per_game_by_slot(pa_df, slots=None):
    """League-average plate appearances per game for lineup slots 1-9.

    Computed over starting-lineup player-games only (slot <= 9); mid-game
    substitutes are excluded because their partial games would drag every
    slot mean down. Returns a Series indexed by slot.
    """
    if slots is None:
        slots = infer_game_slots(pa_df)
    counts = pa_df.groupby(["game_pk", "batter"]).size().rename("n_pa").reset_index()
    merged = counts.merge(slots, on=["game_pk", "batter"], how="inner")
    starters = merged[merged["slot"] <= MAX_LINEUP_SLOT]
    return starters.groupby("slot")["n_pa"].mean()


def park_factors_by_hand(pa_df, pseudo_pa=PARK_PSEUDO_PA, clip=PARK_FACTOR_CLIP):
    """Per-park HR/PA indices split by batter handedness, shrunk toward 1.0.

    For each batter side ('L'/'R') the park's HR/PA is shrunk toward the
    league rate FOR THAT SIDE with pseudo_pa pseudo-observations, then
    expressed as a ratio to that league rate and clipped. Shrinking toward
    the side-specific league rate means the factor isolates the park, not
    the league-wide L/R power difference (that lives in the batter's own
    rate already).

    Returns {(home_team, stand): factor}.
    """
    out = {}
    valid = pa_df.dropna(subset=["home_team", "stand"])
    for stand, grp in valid.groupby("stand"):
        league = float(grp["is_hr"].mean())
        if league <= 0:
            continue
        agg = grp.groupby("home_team")["is_hr"].agg(["sum", "count"])
        shrunk = (agg["sum"] + pseudo_pa * league) / (agg["count"] + pseudo_pa)
        factors = (shrunk / league).clip(*clip)
        for team, f in factors.items():
            out[(team, stand)] = float(f)
    return out


def _norm_name(name):
    """Normalize a player name for lookups: strips accents, lowercases,
    and converts Statcast's 'Last, First' to 'first last'."""
    import unicodedata
    s = str(name).strip()
    if "," in s:
        last, _, first = s.partition(",")
        s = f"{first.strip()} {last.strip()}"
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return " ".join(s.lower().split())


class StructuralModelV2(StructuralModel):
    """v1 plus (a) lineup-slot-aware E[PA] and (b) park factors by
    handedness. Both additions are built from the SAME pre-cutoff pa_df as
    the v1 tables — no test-window data ever enters the constants. The
    only per-game inputs are the batter's slot in the posted lineup and
    the park/handedness of the matchup, all knowable pre-game.
    """

    def __init__(self, pa_df):
        super().__init__(pa_df)

        if {"at_bat_number", "inning_topbot"}.issubset(pa_df.columns):
            slots = infer_game_slots(pa_df)
            self.pa_by_slot = {int(k): float(v)
                               for k, v in pa_per_game_by_slot(pa_df, slots).items()}
        else:
            self.pa_by_slot = {}

        if {"home_team", "stand"}.issubset(pa_df.columns):
            self.park_factors = park_factors_by_hand(pa_df)
            stand_mode = pa_df.dropna(subset=["stand"]).groupby("batter")["stand"] \
                              .agg(lambda s: s.mode().iat[0])
            self.batter_stand = stand_mode.to_dict()
        else:
            self.park_factors = {}
            self.batter_stand = {}

        # Statcast's player_name column is the PITCHER's name on each pitch
        # row; keep a normalized name -> id map so production callers that
        # only know the starter's display name can find his totals.
        if "player_name" in pa_df.columns:
            names = pa_df.dropna(subset=["player_name"]).groupby("pitcher")["player_name"].first()
            self.pitcher_id_by_name = {_norm_name(n): pid for pid, n in names.items()}
        else:
            self.pitcher_id_by_name = {}

    # -- v2 component lookups ----------------------------------------------

    def lookup_pitcher_id(self, name):
        if name is None:
            return None
        return self.pitcher_id_by_name.get(_norm_name(name))

    def park_factor(self, park, stand):
        """Shrunk HR park factor for (park, batter side); 1.0 if unknown."""
        if park is None:
            return 1.0
        return self.park_factors.get((park, stand), 1.0)

    def expected_pa_v2(self, batter, slot=None):
        """Slot-aware E[PA]: blend of the league PA/game mean for the
        batter's posted lineup slot and his trailing regressed PA/game.
        Unknown slot (None, or >9 i.e. a sub) falls back to v1 behavior."""
        trailing = self.expected_pa(batter)
        if slot is not None and slot in self.pa_by_slot:
            return SLOT_EPA_WEIGHT * self.pa_by_slot[slot] + (1.0 - SLOT_EPA_WEIGHT) * trailing
        return trailing

    def predict_v2(self, batter, opposing_pitcher=None, pitcher_hand=None,
                   slot=None, park=None, stand=None,
                   use_slots=True, use_parks=True):
        """P(batter hits >= 1 HR in the game), v2.

        use_slots/use_parks toggle the two upgrades independently so an
        ablation can isolate each one's contribution; with both False this
        reproduces v1's predict() exactly.
        """
        p = self._p_pa(batter, opposing_pitcher, pitcher_hand)
        if use_parks:
            if stand is None:
                stand = self.batter_stand.get(batter)
            p *= self.park_factor(park, stand)
        p = min(p, P_PA_CAP)
        epa = self.expected_pa_v2(batter, slot) if use_slots else self.expected_pa(batter)
        return float(game_hr_probability(p, epa))

    # -- persistence ---------------------------------------------------------
    # The pickle holds a plain dict of tables (never the instance itself):
    # pickling the instance would embed the defining module's name, which
    # breaks when the model is built via `python3 structural_model.py build`
    # (class recorded as __main__.StructuralModelV2) and loaded elsewhere.

    def save(self, path):
        import pickle
        with open(path, "wb") as f:
            pickle.dump({"format": "structural_v2_state", "state": self.__dict__}, f)

    @staticmethod
    def load(path):
        import pickle
        with open(path, "rb") as f:
            payload = pickle.load(f)
        model = StructuralModelV2.__new__(StructuralModelV2)
        model.__dict__.update(payload["state"])
        return model


def build_production_model(paths=("homerun_data_all.csv", "holdout_2026.csv"),
                           out_path="structural_v2.pkl", verbose=True):
    """Build a StructuralModelV2 from all Statcast data on disk and pickle
    it for the dashboard. Deployment (not evaluation) artifact: uses every
    date available, so never score historical games with it."""
    from pathlib import Path
    usecols = {"game_pk", "game_date", "at_bat_number", "pitch_number",
               "inning_topbot", "batter", "pitcher", "player_name",
               "stand", "p_throws", "home_team", "events"}
    frames = []
    for path in paths:
        if not Path(path).exists():
            if verbose:
                print(f"  {path}: not found, skipping")
            continue
        df = pd.read_csv(path, usecols=lambda c: c in usecols, low_memory=False)
        frames.append(df)
        if verbose:
            print(f"  {path}: {len(df):,} pitches")
    if not frames:
        raise FileNotFoundError("no Statcast CSVs found to build the model from")
    pa = pa_table_from_statcast(pd.concat(frames, ignore_index=True))
    model = StructuralModelV2(pa)
    model.save(out_path)
    if verbose:
        print(f"  built from {len(pa):,} PAs -> {out_path}")
        print(f"  PA/game by slot: " +
              ", ".join(f"{s}:{v:.2f}" for s, v in sorted(model.pa_by_slot.items())))
    return model


if __name__ == "__main__":
    import sys
    if sys.argv[1:2] == ["build"]:
        build_production_model()
    else:
        print("usage: python3 structural_model.py build   # writes structural_v2.pkl")
