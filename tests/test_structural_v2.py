"""Offline unit tests for the structural v2 upgrades (lineup-slot E[PA]
and park factors by handedness). No data files needed — every test builds
a tiny synthetic PA table."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from structural_model import (
    MAX_LINEUP_SLOT,
    PARK_FACTOR_CLIP,
    SLOT_EPA_WEIGHT,
    StructuralModelV2,
    _norm_name,
    infer_game_slots,
    pa_per_game_by_slot,
    park_factors_by_hand,
)


def synthetic_game(game_pk, away_order, home_order, pa_per_batter=4, start_ab=1):
    """One game's PA table with known batting orders. Batters alternate
    half-innings in strict lineup rotation, which is what at_bat_number
    ordering encodes in real Statcast data."""
    rows = []
    ab = start_ab
    # Interleave: 3 Top PAs, 3 Bot PAs, repeat (inning structure does not
    # matter to the inference — only at_bat_number order within each half).
    away_seq = [b for _ in range(pa_per_batter) for b in away_order]
    home_seq = [b for _ in range(pa_per_batter) for b in home_order]
    ai = hi = 0
    while ai < len(away_seq) or hi < len(home_seq):
        for _ in range(3):
            if ai < len(away_seq):
                rows.append({"game_pk": game_pk, "inning_topbot": "Top",
                             "batter": away_seq[ai], "at_bat_number": ab})
                ai += 1
                ab += 1
        for _ in range(3):
            if hi < len(home_seq):
                rows.append({"game_pk": game_pk, "inning_topbot": "Bot",
                             "batter": home_seq[hi], "at_bat_number": ab})
                hi += 1
                ab += 1
    df = pd.DataFrame(rows)
    df["is_hr"] = 0
    return df


class TestSlotInference(unittest.TestCase):
    def test_known_batting_order_recovered(self):
        away = [100 + i for i in range(9)]
        home = [200 + i for i in range(9)]
        pa = synthetic_game(1, away, home)
        slots = infer_game_slots(pa).set_index("batter")["slot"]
        for i, b in enumerate(away, start=1):
            self.assertEqual(slots[b], i)
        for i, b in enumerate(home, start=1):
            self.assertEqual(slots[b], i)

    def test_pinch_hitter_gets_slot_above_nine(self):
        away = [100 + i for i in range(9)]
        home = [200 + i for i in range(9)]
        pa = synthetic_game(1, away, home)
        # Pinch hitter 999 bats once, late in the game, for the away team.
        pa = pd.concat([pa, pd.DataFrame([{
            "game_pk": 1, "inning_topbot": "Top", "batter": 999,
            "at_bat_number": pa["at_bat_number"].max() + 1, "is_hr": 0,
        }])], ignore_index=True)
        slots = infer_game_slots(pa).set_index("batter")["slot"]
        self.assertGreater(slots[999], MAX_LINEUP_SLOT)
        # Starters' slots are unaffected.
        self.assertEqual(slots[100], 1)
        self.assertEqual(slots[108], 9)

    def test_slot_independent_of_row_order(self):
        away = [100 + i for i in range(9)]
        home = [200 + i for i in range(9)]
        pa = synthetic_game(1, away, home)
        shuffled = pa.sample(frac=1.0, random_state=7)
        s1 = infer_game_slots(pa).set_index("batter")["slot"]
        s2 = infer_game_slots(shuffled).set_index("batter")["slot"]
        pd.testing.assert_series_equal(s1.sort_index(), s2.sort_index())


class TestPaBySlot(unittest.TestCase):
    def test_top_slots_get_more_pa(self):
        # 20 games where slots 1-4 bat 5 times, 5-9 bat 4 times: the
        # classic leadoff advantage, exaggerated for a clean test.
        frames = []
        for g in range(1, 21):
            away = [100 + i for i in range(9)]
            home = [200 + i for i in range(9)]
            base = synthetic_game(g, away, home, pa_per_batter=4)
            extra = []
            ab = base["at_bat_number"].max() + 1
            for half, order in [("Top", away), ("Bot", home)]:
                for b in order[:4]:
                    extra.append({"game_pk": g, "inning_topbot": half,
                                  "batter": b, "at_bat_number": ab, "is_hr": 0})
                    ab += 1
            frames.append(pd.concat([base, pd.DataFrame(extra)], ignore_index=True))
        pa = pd.concat(frames, ignore_index=True)
        tbl = pa_per_game_by_slot(pa)
        self.assertEqual(set(tbl.index), set(range(1, 10)))
        for s in range(1, 5):
            self.assertAlmostEqual(tbl[s], 5.0)
        for s in range(5, 10):
            self.assertAlmostEqual(tbl[s], 4.0)
        # Monotone non-increasing from leadoff to ninth.
        self.assertTrue(all(tbl[s] >= tbl[s + 1] for s in range(1, 9)))

    def test_pinch_hitters_excluded_from_slot_means(self):
        away = [100 + i for i in range(9)]
        home = [200 + i for i in range(9)]
        pa = synthetic_game(1, away, home)
        ph = pd.DataFrame([{"game_pk": 1, "inning_topbot": "Top", "batter": 999,
                            "at_bat_number": pa["at_bat_number"].max() + 1, "is_hr": 0}])
        tbl = pa_per_game_by_slot(pd.concat([pa, ph], ignore_index=True))
        # The one-PA pinch hitter must not create a slot-10 row or shift
        # any starter mean.
        self.assertEqual(tbl.index.max(), 9)
        self.assertAlmostEqual(tbl[1], 4.0)


def park_pa(team, stand, n, hr_rate, start_pk=0):
    """n PAs in one park for one batter side with a fixed HR rate."""
    n_hr = int(round(n * hr_rate))
    return pd.DataFrame({
        "game_pk": np.arange(n) // 70 + start_pk,
        "batter": 1, "home_team": team, "stand": stand,
        "is_hr": [1] * n_hr + [0] * (n - n_hr),
    })


class TestParkFactors(unittest.TestCase):
    def test_small_sample_park_pulled_toward_one(self):
        # Big neutral pool sets the league rate; a tiny park at 3x the
        # league rate must stay near 1.0 after shrinkage.
        pool = park_pa("NEU", "R", 20000, 0.03)
        hot = park_pa("HOT", "R", 100, 0.09, start_pk=10_000)
        factors = park_factors_by_hand(pd.concat([pool, hot], ignore_index=True))
        # Raw ratio is ~3.0; with only 100 PA against 2000 pseudo-PA the
        # posterior must sit within ~10% of neutral.
        self.assertLess(factors[("HOT", "R")], 1.11)
        self.assertGreaterEqual(factors[("HOT", "R")], 1.0)

    def test_shrinkage_monotonic_in_sample_size(self):
        # Same Coors-like (~1.33x league) park rate at growing sample
        # sizes: the factor must move monotonically away from 1.0 toward
        # the raw ratio without hitting the clip.
        prev = 1.0
        for n in (200, 1000, 4000, 16000):
            pool = park_pa("NEU", "R", 50000, 0.03)
            hot = park_pa("HOT", "R", n, 0.04, start_pk=10_000)
            factors = park_factors_by_hand(pd.concat([pool, hot], ignore_index=True))
            f = factors[("HOT", "R")]
            self.assertGreater(f, prev)
            prev = f
        self.assertLessEqual(prev, PARK_FACTOR_CLIP[1])

    def test_factors_respect_clip(self):
        pool = park_pa("NEU", "R", 20000, 0.03)
        extreme = park_pa("MOON", "R", 50000, 0.30, start_pk=10_000)
        dead = park_pa("DEAD", "R", 50000, 0.0, start_pk=20_000)
        factors = park_factors_by_hand(pd.concat([pool, extreme, dead], ignore_index=True))
        self.assertLessEqual(factors[("MOON", "R")], PARK_FACTOR_CLIP[1])
        self.assertGreaterEqual(factors[("DEAD", "R")], PARK_FACTOR_CLIP[0])

    def test_split_by_handedness(self):
        # Park favors lefties only; the R factor must stay neutral.
        pool_r = park_pa("NEU", "R", 30000, 0.03)
        pool_l = park_pa("NEU", "L", 30000, 0.03, start_pk=1000)
        short_porch_l = park_pa("YS", "L", 8000, 0.05, start_pk=2000)
        neutral_r = park_pa("YS", "R", 8000, 0.03, start_pk=3000)
        factors = park_factors_by_hand(
            pd.concat([pool_r, pool_l, short_porch_l, neutral_r], ignore_index=True))
        self.assertGreater(factors[("YS", "L")], 1.05)
        self.assertAlmostEqual(factors[("YS", "R")], 1.0, delta=0.02)


class TestPredictV2(unittest.TestCase):
    def _model(self):
        rng = np.random.default_rng(0)
        games = []
        away = [100 + i for i in range(9)]
        home = [200 + i for i in range(9)]
        for g in range(1, 41):
            df = synthetic_game(g, away, home)
            games.append(df)
        pa = pd.concat(games, ignore_index=True)
        pa["is_hr"] = (rng.random(len(pa)) < 0.03).astype(int)
        pa["pitcher"] = 500
        pa["p_throws"] = "R"
        pa["stand"] = "R"
        pa["home_team"] = "NEU"
        pa["game_date"] = "2025-06-01"
        pa["events"] = np.where(pa["is_hr"] == 1, "home_run", "field_out")
        return StructuralModelV2(pa)

    def test_both_toggles_off_equals_v1(self):
        m = self._model()
        for b in (100, 104, 208):
            self.assertAlmostEqual(
                m.predict_v2(b, opposing_pitcher=500, pitcher_hand="R",
                             slot=1, park="NEU", use_slots=False, use_parks=False),
                m.predict(b, opposing_pitcher=500, pitcher_hand="R"),
                places=12)

    def test_leadoff_slot_raises_probability_vs_ninth(self):
        m = self._model()
        p1 = m.predict_v2(104, slot=1)
        p9 = m.predict_v2(104, slot=9)
        if m.pa_by_slot[1] > m.pa_by_slot[9]:
            self.assertGreater(p1, p9)
        else:  # synthetic data may have equal slot means; then probs match
            self.assertAlmostEqual(p1, p9)

    def test_unknown_slot_falls_back_to_trailing(self):
        m = self._model()
        self.assertAlmostEqual(m.predict_v2(104, slot=None),
                               m.predict_v2(104, use_slots=False), places=12)
        self.assertAlmostEqual(m.predict_v2(104, slot=12),
                               m.predict_v2(104, use_slots=False), places=12)

    def test_slot_blend_weight(self):
        m = self._model()
        expected = (SLOT_EPA_WEIGHT * m.pa_by_slot[3]
                    + (1 - SLOT_EPA_WEIGHT) * m.expected_pa(104))
        self.assertAlmostEqual(m.expected_pa_v2(104, slot=3), expected, places=12)

    def test_unknown_park_is_neutral(self):
        m = self._model()
        self.assertAlmostEqual(m.predict_v2(104, park="XXX", slot=None),
                               m.predict_v2(104, use_parks=False, slot=None), places=12)

    def test_save_load_round_trip(self):
        import tempfile
        m = self._model()
        with tempfile.TemporaryDirectory() as d:
            path = str(Path(d) / "m.pkl")
            m.save(path)
            loaded = StructuralModelV2.load(path)
        self.assertEqual(loaded.pa_by_slot, m.pa_by_slot)
        self.assertAlmostEqual(
            loaded.predict_v2(104, opposing_pitcher=500, pitcher_hand="R",
                              slot=2, park="NEU"),
            m.predict_v2(104, opposing_pitcher=500, pitcher_hand="R",
                         slot=2, park="NEU"),
            places=12)


class TestNameNormalization(unittest.TestCase):
    def test_statcast_and_display_forms_match(self):
        self.assertEqual(_norm_name("Cole, Gerrit"), _norm_name("Gerrit Cole"))
        self.assertEqual(_norm_name("Suárez, Ranger"), _norm_name("Ranger Suarez"))
        self.assertEqual(_norm_name("  Aaron   Judge "), "aaron judge")


if __name__ == "__main__":
    unittest.main()
