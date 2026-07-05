"""Tests for GBM v3's feature assembly (train_model_v3.py).

The critical property: structural_feature_frame (the vectorized feature
builder) must be deterministic and EXACTLY consistent with
StructuralModelV2.predict_v2, so the ML sees the same structural inputs
the formula uses. All fixtures are synthetic — no data files, no network.
"""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from structural_model import StructuralModelV2
from train_model_v3 import (STRUCT_FEATURES, apply_calibrator,
                            fit_calibrators, structural_feature_frame)


def synthetic_pa(seed=7, n=4000):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "game_pk": rng.integers(1, 60, n),
        "at_bat_number": np.arange(n) % 70 + 1,
        "inning_topbot": rng.choice(["Top", "Bot"], n),
        "batter": rng.integers(100, 140, n),
        "pitcher": rng.integers(200, 220, n),
        "p_throws": rng.choice(["R", "L"], n),
        "stand": rng.choice(["R", "L"], n),
        "home_team": rng.choice(["NYY", "BOS", "COL", "MIA"], n),
        "game_date": pd.Timestamp("2025-06-01"),
        "is_hr": (rng.random(n) < 0.03).astype(int),
    })


def synthetic_pg(seed=11, n=300):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "batter": rng.integers(100, 145, n),          # some unseen batters
        "opp_starter_id": rng.integers(200, 225, n),  # some unseen pitchers
        "opp_starter_hand": rng.choice(["R", "L", None], n),
        "slot": rng.choice([1.0, 5.0, 9.0, np.nan], n),
        "home_team": rng.choice(["NYY", "BOS", "COL", "SD"], n),
        "stand": rng.choice(["R", "L", None], n),
    })


class TestStructuralFeatureFrame(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = StructuralModelV2(synthetic_pa())
        cls.pg = synthetic_pg()

    def test_deterministic(self):
        a = structural_feature_frame(self.model, self.pg)
        b = structural_feature_frame(self.model, self.pg)
        pd.testing.assert_frame_equal(a, b)

    def test_matches_predict_v2_exactly(self):
        sf = structural_feature_frame(self.model, self.pg)
        for i, r in enumerate(self.pg.itertuples()):
            ref = self.model.predict_v2(
                r.batter, opposing_pitcher=r.opp_starter_id,
                pitcher_hand=r.opp_starter_hand,
                slot=int(r.slot) if pd.notna(r.slot) else None,
                park=r.home_team,
                stand=r.stand if pd.notna(r.stand) else None)
            self.assertAlmostEqual(sf["s_struct_v2_prob"].iloc[i], ref, places=12)

    def test_column_layout_is_fixed(self):
        sf = structural_feature_frame(self.model, self.pg)
        self.assertEqual(list(sf.columns), STRUCT_FEATURES + ["s_struct_v2_prob"])

    def test_unknown_batter_gets_prior_and_league_epa(self):
        pg = pd.DataFrame({
            "batter": [999999], "opp_starter_id": [np.nan],
            "opp_starter_hand": [None], "slot": [np.nan],
            "home_team": ["NYY"], "stand": [None]})
        sf = structural_feature_frame(self.model, pg)
        prior = self.model.alpha_b / (self.model.alpha_b + self.model.beta_b)
        self.assertAlmostEqual(sf["s_batter_rate"].iloc[0], prior, places=12)
        self.assertAlmostEqual(sf["s_epa"].iloc[0],
                               self.model.league_pa_per_game, places=12)
        self.assertEqual(sf["s_pitcher_factor"].iloc[0], 1.0)
        self.assertEqual(sf["s_platoon_factor"].iloc[0], 1.0)

    def test_nan_slot_falls_back_to_trailing_epa(self):
        batter = int(self.model.batter_totals.index[0])
        base = {"opp_starter_id": [np.nan], "opp_starter_hand": [None],
                "home_team": ["NYY"], "stand": [None]}
        no_slot = structural_feature_frame(
            self.model, pd.DataFrame({"batter": [batter], "slot": [np.nan], **base}))
        self.assertAlmostEqual(no_slot["s_epa"].iloc[0],
                               self.model.expected_pa(batter), places=12)

    def test_factors_respect_clips(self):
        sf = structural_feature_frame(self.model, self.pg)
        self.assertTrue((sf["s_pitcher_factor"] >= 0.5 * 0.65 + 0.35 - 1e-9).all())
        self.assertTrue((sf["s_platoon_factor"].between(0.7, 1.4)).all())
        self.assertTrue((sf["s_park_factor"].between(0.80, 1.25)).all())
        self.assertTrue((sf["s_struct_v2_prob"].between(0, 1)).all())


class TestCalibrators(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(3)
        self.p = rng.uniform(0.01, 0.4, 2000)
        self.y = (rng.random(2000) < self.p).astype(int)
        self.cal = fit_calibrators(self.p, self.y)

    def test_outputs_are_probabilities(self):
        grid = np.linspace(0, 1, 101)
        for method in ("isotonic", "sigmoid"):
            out = apply_calibrator(self.cal, method, grid)
            self.assertTrue(np.all((out > 0) & (out < 1)), method)

    def test_monotone_non_decreasing(self):
        grid = np.linspace(0, 1, 101)
        for method in ("isotonic", "sigmoid"):
            out = apply_calibrator(self.cal, method, grid)
            self.assertTrue(np.all(np.diff(out) >= -1e-12), method)

    def test_calibration_roughly_recovers_truth(self):
        # Labels were drawn with P(y=1)=p, so calibration should be near
        # the identity in the well-populated middle of the range.
        mid = np.linspace(0.1, 0.3, 21)
        for method in ("isotonic", "sigmoid"):
            out = apply_calibrator(self.cal, method, mid)
            self.assertLess(np.abs(out - mid).mean(), 0.05, method)


if __name__ == "__main__":
    unittest.main()
