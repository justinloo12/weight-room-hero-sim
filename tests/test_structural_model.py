"""Offline unit tests for the structural baseline math (no data files needed)."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from structural_model import (
    EPA_PRIOR_GAMES,
    PITCHER_FACTOR_CLIP,
    PLATOON_FACTOR_CLIP,
    expected_pa_per_game,
    fit_beta_binomial_mom,
    game_hr_probability,
    pitcher_factor,
    platoon_factor,
    shrunk_rate,
)


class TestShrinkage(unittest.TestCase):
    def setUp(self):
        # Prior centred at 3% HR/PA, worth 300 pseudo-PA.
        self.alpha = 0.03 * 300
        self.beta = 0.97 * 300
        self.prior_mean = self.alpha / (self.alpha + self.beta)

    def test_posterior_between_raw_and_prior(self):
        # Raw rate above prior mean: posterior must land strictly between.
        hr, pa = 12, 150  # raw 8%
        post = shrunk_rate(hr, pa, self.alpha, self.beta)
        self.assertGreater(post, self.prior_mean)
        self.assertLess(post, hr / pa)

        # Raw rate below prior mean: posterior between raw and prior.
        hr, pa = 1, 200  # raw 0.5%
        post = shrunk_rate(hr, pa, self.alpha, self.beta)
        self.assertLess(post, self.prior_mean)
        self.assertGreater(post, hr / pa)

    def test_posterior_monotonic_in_sample_size(self):
        # Same raw rate (8%), growing sample: posterior approaches raw rate.
        raw = 0.08
        posts = [shrunk_rate(raw * pa, pa, self.alpha, self.beta)
                 for pa in [25, 100, 400, 1600, 6400]]
        for smaller, larger in zip(posts, posts[1:]):
            self.assertLess(smaller, larger)
        self.assertAlmostEqual(posts[-1], raw, delta=0.005)

    def test_zero_pa_gives_prior_mean(self):
        self.assertAlmostEqual(
            shrunk_rate(0, 0, self.alpha, self.beta), self.prior_mean, places=12
        )

    def test_mom_fit_recovers_reasonable_prior(self):
        rng = np.random.default_rng(7)
        true_rates = rng.beta(9.0, 291.0, size=400)  # mean 3%
        pa = rng.integers(120, 600, size=400)
        hr = rng.binomial(pa, true_rates)
        alpha, beta = fit_beta_binomial_mom(hr, pa)
        self.assertGreater(alpha, 0)
        self.assertGreater(beta, 0)
        prior_mean = alpha / (alpha + beta)
        self.assertAlmostEqual(prior_mean, 0.03, delta=0.01)


class TestGameProbability(unittest.TestCase):
    def test_zero_probability(self):
        self.assertEqual(game_hr_probability(0.0, 4.2), 0.0)

    def test_zero_pa(self):
        self.assertEqual(game_hr_probability(0.05, 0.0), 0.0)

    def test_certainty(self):
        self.assertEqual(game_hr_probability(1.0, 1.0), 1.0)

    def test_negative_pa_treated_as_zero(self):
        self.assertEqual(game_hr_probability(0.05, -3.0), 0.0)

    def test_known_value(self):
        # 1 - (1 - 0.04)^4 = 1 - 0.96^4
        self.assertAlmostEqual(game_hr_probability(0.04, 4.0), 1 - 0.96 ** 4, places=12)

    def test_monotonic_in_p_and_n(self):
        self.assertLess(game_hr_probability(0.02, 4.0), game_hr_probability(0.05, 4.0))
        self.assertLess(game_hr_probability(0.05, 3.0), game_hr_probability(0.05, 5.0))

    def test_bounded(self):
        p = game_hr_probability(0.9, 10.0)
        self.assertGreaterEqual(p, 0.0)
        self.assertLessEqual(p, 1.0)


class TestExpectedPA(unittest.TestCase):
    def test_no_games_gives_league_mean(self):
        self.assertAlmostEqual(expected_pa_per_game(0, 0, 3.9), 3.9, places=12)

    def test_large_sample_converges_to_raw(self):
        # 4.6 PA/game over 1000 games should dominate the prior.
        est = expected_pa_per_game(4600, 1000, 3.9)
        self.assertAlmostEqual(est, 4.6, delta=0.01)

    def test_regression_between_raw_and_league(self):
        # 10 games at 4.8 PA/game, league 3.9: estimate strictly between.
        est = expected_pa_per_game(48, 10, 3.9)
        self.assertGreater(est, 3.9)
        self.assertLess(est, 4.8)

    def test_prior_weight_matches_formula(self):
        est = expected_pa_per_game(48, 10, 3.9, prior_games=EPA_PRIOR_GAMES)
        expected = (48 + EPA_PRIOR_GAMES * 3.9) / (10 + EPA_PRIOR_GAMES)
        self.assertAlmostEqual(est, expected, places=12)


class TestFactors(unittest.TestCase):
    def test_pitcher_factor_neutral_for_league_average(self):
        # A pitcher exactly at league rate with a big sample: factor ~ 1.
        alpha, beta = 0.03 * 300, 0.97 * 300
        f = pitcher_factor(30, 1000, alpha, beta, league_rate=0.03)
        self.assertAlmostEqual(f, 1.0, delta=0.02)

    def test_pitcher_factor_clipped_and_blended(self):
        alpha, beta = 0.03 * 300, 0.97 * 300
        # Absurd HR-prone pitcher: raw factor is clipped, then blended
        # toward 1.0 by the starter share, so it stays below the clip cap.
        f = pitcher_factor(200, 1000, alpha, beta, league_rate=0.03)
        self.assertLessEqual(f, PITCHER_FACTOR_CLIP[1])
        self.assertGreater(f, 1.0)

    def test_platoon_factor_clipped(self):
        # Extreme small-sample split cannot exceed the clip bounds.
        f_hi = platoon_factor(10, 20, overall_rate=0.03)
        f_lo = platoon_factor(0, 500, overall_rate=0.03)
        self.assertLessEqual(f_hi, PLATOON_FACTOR_CLIP[1])
        self.assertGreaterEqual(f_lo, PLATOON_FACTOR_CLIP[0])

    def test_platoon_factor_neutral_with_no_split_data(self):
        # Zero vs-hand PA: shrinkage returns exactly the overall rate.
        self.assertAlmostEqual(platoon_factor(0, 0, overall_rate=0.03), 1.0, places=12)


if __name__ == "__main__":
    unittest.main()
