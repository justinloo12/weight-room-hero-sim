"""Offline unit tests for the pure betting-math helpers in betting_math.py."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from betting_math import (
    american_to_implied,
    implied_to_american,
    edge_percentage,
    expected_roi,
    kelly_fraction,
    pnl_from_odds,
)


class TestAmericanToImplied(unittest.TestCase):
    def test_positive_odds(self):
        self.assertAlmostEqual(american_to_implied(100), 0.5)
        self.assertAlmostEqual(american_to_implied(300), 0.25)
        self.assertAlmostEqual(american_to_implied(250), 100 / 350)

    def test_negative_odds(self):
        self.assertAlmostEqual(american_to_implied(-100), 0.5)
        self.assertAlmostEqual(american_to_implied(-150), 0.6)
        self.assertAlmostEqual(american_to_implied(-300), 0.75)

    def test_longshot_hr_prop_odds(self):
        # Typical HR prop price
        self.assertAlmostEqual(american_to_implied(450), 100 / 550)


class TestImpliedToAmerican(unittest.TestCase):
    def test_underdog(self):
        self.assertEqual(implied_to_american(25.0), 300)
        self.assertEqual(implied_to_american(20.0), 400)

    def test_even_money(self):
        self.assertEqual(implied_to_american(50.0), -100)

    def test_favorite(self):
        self.assertEqual(implied_to_american(60.0), -150)
        self.assertEqual(implied_to_american(75.0), -300)

    def test_clamps_extremes(self):
        # Probabilities are clamped to [1%, 99%]
        self.assertEqual(implied_to_american(0.0), implied_to_american(1.0))
        self.assertEqual(implied_to_american(100.0), implied_to_american(99.0))

    def test_round_trip(self):
        for pct in [10.0, 20.0, 25.0, 40.0, 50.0, 60.0, 75.0]:
            odds = implied_to_american(pct)
            self.assertAlmostEqual(american_to_implied(odds) * 100, pct, places=5)


class TestEdgePercentage(unittest.TestCase):
    def test_positive_edge(self):
        self.assertEqual(edge_percentage(15.0, 12.5), 2.5)

    def test_negative_edge(self):
        self.assertEqual(edge_percentage(10.0, 14.0), -4.0)

    def test_rounding(self):
        self.assertEqual(edge_percentage(12.3456, 10.0), 2.35)


class TestExpectedRoi(unittest.TestCase):
    def test_fair_bet_is_zero(self):
        # 25% model prob at +300 (implied 25%) is break-even
        self.assertAlmostEqual(expected_roi(25.0, 300), 0.0)

    def test_positive_ev(self):
        # 30% at +300: 0.30 * 3 - 0.70 = 0.20
        self.assertAlmostEqual(expected_roi(30.0, 300), 0.20)

    def test_negative_odds(self):
        # 60% at -150: 0.60 * (100/150) - 0.40 = 0.0
        self.assertAlmostEqual(expected_roi(60.0, -150), 0.0)

    def test_degenerate_inputs(self):
        self.assertEqual(expected_roi(0.0, 300), -1.0)
        self.assertEqual(expected_roi(None, 300), -1.0)
        self.assertEqual(expected_roi(25.0, None), -1.0)


class TestKellyFraction(unittest.TestCase):
    def test_no_edge_gives_zero(self):
        # 25% at +300 is exactly fair — Kelly says bet nothing
        self.assertAlmostEqual(kelly_fraction(25.0, 300), 0.0)

    def test_positive_edge(self):
        # p=0.30, b=3: f* = (3*0.3 - 0.7) / 3 = 0.2/3
        self.assertAlmostEqual(kelly_fraction(30.0, 300), 0.2 / 3)

    def test_negative_edge_clamped_to_zero(self):
        self.assertEqual(kelly_fraction(10.0, 300), 0.0)

    def test_negative_odds(self):
        # p=0.70, b=100/150: f* = (2/3*0.7 - 0.3) / (2/3) = 0.25
        self.assertAlmostEqual(kelly_fraction(70.0, -150), 0.25)

    def test_degenerate_inputs(self):
        self.assertEqual(kelly_fraction(0.0, 300), 0.0)
        self.assertEqual(kelly_fraction(None, 300), 0.0)
        self.assertEqual(kelly_fraction(100.0, 300), 0.0)
        self.assertEqual(kelly_fraction(30.0, None), 0.0)


class TestPnlFromOdds(unittest.TestCase):
    def test_positive_odds(self):
        self.assertAlmostEqual(pnl_from_odds(300, 100), 300.0)
        self.assertAlmostEqual(pnl_from_odds(450, 50), 225.0)

    def test_negative_odds(self):
        self.assertAlmostEqual(pnl_from_odds(-150, 100), 100 * 100 / 150)
        self.assertAlmostEqual(pnl_from_odds(-200, 100), 50.0)

    def test_even_money(self):
        self.assertAlmostEqual(pnl_from_odds(100, 100), 100.0)


if __name__ == "__main__":
    unittest.main()
