"""Tests for the live scorecard pipeline (predict_today / grade_predictions).

Everything here runs offline against synthetic payloads: schedule parsing,
matchup assembly, log-append idempotency, grading math, and the scoreboard
calculation. Also asserts the pipeline never touches the Odds API.
"""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import grade_predictions as gp
import predict_today as pt


def fake_schedule():
    def game(pk, state, home, away, home_lineup, away_lineup,
             home_pp=None, away_pp=None):
        return {
            "gamePk": pk,
            "status": {"abstractGameState": state},
            "teams": {
                "home": {"team": {"abbreviation": home},
                         "probablePitcher": home_pp or {}},
                "away": {"team": {"abbreviation": away},
                         "probablePitcher": away_pp or {}},
            },
            "lineups": {"homePlayers": home_lineup, "awayPlayers": away_lineup},
        }
    return {"dates": [{"games": [
        game(1, "Preview", "NYY", "BOS",
             [{"id": 10, "fullName": "H One"}, {"id": 11, "fullName": "H Two"}],
             [{"id": 20, "fullName": "A One"}],
             home_pp={"id": 500, "fullName": "Home Starter"},
             away_pp={"id": 600, "fullName": "Away Starter"}),
        game(2, "Live", "LAD", "SD",
             [{"id": 30, "fullName": "X"}], [{"id": 31, "fullName": "Y"}]),
        game(3, "Preview", "COL", "MIA", [], []),  # no lineups posted yet
    ]}]}


class TestParseAndAssemble(unittest.TestCase):
    def test_parse_skips_non_preview_games(self):
        games = pt.parse_games(fake_schedule())
        self.assertEqual([g["game_pk"] for g in games], [1, 3])

    def test_lineups_and_probables_extracted(self):
        g = pt.parse_games(fake_schedule())[0]
        self.assertEqual(g["home"], "NYY")
        self.assertEqual(len(g["home_lineup"]), 2)
        self.assertEqual(g["away_pitcher"], {"id": 600, "name": "Away Starter"})

    def test_matchup_frame_opponent_mapping(self):
        games = pt.parse_games(fake_schedule())
        pg = pt.build_matchup_frame(games, "2026-07-05", {500: "L", 600: "R"})
        # Unposted lineups (game 3) contribute nothing.
        self.assertEqual(len(pg), 3)
        home = pg[pg["team"] == "NYY"]
        away = pg[pg["team"] == "BOS"]
        # Home batters face the AWAY probable and vice versa.
        self.assertTrue((home["opp_starter_id"] == 600).all())
        self.assertTrue((home["opp_starter_hand"] == "R").all())
        self.assertTrue((away["opp_starter_id"] == 500).all())
        # Park is always the home team's; slots follow lineup order.
        self.assertTrue((pg["home_team"] == "NYY").all())
        self.assertEqual(list(home["slot"]), [1.0, 2.0])

    def test_log_rows_schema(self):
        games = pt.parse_games(fake_schedule())
        pg = pt.build_matchup_frame(games, "2026-07-05", {})
        rows = pt.to_log_rows(pg, [0.1, 0.2, 0.3])
        self.assertEqual(list(rows.columns), pt.LOG_COLUMNS)
        self.assertTrue((rows["model"] == pt.MODEL_ID).all())
        self.assertTrue((rows["hr"] == "").all())


class TestAppendIdempotency(unittest.TestCase):
    def rows(self, ids, date="2026-07-05", prob=0.1):
        return pd.DataFrame([{c: "" for c in pt.LOG_COLUMNS}
                             | {"date": date, "batter_id": i, "probability": prob}
                             for i in ids])

    def test_first_append(self):
        combined, n = pt.append_predictions(None, self.rows([1, 2]))
        self.assertEqual((n, len(combined)), (2, 2))

    def test_reappend_is_noop(self):
        combined, _ = pt.append_predictions(None, self.rows([1, 2]))
        combined2, n = pt.append_predictions(combined, self.rows([1, 2], prob=0.9))
        self.assertEqual(n, 0)
        pd.testing.assert_frame_equal(combined, combined2)

    def test_first_write_wins_and_new_rows_are_added(self):
        combined, _ = pt.append_predictions(None, self.rows([1], prob=0.1))
        combined.loc[0, "hr"] = "1"  # already graded
        combined2, n = pt.append_predictions(
            combined, self.rows([1, 2], prob=0.9))
        self.assertEqual(n, 1)
        self.assertEqual(len(combined2), 2)
        row1 = combined2[combined2["batter_id"] == 1].iloc[0]
        self.assertEqual(row1["probability"], 0.1)  # not overwritten
        self.assertEqual(row1["hr"], "1")           # grade preserved
        combined3, n3 = pt.append_predictions(combined2, self.rows([2], date="2026-07-06"))
        self.assertEqual(n3, 1)  # same batter, new date -> new row
        self.assertEqual(len(combined3), 3)


class TestGrading(unittest.TestCase):
    def log(self):
        return pd.DataFrame({
            "date": ["2026-07-01", "2026-07-01", "2026-07-01",
                     "2026-07-02", "2026-07-05"],
            "batter_id": [1, 2, 3, 1, 1],
            "probability": [0.2, 0.1, 0.15, 0.3, 0.25],
            "hr": pd.array(["", "", "", "1", ""], dtype="string"),
        })

    def test_grade_rows(self):
        outcomes = {
            1: ({"2026-07-01"}, {"2026-07-01"}),   # played, homered
            2: ({"2026-07-01"}, set()),            # played, no HR
            3: (set(), set()),                     # did not play
        }
        graded, n = gp.grade_rows(self.log(), outcomes, today="2026-07-05")
        self.assertEqual(n, 3)
        self.assertEqual(list(graded["hr"]), ["1", "0", "dnp", "1", ""])

    def test_already_graded_and_future_rows_untouched(self):
        graded, _ = gp.grade_rows(self.log(), {1: (set(), set())},
                                  today="2026-07-05")
        self.assertEqual(graded.loc[3, "hr"], "1")   # graded stays
        self.assertEqual(graded.loc[4, "hr"], "")    # today's row stays pending

    def test_missing_outcome_leaves_row_pending(self):
        graded, n = gp.grade_rows(self.log(), {}, today="2026-07-05")
        self.assertEqual(n, 0)
        self.assertEqual(list(graded["hr"]), ["", "", "", "1", ""])


class TestScoreboardMath(unittest.TestCase):
    def graded_log(self):
        return pd.DataFrame({
            "date": ["2026-07-01"] * 4,
            "batter_id": [1, 2, 3, 4],
            "probability": [0.5, 0.1, 0.2, 0.3],
            "hr": pd.array(["1", "0", "dnp", ""], dtype="string"),
        })

    def test_scored_subset_excludes_dnp_and_pending(self):
        sub = gp.scored_subset(self.graded_log())
        self.assertEqual(list(sub["batter_id"]), [1, 2])

    def test_brier_hand_computed(self):
        # rows: (p=0.5, y=1), (p=0.1, y=0) -> ((0.5-1)^2 + (0.1-0)^2)/2 = 0.13
        sub = gp.scored_subset(self.graded_log())
        self.assertAlmostEqual(gp.brier(sub["y"], sub["p"]), 0.13, places=12)

    def test_log_loss_hand_computed(self):
        sub = gp.scored_subset(self.graded_log())
        expected = -(np.log(0.5) + np.log(0.9)) / 2
        self.assertAlmostEqual(gp.log_loss_score(sub["y"], sub["p"]),
                               expected, places=12)

    def test_calibration_table_bins(self):
        sub = gp.scored_subset(self.graded_log())
        tbl = gp.calibration_table(sub, bins=[0.0, 0.25, 1.0])
        self.assertEqual([r["n"] for r in tbl], [1, 1])
        self.assertAlmostEqual(tbl[0]["mean_pred"], 0.1)
        self.assertAlmostEqual(tbl[0]["observed"], 0.0)
        self.assertAlmostEqual(tbl[1]["observed"], 1.0)

    def test_render_scoreboard_contains_key_numbers(self):
        md = gp.render_scoreboard(self.graded_log(), "2026-07-05 10:00 ET")
        self.assertIn("0.13000", md)          # model Brier
        self.assertIn("Pending: 1", md)
        self.assertIn("Did not play (excluded): 1", md)
        self.assertIn("PROTOCOL.json", md)

    def test_render_scoreboard_empty_log(self):
        empty = pd.DataFrame({"date": [], "batter_id": [], "probability": [],
                              "hr": pd.array([], dtype="string")})
        md = gp.render_scoreboard(empty, "2026-07-05 10:00 ET")
        self.assertIn("Graded predictions: **0**", md)


class TestNoOddsApi(unittest.TestCase):
    """The pre-registered pipeline must never spend odds credits."""

    def test_pipeline_sources_never_reference_the_odds_api(self):
        root = Path(__file__).resolve().parents[1]
        for name in ["predict_today.py", "grade_predictions.py",
                     ".github/workflows/predictions.yml"]:
            text = (root / name).read_text().lower()
            self.assertNotIn("the-odds-api", text, name)
            self.assertNotIn("odds_api", text, name)
            self.assertNotIn("api.the-odds", text, name)


if __name__ == "__main__":
    unittest.main()
