import unittest
import pandas as pd
import numpy as np

from pyrate.rate import ratingbase
from pyrate.rate import leastsquares

class ToyLeagueHyper(unittest.TestCase):

    def setUp(self):
        # Data from Massey (1997) Example 4.2.  Home/away values have
        # been added.
        self.raw_df = pd.DataFrame(
            [[1, 1, 10, 'H'],
             [1, 2, 6, 'A'],
             [2, 3, 4, 'H'],
             [2, 4, 4, 'A'],
             [3, 4, 9, 'H'],
             [3, 2, 2, 'A'],
             [4, 1, 8, 'A'],
             [4, 4, 6, 'H'],
             [5, 2, 3, 'H'],
             [5, 3, 2, 'A']], columns=['game_id', 'team_id', 'points', 'location'])

        self.league = ratingbase.League.from_hyper_table(self.raw_df)
        
    def testLeastSquares(self):
        expected_ratings = [2.375, -2.5, -1.125, 1.25]
        lsq = leastsquares.LeastSquares(self.league)
        for team, expected_rating in zip(lsq.teams, expected_ratings):
            self.assertAlmostEqual(team.rating, expected_rating)

    def testScoreCap(self):
        expected_ratings = [0.875, -0.25, -0.625, 0.0]
        lsq = leastsquares.LeastSquares(self.league, score_cap=1)
        for team, expected_rating in zip(lsq.teams, expected_ratings):
            self.assertAlmostEqual(team.rating, expected_rating)

    def testHomeCourt(self):
        expected_ratings = [1.875, -2.5, -1.625, 0.25]
        lsq = leastsquares.LeastSquares(self.league, homecourt=True)
        for team, expected_rating in zip(lsq.teams, expected_ratings):
            self.assertAlmostEqual(team.rating, expected_rating)
        self.assertAlmostEqual(lsq.home_adv, 2.0)

    def testStrengtOfSchedule(self):
        expected_sos_vals = [-0.625, 0.83333333, -0.625, -0.416666666]
        lsq = leastsquares.LeastSquares(self.league)
        for team, expected_sos in zip(lsq.teams, expected_sos_vals):
            self.assertAlmostEqual(team.sos_past, expected_sos)

    def testEvaluatePredWins(self):
        lsq = leastsquares.LeastSquares(self.league)
        correct, count = lsq.evaluate_predicted_wins()
        self.assertEqual(correct, 3)
        self.assertEqual(count, 5)

class ToyLeagueGames(unittest.TestCase):

    def setUp(self):
        # Data from Massey (1997) Example 4.2.  Home/away values have
        # been added.
        self.raw_df = pd.DataFrame(
            [[1, 10, 'H', 2, 6, 'A'],
             [3, 4, 'H', 4, 4, 'A'],
             [4, 9, 'H', 2, 2, 'A'],
             [1, 8, 'A', 4, 6, 'H'],
             [2, 3, 'H', 3, 2, 'A']],
            columns=['team_id', 'points', 'location', 'opponent_id', 'opponent_points', 'opponent_location'])

        self.league = ratingbase.League.from_games_table(self.raw_df)
        
    def testLeastSquares(self):
        expected_ratings = [2.375, -2.5, -1.125, 1.25]
        lsq = leastsquares.LeastSquares(self.league)
        for team, expected_rating in zip(lsq.teams, expected_ratings):
            self.assertAlmostEqual(team.rating, expected_rating)

    def testScoreCap(self):
        expected_ratings = [0.875, -0.25, -0.625, 0.0]
        lsq = leastsquares.LeastSquares(self.league, score_cap=1)
        for team, expected_rating in zip(lsq.teams, expected_ratings):
            self.assertAlmostEqual(team.rating, expected_rating)

    def testHomeCourt(self):
        expected_ratings = [1.875, -2.5, -1.625, 0.25]
        lsq = leastsquares.LeastSquares(self.league, homecourt=True)
        for team, expected_rating in zip(lsq.teams, expected_ratings):
            self.assertAlmostEqual(team.rating, expected_rating)
        self.assertAlmostEqual(lsq.home_adv, 2.0)

    def testStrengtOfSchedule(self):
        expected_sos_vals = [-0.625, 0.83333333, -0.625, -0.416666666]
        lsq = leastsquares.LeastSquares(self.league)
        for team, expected_sos in zip(lsq.teams, expected_sos_vals):
            self.assertAlmostEqual(team.sos_past, expected_sos)

    def testEvaluatePredWins(self):
        lsq = leastsquares.LeastSquares(self.league)
        correct, count = lsq.evaluate_predicted_wins()
        self.assertEqual(correct, 3)
        self.assertEqual(count, 5)

        
class ToyLeagueScheduled(unittest.TestCase):

    def setUp(self):
        self.raw_df = pd.DataFrame(
            [[1, 10, 'H', 2, 6, 'A'],
             [3, 4, 'H', 4, 4, 'A'],
             [4, 9, 'H', 2, 2, 'A'],
             [1, 8, 'A', 4, 6, 'H'],
             [2, 3, 'H', 3, 2, 'A'],
             [1, np.nan, 'H', 2, np.nan, 'A'],
             [3, np.nan, 'H', 4, np.nan, 'A'],
             [2, np.nan, 'H', 4, np.nan, 'A']],
            columns=['team_id', 'points', 'location', 'opponent_id', 'opponent_points', 'opponent_location'])

        self.league = ratingbase.League.from_games_table(self.raw_df)

    def testLeastSquares(self):
        expected_ratings = [2.375, -2.5, -1.125, 1.25]
        lsq = leastsquares.LeastSquares(self.league)
        for team, expected_rating in zip(lsq.teams, expected_ratings):
            self.assertAlmostEqual(team.rating, expected_rating)

    def testSchedule(self):
        expected_scheduled_counts = [1, 2, 1, 2]
        for team, expected_scheduled_count in zip(self.league.teams, expected_scheduled_counts):
            self.assertEqual(len(team.scheduled), expected_scheduled_count)

class ErrorTest(unittest.TestCase):

    def testGameIDMismatch(self):
        self.raw_df = pd.DataFrame(
            [[1, 1, 10, 'H'],
             [1, 2, 6, 'A'],
             [2, 3, 4, 'H'],
             [2, 4, 4, 'A'],
             [3, 4, 9, 'H'],
             [3, 2, 2, 'A'],
             [4, 1, 8, 'A'],
             [4, 4, 6, 'H'],
             [5, 2, 3, 'H'],
             [6, 3, 2, 'A']], columns=['game_id', 'team_id', 'points', 'location'])

        with self.assertRaises(ValueError):
            self.league = ratingbase.League.from_hyper_table(self.raw_df)

if __name__ == '__main__':
    unittest.main()
             
