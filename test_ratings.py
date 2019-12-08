import unittest
import pandas as pd

import ratingbase
import leastsquares

class ToyLeague(unittest.TestCase):

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
             [5, 3, 2, 'A']], columns=['GAME_ID', 'TEAM_ID', 'PTS', 'LOC'])

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

    def testEvaluatePredWins(self):
        lsq = leastsquares.LeastSquares(self.league)
        correct, count = lsq.evaluate_predicted_wins()
        self.assertEqual(correct, 3)
        self.assertEqual(count, 5)

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
             [6, 3, 2, 'A']], columns=['GAME_ID', 'TEAM_ID', 'PTS', 'LOC'])

        with self.assertRaises(ValueError):
            self.league = ratingbase.League.from_hyper_table(self.raw_df)

if __name__ == '__main__':
    unittest.main()
             
