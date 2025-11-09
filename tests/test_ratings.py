import unittest

import numpy as np
import pandas as pd

from pyrate.rate import gom, leastsquares, mle, ratingbase


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
        for rating, expected_rating in zip(lsq.df_teams['rating'], expected_ratings):
            self.assertAlmostEqual(rating, expected_rating)

    def testScoreCap(self):
        expected_ratings = [0.875, -0.25, -0.625, 0.0]
        lsq = leastsquares.LeastSquares(self.league, game_outcome_measure=gom.CappedPointDifference(1))
        for rating, expected_rating in zip(lsq.df_teams['rating'], expected_ratings):
            self.assertAlmostEqual(rating, expected_rating)

    def testHomeCourt(self):
        expected_ratings = [2.375, -2.15277778, -1.125, 0.90277778]
        lsq = leastsquares.LeastSquares(self.league, homecourt=True)
        for rating, expected_rating in zip(lsq.df_teams['rating'], expected_ratings):
            self.assertAlmostEqual(rating, expected_rating)
        self.assertAlmostEqual(lsq.home_adv, 1.38888888)

    def testStrengtOfSchedule(self):
        expected_sos_vals = [-0.625, 0.83333333, -0.625, -0.416666666]
        lsq = leastsquares.LeastSquares(self.league)
        for sos_past, expected_sos in zip(lsq.df_teams['strength_of_schedule_past'], expected_sos_vals):
            self.assertAlmostEqual(sos_past, expected_sos)

    def testEvaluatePredWins(self):
        lsq = leastsquares.LeastSquares(self.league)
        correct, count = lsq.evaluate_predicted_wins()
        self.assertEqual(correct, 3)
        self.assertEqual(count, 5)

    def testTrainInterval(self):
        expected = {1: True, 2: False, 3: False, 4:True, 5:False}
        base = ratingbase.RatingSystem(self.league, train_interval=3)
        for i,val in expected.items():
            self.assertTrue((base.double_games.loc[i,'train'] == val).all())

    def testTestInterval(self):
        expected = {1: False, 2: True, 3: True, 4:False, 5:True}
        base = ratingbase.RatingSystem(self.league, test_interval=3)
        for i,val in expected.items():
            self.assertTrue((base.double_games.loc[i,'train'] == val).all())
    def testMLE(self):
        expected_ratings = [1.36997346, 0.65339296, 0.85526324, 1.30621178]
        sol = mle.MaximumLikelihood(self.league, method=mle.Points())
        for rating, expected_rating in zip(sol.df_teams['rating'], expected_ratings):
            self.assertAlmostEqual(rating, expected_rating)

    def testMLEStrengtOfSchedule(self):
        expected_sos_vals = [0.9238342, 1.152414, 0.9238342, 0.9148056]
        sol = mle.MaximumLikelihood(self.league, method=mle.Points())
        for sos_past, expected_sos in zip(sol.df_teams['strength_of_schedule_past'], expected_sos_vals):
            self.assertAlmostEqual(sos_past, expected_sos)

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

        self.league = ratingbase.League(self.raw_df, duplicated_games=False)

    def testLeastSquares(self):
        expected_ratings = [2.375, -2.5, -1.125, 1.25]
        lsq = leastsquares.LeastSquares(self.league)
        for rating, expected_rating in zip(lsq.df_teams['rating'], expected_ratings):
            self.assertAlmostEqual(rating, expected_rating)

    def testScoreCap(self):
        expected_ratings = [0.875, -0.25, -0.625, 0.0]
        lsq = leastsquares.LeastSquares(self.league, game_outcome_measure=gom.CappedPointDifference(1))
        for rating, expected_rating in zip(lsq.df_teams['rating'], expected_ratings):
            self.assertAlmostEqual(rating, expected_rating)

    def testHomeCourt(self):
        expected_ratings = [2.375, -2.15277778, -1.125, 0.90277778]
        lsq = leastsquares.LeastSquares(self.league, homecourt=True)
        for rating, expected_rating in zip(lsq.df_teams['rating'], expected_ratings):
            self.assertAlmostEqual(rating, expected_rating)
        self.assertAlmostEqual(lsq.home_adv, 1.38888888)

    def testStrengtOfSchedule(self):
        expected_sos_vals = [-0.625, 0.83333333, -0.625, -0.416666666]
        lsq = leastsquares.LeastSquares(self.league)
        for sos_past, expected_sos in zip(lsq.df_teams['strength_of_schedule_past'], expected_sos_vals):
            self.assertAlmostEqual(sos_past, expected_sos)

    def testEvaluatePredWins(self):
        lsq = leastsquares.LeastSquares(self.league)
        correct, count = lsq.evaluate_predicted_wins()
        self.assertEqual(correct, 3)
        self.assertEqual(count, 5)

    def testOffenseDefense(self):
        expected_offense = [8.625, 4.0625, 2.625, 6.1875]
        expected_defense = [-0.875, -1.1875, 1.625, 0.4375]

        lsq = leastsquares.LeastSquares(self.league)
        for offense, expected_offense in zip(lsq.df_teams['offense'], expected_offense):
            self.assertAlmostEqual(offense, expected_offense)
        for defense, expected_defense in zip(lsq.df_teams['defense'], expected_defense):
            self.assertAlmostEqual(defense, expected_defense)

    def testOffenseDefenseHome(self):
        expected_offense = [8.625, 4.2361111 , 2.625, 6.0138889]
        expected_defense = [-0.875, -1.0138889, 1.625, 0.2638889]

        lsq = leastsquares.LeastSquares(self.league, homecourt=True)
        for offense, expected_offense in zip(lsq.df_teams['offense'], expected_offense):
            self.assertAlmostEqual(offense, expected_offense)
        for defense, expected_defense in zip(lsq.df_teams['defense'], expected_defense):
            self.assertAlmostEqual(defense, expected_defense)

    def testWeightedLeastSquares(self):
        expected_ratings = [2.98571429, -2.5, -2.9571429, 2.4714286]

        def weight_func(games):
            weights = np.ones(len(games))
            # .values needed when called with double_games...
            weights[(games['points'] == games['opponent_points']).values] = 0.1
            return weights

        lsq = leastsquares.LeastSquares(self.league, weight_function=weight_func)
        for rating, expected_rating in zip(lsq.df_teams['rating'], expected_ratings):
            self.assertAlmostEqual(rating, expected_rating)

    def testTrainInterval(self):
        expected = {0: True, 1: False, 2: False, 3:True, 4:False}
        base = ratingbase.RatingSystem(self.league, train_interval=3)
        for i,val in expected.items():
            self.assertTrue((base.double_games.loc[i,'train'] == val).all())

    def testTestInterval(self):
        expected = {0: False, 1: True, 2: True, 3:False, 4:True}
        base = ratingbase.RatingSystem(self.league, test_interval=3)
        for i,val in expected.items():
            self.assertTrue((base.double_games.loc[i,'train'] == val).all())

class LeaveOneOutPredictions(unittest.TestCase):

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

    def leaveOneOutDriver(self, weight_func=None):

        league = ratingbase.League(self.raw_df, duplicated_games=False)
        lsq_full = leastsquares.LeastSquares(league, weight_function=weight_func)
        loo_preds = lsq_full.leave_one_out_predictions()
        loo_preds.sort_index(inplace=True)

        def get_actual_LOO_pred(index):
            df = self.raw_df.drop(index=index)
            league = ratingbase.League(df, duplicated_games=False)
            lsq = leastsquares.LeastSquares(league, weight_function=weight_func)
            pred_gom = lsq.predict_game_outcome_measure(self.raw_df.loc[[index]])
            # In case the constructor flipped the orientation of the
            # teams, reverse the prediction
            if self.raw_df.loc[index,'team_id'] != lsq_full.single_games.loc[index,'team_id']:
                pred_gom = pred_gom * -1.0
            return pred_gom.iat[0]

        actual_loo_preds = pd.Series([get_actual_LOO_pred(index) for index in self.raw_df.index], index=self.raw_df.index)
        actual_loo_preds.sort_index(inplace=True)

        for index, loo_actual in actual_loo_preds.items():
            self.assertAlmostEqual(loo_actual, loo_preds[index])

    def testLeaveOneOut(self):
        self.leaveOneOutDriver()

    def testLeaveOneOutWeighted(self):
        def weight_func(games):
            weights = np.ones(len(games))
            weights[(games['points'] == games['opponent_points']).values] = 0.1
            return weights

        self.leaveOneOutDriver(weight_func)


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

        self.league = ratingbase.League(self.raw_df, duplicated_games=False)

    def testLeastSquares(self):
        expected_ratings = [2.375, -2.5, -1.125, 1.25]
        lsq = leastsquares.LeastSquares(self.league)
        for rating, expected_rating in zip(lsq.df_teams['rating'], expected_ratings):
            self.assertAlmostEqual(rating, expected_rating)

    def testSchedule(self):
        expected_scheduled_counts = [1, 2, 1, 2]
        for team_id, expected_scheduled_count in zip(self.league.teams.index, expected_scheduled_counts):
            self.assertEqual(sum(self.league.double_schedule['team_id']==team_id), expected_scheduled_count)

class Underdetermined(unittest.TestCase):
    """Test ratings for a system with fewer games than teams."""

    def setUp(self):
        self.raw_df = pd.DataFrame(
            [[1, 5, 'N', 2, 0, 'N'],
             [3, 1, 'N', 4, 0, 'N']],
            columns=['team_id', 'points', 'location', 'opponent_id', 'opponent_points', 'opponent_location'])

        self.league = ratingbase.League(self.raw_df, duplicated_games=False)

    def testLeastSquares(self):
        # Note: the current normalization doesn't work exactly as
        # desired for this case.  The min norm solution is computed
        # with the last team (equation) removed, so it skews the
        # resulting ratings slightly vs what we would expect
        # expected_ratings = [2.5, -2.5, 0.5, -0.5]
        expected_ratings = [2.25, -2.75, 0.75, -0.25]
        lsq = leastsquares.LeastSquares(self.league)
        for rating, expected_rating in zip(lsq.df_teams['rating'], expected_ratings):
            self.assertAlmostEqual(rating, expected_rating)

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
