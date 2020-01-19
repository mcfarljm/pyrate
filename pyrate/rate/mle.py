"""Maximum likelihood ratings"""
import numpy as np
import pandas as pd
import scipy.optimize
from scipy.stats import gmean as geometric_mean

from .ratingbase import RatingSystem

def fixed_point_func(r, double_games, get_win_count, get_available_win_array, debug=False):
    """Function h(r) = r

    Parameters
    ----------
    r : array
        Length is nteam-1, so that the last rating is assumed to be 1
    """
    # Todo: vectorize or rewrite in C?
    if debug:
        print('r input:', r)
    result = np.empty(len(r))
    ratings_full = np.append(r, 1.0)
    for i, ri in enumerate(r):
        df = double_games[double_games['team_index'] == i]
        rjs = ratings_full[df['opponent_index'].values]
        # denom = sum( 1.0 / (ri + rjs) )
        denom = sum( get_available_win_array(df) / (ri + rjs) )
        if debug:
            print('avail wins:', i, get_available_win_array(df))
        # num = sum( df['result'] == 'W' )
        num = get_win_count(df)
        if debug:
            print('win count:', i, get_win_count(df))
        result[i] = num / denom
    if debug:
        print('r output:', r)
    return result

def get_points_for(games):
    return sum(games['points'])

def total_points_array(games):
    return np.array(games['points'] + games['opponent_points'])

class MaximumLikelihood(RatingSystem):
    def __init__(self, league, tol=1e-8):
        """
        Parameters
        ----------
        tol : float
            Solution tolerance for ratings
        """
        super().__init__(league)
        self.homecourt = False


        self.fit_ratings(tol)

    def _initialize_ratings(self):
        return np.ones(len(self.df_teams))

    def fit_ratings(self, tol):
        # Copy used in case of modification
        self.single_games = self.double_games[ self.double_games['team_id'] < self.double_games['opponent_id'] ].copy()

        r0 = self._initialize_ratings()
        r0 = np.delete(r0, -1)
        r = scipy.optimize.fixed_point(fixed_point_func, r0, args=[self.double_games, get_points_for, total_points_array], xtol=tol)

        r = np.append(r, 1.0)
        # Rescale to geometric mean of 1
        r /= geometric_mean(r)

        self.home_adv = None

        self.store_ratings(r)
        self.store_predictions()

    def strength_of_schedule(self, ratings):
        return geometric_mean(ratings)

    def predict_win_probability(self, games):
        """Predict win probability for each game in DataFrame"""
        ri = self.df_teams.loc[games['team_id'],'rating'].values
        p = ri / ( ri + self.df_teams.loc[games['opponent_id'],'rating'].values )
        # Convert to Series for convenience
        return pd.Series(p, index=games.index)

    def predict_result(self, games):
        p = self.predict_win_probability(games)
        return p.apply(lambda pi: 'W' if pi > 0.5 else 'L')
