"""Maximum likelihood ratings"""
import numpy as np
import pandas as pd
import scipy.optimize
from scipy.stats import gmean as geometric_mean

from .ratingbase import RatingSystem

def fixed_point_func(logr, double_games, get_win_count, get_available_win_array, debug=False):
    """Function h(logr) = logr

    Parameters
    ----------
    logr : array
        Log of rating vector.  Length is nteam-1, so that the last
        rating is assumed to be 1
    """
    # Todo: vectorize or rewrite in C?
    if debug:
        print('r input:', np.exp(logr))
    result = np.empty(len(logr))
    ratings_full = np.append(np.exp(logr), 1.0)
    for i, logri in enumerate(logr):
        df = double_games[double_games['team_index'] == i]
        rjs = ratings_full[df['opponent_index'].values]
        # denom = sum( 1.0 / (ri + rjs) )
        denom = sum( get_available_win_array(df) / (np.exp(logri) + rjs) )
        # if debug:
        #     print('avail wins:', i, get_available_win_array(df))
        # num = sum( df['result'] == 'W' )
        num = get_win_count(df)
        # if debug:
        #     print('win count:', i, get_win_count(df))
        result[i] = num / denom
    result = np.log(result)
    if debug:
        print('r output:', np.exp(result))
    return result

class Wins:
    def __init__(self, win_value=1.0):
        if win_value > 1.0 or win_value < 0.0:
            raise ValueError
        self.win_value = win_value
    def win_count(self, games):
        return sum(games['result'] == 'W') * self.win_value + sum(games['result']=='L') * (1.0 - self.win_value)
    def game_count_per_game(self, games):
        return 1.0

class Points:
    def __init__(self):
        pass
    def win_count(self, games):
        return sum(games['points'])
    def game_count_per_game(self, games):
        return np.array(games['points'] + games['opponent_points'])

class MaximumLikelihood(RatingSystem):
    def __init__(self, league, method=Wins(), tol=1e-8):
        """
        Parameters
        ----------
        method : class instance
            Class instance that implements win_count and
            game_count_per_game methods
        tol : float
            Solution tolerance for ratings
        """
        super().__init__(league)
        self.method = method

        self.homecourt = False

        self.fit_ratings(tol)

    def _initialize_ratings(self):
        # Just start with ones:
        return np.ones(len(self.df_teams))

        # Start with plain win/loss ratio:
        # return np.array(self.df_teams['wins'] / self.df_teams['losses'])

        # Start with "modified" win/loss ratio:
        # r0 = np.ones(len(self.df_teams))
        # for i in range(len(self.df_teams)):
        #     df = self.double_games[self.double_games['team_index'] == i]
        #     w = self.method.win_count(df)
        #     l = len(df) - w
        #     r0[i] = w/l
        # return r0

    def fit_ratings(self, tol):
        # Copy used in case of modification
        self.single_games = self.double_games[ self.double_games['team_id'] < self.double_games['opponent_id'] ].copy()

        r0 = self._initialize_ratings()
        r0 = np.delete(r0, -1)
        # Have seen problems when using the default solution method,
        # especially when adjusting the value per win
        logr = scipy.optimize.fixed_point(fixed_point_func, np.log(r0), args=[self.double_games, self.method.win_count, self.method.game_count_per_game], xtol=tol, method='iteration')
        r = np.exp(logr)

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
