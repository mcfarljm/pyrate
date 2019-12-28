import numpy as np
import scipy.linalg
import pandas as pd
import math

from .ratingbase import RatingSystem

loc_map = {'H': 1, 'A': -1, 'N': 0}

def normcdf(x, mu=0, sigma=1):
    """Use np.erf so don't need scipy"""
    z = (x-mu)/sigma
    return (1.0 + math.erf(z / np.sqrt(2.0))) / 2.0

class LeastSquares(RatingSystem):
    def __init__(self, league, score_cap=None, homecourt=False):
        super().__init__(league)
        self.score_cap = score_cap
        self.homecourt = homecourt

        self.fit_ratings()

    def fit_ratings(self):
        nteam = len(self.teams)

        # Need game outcome measure and normalized score to be stored
        # in double_games.  Compute them directly for simplicity, even
        # though half of calc's are redundant.
        points = self.double_games['points'] - self.double_games['opponent_points']
        if self.score_cap is not None:
            self.double_games['GOM'] = np.sign(points) * np.fmin(self.score_cap, np.abs(points))
        else:
            self.double_games['GOM'] = points

        self.single_games = self.double_games[ self.double_games['team_id'] < self.double_games['opponent_id'] ]

        games = self.single_games[self.single_games['train']]
        ngame = len(games)

        nvar = nteam + 1 if self.homecourt else nteam

        X = np.zeros((ngame,nvar), dtype='int32')
        X[np.arange(ngame), games['team_index']] = 1
        X[np.arange(ngame), games['opponent_index']] = -1
        # One rating is redundant, so assume rating of last team is 0
        X = np.delete(X, nteam-1, axis=1)

        if self.homecourt:
            X[:,-1] = games['location'].map(loc_map)

        ratings = scipy.linalg.lapack.dgels(X, games['GOM'])[1][:nvar-1]

        residuals = games['GOM'] - np.dot(X, ratings)

        if self.homecourt:
            self.home_adv = ratings[-1]
            ratings = np.delete(ratings, -1)
        else:
            self.home_adv = None

        ratings = np.append(ratings, 0) # Add 0 rating for last team
        ratings -= np.mean(ratings) # Renormalize
        self.ratings = ratings

        # Store normalized score:
        self.double_games['normalized_score'] = self.double_games['GOM'] + ratings[self.double_games['opponent_index']]
        if self.homecourt:
            self.double_games['normalized_score'] -= self.double_games['location'].map(loc_map) * self.home_adv

        # Estimate R-squared and residual standard deviation, which
        # can be used in probability calculations.  Since the model
        # does not include an intercept term, we compute SST without
        # subtracting the mean of the dependent variable.
        SST = sum( games['GOM']**2 )
        SSE = sum(residuals**2)
        self.Rsquared = 1.0 - SSE/SST
        # DOF: number of observations less model parameters.  There is
        # one less parameter than teams because one rating is
        # arbitrary (sets the location).
        dof = len(games) - (len(self.teams) - 1)
        if self.homecourt:
            dof = dof -1
        self.sigma = np.sqrt( SSE / dof )

        self.store_ratings()
        self.store_predictions()
        
    def predict_game_outcome_measure(self, games):
        """Predict GOM for games in DataFrame

        Parameters
        ----------
        games : DataFrame
            Contains the following columns: 'team_id', 'opponent_id',
            and 'location' (optional)

        """
        # Storing as a series here isn't strictly necessary, but it
        # retains the indexing of "games" and makes it possible to use
        # Series operations in subsquent calculations.  Note that we
        # convert to values first because otherwise we end up with the
        # indexes from self.ratings, which is not what we want.
        y = pd.Series(self.ratings.loc[games['team_id'],'rating'].values - self.ratings.loc[games['opponent_id'],'rating'].values, index=games.index)
        if self.homecourt and 'location' in games:
            y += games['location'].map(loc_map)
        return y

    def predict_result(self, games):
        y = self.predict_game_outcome_measure(games)
        return y.apply(lambda gom: 'W' if gom > 0.0 else 'L')

    def predict_win_probability(self, games):
        """Predict win probability for each game in DataFrame

        For the least squares system, this is based on normally
        distributed residuals
        """
        mu = self.predict_game_outcome_measure(games)
        # 1-normcdf(0,mu) = normcdf(mu)
        # Todo: use vectorized version of CDF calc (SciPy?)
        return mu.apply(lambda x: normcdf(x, sigma=self.sigma))

