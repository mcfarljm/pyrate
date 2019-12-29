import numpy as np
import pandas as pd
import scipy.linalg
import scipy.stats

from .ratingbase import RatingSystem
from . import gom

loc_map = {'H': 1, 'A': -1, 'N': 0}

class LeastSquares(RatingSystem):
    def __init__(self, league, game_outcome_measure=None, homecourt=False):
        """
        Parameters
        ----------
        game_outcome_measure : callable
            Callable that accepts an array of point differences and
            returns corresponding array of game outcome measures.  May
            be an instance of a GameOutcomeMeasure subclass.
        homecourt : bool
           Whether to account for homecourt advantage
        """
        super().__init__(league)
        self.homecourt = homecourt
        if game_outcome_measure is None:
            self.gom = gom.PointDifference()
        else:
            self.gom = game_outcome_measure

        self.fit_ratings()

    def fit_ratings(self):
        # Need game outcome measure and normalized score to be stored
        # in double_games.  Compute them directly for simplicity, even
        # though half of calc's are redundant.
        points = self.double_games['points'] - self.double_games['opponent_points']
        self.double_games['GOM'] = self.gom(points)

        self.single_games = self.double_games[ self.double_games['team_id'] < self.double_games['opponent_id'] ]

        games = self.single_games[self.single_games['train']]

        X, nvar = self.get_basis_matrix(games)

        lqr,ratings,info = scipy.linalg.lapack.dgels(X, games['GOM'])
        ratings = ratings[:nvar-1]
        self.XXinv = lqr[:nvar-1,:]
        self.XXinv,info = scipy.linalg.lapack.dpotri(self.XXinv,lower=0,overwrite_c=1)
        # Copy upper to lower triangle
        i_lower = np.tril_indices(len(self.XXinv), -1)
        self.XXinv[i_lower] = self.XXinv.T[i_lower]        
        # Now have inv(XX')

        residuals = games['GOM'] - np.dot(X, ratings)

        if self.homecourt:
            self.home_adv = ratings[-1]
            ratings = np.delete(ratings, -1)
        else:
            self.home_adv = None

        ratings = np.append(ratings, 0) # Add 0 rating for last team
        ratings -= np.mean(ratings) # Renormalize

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
        dof = len(games) - (len(self.df_teams) - 1)
        if self.homecourt:
            dof = dof -1
        self.sigma = np.sqrt( SSE / dof )

        self.store_ratings(ratings)
        self.store_predictions()

    def get_basis_matrix(self, games):
        nteam = len(self.df_teams)
        ngame = len(games)
        nvar = nteam + 1 if self.homecourt else nteam

        X = np.zeros((ngame,nvar), dtype='int32')
        X[np.arange(ngame), games['team_index']] = 1
        X[np.arange(ngame), games['opponent_index']] = -1
        # One rating is redundant, so assume rating of last team is 0
        X = np.delete(X, nteam-1, axis=1)

        if self.homecourt:
            X[:,-1] = games['location'].map(loc_map)

        return X, nvar

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
        # indexes from self.df_teams, which is not what we want.
        y = pd.Series(self.df_teams.loc[games['team_id'],'rating'].values - self.df_teams.loc[games['opponent_id'],'rating'].values, index=games.index)
        if self.homecourt and 'location' in games:
            y += games['location'].map(loc_map)
        return y

    def predict_result(self, games):
        y = self.predict_game_outcome_measure(games)
        return y.apply(lambda gom: 'W' if gom > 0.0 else 'L')

    def predict_win_probability(self, games):
        """Predict win probability for each game in DataFrame

        For the least squares system, this is based on the predictive
        distribution (accounts for rating parameter uncertainty and
        residual variation in game outcome), but using a normal
        distribution instead of t distribution.

        """
        mu = self.predict_game_outcome_measure(games)

        # Get terms that account for parameter uncertainty:
        X, _ = self.get_basis_matrix(games)
        var_terms = np.array([np.dot(np.dot(x, self.XXinv), x) for x in X])
        
        sigma = np.sqrt(self.sigma**2 * (1.0 + var_terms))

        # 1-normcdf(0,mu) = normcdf(mu)
        return scipy.stats.norm.cdf(mu, scale=sigma)

