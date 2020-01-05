import numpy as np
import pandas as pd
import scipy.linalg
import scipy.stats

from .ratingbase import RatingSystem
from . import gom

loc_map = {'H': 1, 'A': -1, 'N': 0}

def fit_linear_least_squares(X, y, weights=None):
    """Fit linear model

    Returns
    -------
    coefs : array
    XXinv : array
        Inverse of (X*transpose(X))
    """
    num_coefs = np.size(X,1)
    if weights is None:
        X,y = X,y
    else:
        rtw = np.diag(np.sqrt(weights))
        X = np.dot(rtw, X)
        y = np.dot(rtw, y)
    lqr,coefs,info = scipy.linalg.lapack.dgels(X, y)
    coefs = coefs[:num_coefs]
    XXinv = lqr[:num_coefs,:]
    XXinv,info = scipy.linalg.lapack.dpotri(XXinv,lower=0,overwrite_c=1)

    # Copy upper to lower triangle
    i_lower = np.tril_indices(len(XXinv), -1)
    XXinv[i_lower] = XXinv.T[i_lower]
    # Now have inv(XX')
    return coefs, XXinv

def test_weight_func(games):
    return np.repeat(.2, len(games))
    # weights = np.ones(len(games))
    # weights[:len(games)//2] = .5
    # return weights

def test_weight_func2(games):
    weights = np.ones(len(games))
    weights[np.abs(games['points'] - games['opponent_points']) > 17] = 0.3
    return weights

class LeastSquares(RatingSystem):
    def __init__(self, league, homecourt=False, game_outcome_measure=None, weight_function=None):
        """
        Parameters
        ----------
        game_outcome_measure : callable
            Callable that accepts an array of point differences and
            returns corresponding array of game outcome measures.  May
            be an instance of a GameOutcomeMeasure subclass.
        homecourt : bool
            Whether to account for homecourt advantage
        weight_function : callable
            Function that accepts a games data frame and ratings
            array, and returns an array of weights
        """
        super().__init__(league)
        self.homecourt = homecourt
        if game_outcome_measure is None:
            self.gom = gom.PointDifference()
        else:
            self.gom = game_outcome_measure
        self.weight_function = weight_function

        self.fit_ratings()

    def fit_ratings(self):
        # Need game outcome measure and normalized score to be stored
        # in double_games.  Compute them directly for simplicity, even
        # though half of calc's are redundant.
        points = self.double_games['points'] - self.double_games['opponent_points']
        self.double_games['GOM'] = self.gom(points)
        self.double_games = self.double_games.astype({'GOM':'float64'})

        # Copy is used here so that loo_predicted_results can be added:
        self.single_games = self.double_games[ self.double_games['team_id'] < self.double_games['opponent_id'] ].copy()

        games = self.single_games[self.single_games['train']]

        X = self.get_basis_matrix(games)
        if self.weight_function:
            weights = self.weight_function(self.single_games)
        else:
            weights = None
        ratings, self.XXinv = fit_linear_least_squares(X, games['GOM'].values, weights=weights)

        if self.weight_function:
            # Recalculate, since weights could depend on ratings
            self.weights = self.weight_function(self.single_games)
            # And for offense/defense calcs:
            double_weights = self.weight_function(self.double_games)
        else:
            double_weights = None

        residuals = games['GOM'] - np.dot(X, ratings)

        if self.homecourt:
            self.home_adv = ratings[-1]
            ratings = np.delete(ratings, -1)
        else:
            self.home_adv = None

        ratings = np.append(ratings, 0) # Add 0 rating for last team
        ratings -= np.mean(ratings) # Renormalize

        offense, defense = self.get_offense_defense(self.double_games[self.double_games['train']], ratings, double_weights)

        # Store normalized score:
        self.double_games['normalized_score'] = self.double_games['GOM'] + ratings[self.double_games['opponent_index'].values]
        if self.homecourt:
            self.double_games['normalized_score'] -= self.double_games['location'].map(loc_map) * self.home_adv

        # Estimate R-squared and residual standard deviation, which
        # can be used in probability calculations.  Since the model
        # does not include an intercept term, we compute SST without
        # subtracting the mean of the dependent variable.
        weights = self.weights if self.weight_function else 1.0
        SST = sum( games['GOM']**2 * weights )
        SSE = sum( residuals**2 * weights )
        self.Rsquared = 1.0 - SSE/SST

        # DOF: number of observations less model parameters.  There is
        # one less parameter than teams because one rating is
        # arbitrary (sets the location).
        nparam = len(self.df_teams) - 1
        if self.homecourt:
            nparam += 1
        if self.weight_function:
            dof = len(games)
            V1 = sum(self.weights)
            V2 = sum(self.weights**2)
            # This should be further reviewed.  It is based on the
            # results here:
            # https://en.wikipedia.org/wiki/Weighted_arithmetic_mean#Reliability_weights,
            # after using an ad-hoc modification to try and extend it
            # toward multiple lost degrees of freedom.
            dof = V2/V1 *(V1**2 / V2 - nparam)
        else:
            dof = len(games) - nparam

        self.sigma = np.sqrt( SSE / dof )

        if dof > 0:
            self.leverages = np.array([np.dot(np.dot(x, self.XXinv), x) for x in X])
            if self.weight_function:
                # This is essentially like putting the weights into
                # the above calculation (i.e., rescaling x just like
                # the basis was scaled prior to getting XXinv).
                self.leverages *= self.weights
        else:
            self.leverages = np.ones(len(games))
        self.residuals = residuals
        self.store_leave_one_out_predicted_results()

        self.store_ratings(ratings, offense, defense)
        self.store_predictions()

    def get_basis_matrix(self, games):
        nteam = len(self.df_teams)
        ngame = len(games)
        nvar = nteam + 1 if self.homecourt else nteam

        # Use float dtype for consistency with GOM in least squares
        # solve, just to be safe, although doesn't seem to be strictly
        # necessary.
        X = np.zeros((ngame,nvar), dtype='float64')
        X[np.arange(ngame), games['team_index']] = 1
        X[np.arange(ngame), games['opponent_index']] = -1
        # One rating is redundant, so assume rating of last team is 0
        X = np.delete(X, nteam-1, axis=1)

        if self.homecourt:
            X[:,-1] = games['location'].map(loc_map)

        return X

    def get_basis_matrix_defense(self, double_games):
        """Basis matrix for solving off/def equations wrt defense

        Each games needs to be represented twice, once in each position"""
        nteam = len(self.df_teams)
        ngame = len(double_games)
        nvar = nteam

        # Use float dtype for consistency with GOM in least squares
        # solve, just to be safe, although doesn't seem to be strictly
        # necessary.
        X = np.zeros((ngame,nvar), dtype='float64')
        X[np.arange(ngame), double_games['team_index']] = 1
        X[np.arange(ngame), double_games['opponent_index']] = 1
        return X

    def get_offense_defense(self, double_games, ratings, weights):
        X = self.get_basis_matrix_defense(double_games)
        # Create rhs. The relationship GOM=PF-PA is implied but will not
        # strictly hold if GOM is not defined this way. As a workaround, define
        # "psuedo points for" as GOM+PA.
        pseudo_pf = double_games['GOM'] + double_games['opponent_points']
        # For consistency, half of the home court advantage needs to be
        # assigned to each of the two entries for each game.
        if self.homecourt:
            pseudo_pf -= 0.5 * self.home_adv * double_games['location'].map(loc_map)
        rhs = ratings[double_games['team_index'].values] - pseudo_pf

        defense, _ = fit_linear_least_squares(X, rhs, weights=weights)
        # At this point, defense has an implied normalization carried over from
        # ratings. We can change this and normalize average defense to 0, and
        # after applying the same normalization, offense will be interpreted as
        # expected points against average defense.
        offense = ratings - defense
        mean_def = np.mean(defense)
        defense -= mean_def
        offense -= mean_def
        return offense, defense

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
        X = self.get_basis_matrix(games)
        var_terms = np.array([np.dot(np.dot(x, self.XXinv), x) for x in X])

        sigma = np.sqrt(self.sigma**2 * (1.0 + var_terms))

        # 1-normcdf(0,mu) = normcdf(mu)
        return scipy.stats.norm.cdf(mu, scale=sigma)

    def leave_one_out_predictions(self):
        """Compute leave-one-out predictions of game outcome measure"""
        if any(self.leverages > 1.0):
            # This shouldn't happen
            raise ValueError("unexpected leverage value greater than 1")
        loo_resids = self.residuals / (1.0 - self.leverages)
        loo_gom_preds = self.single_games.loc[self.single_games['train'],'GOM'] - loo_resids
        return loo_gom_preds

    def store_leave_one_out_predicted_results(self):
        """Compute and store predicted results based on leave-one-out models"""
        loo_gom_preds = self.leave_one_out_predictions()
        loo_results = ['W' if gom > 0.0 else 'L' for gom in loo_gom_preds]

        self.single_games.loc[self.single_games['train'],'loo_predicted_result'] = loo_results
