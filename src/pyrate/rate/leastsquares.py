"""Least squares rating system"""

import contextlib

import numpy as np
import pandas as pd
import scipy.linalg
import scipy.stats

with contextlib.suppress(ImportError):
    import matplotlib.pyplot as plt

from . import gom
from .ratingbase import RatingSystem

loc_map = {"H": 1, "A": -1, "N": 0}


class LeastSquaresError(Exception):
    pass


def fit_linear_least_squares(X, y, weights=None):
    """Fit linear model

    Returns
    -------
    coefs : array
    XXinv : array
        Inverse of (X*transpose(X)), or None if system is not full rank
    """
    num_coefs = np.size(X, 1)
    if weights is not None:
        rtw = np.diag(np.sqrt(weights))
        X = np.dot(rtw, X)
        y = np.dot(rtw, y)
    underdetermined = len(y) < num_coefs
    if underdetermined:
        # For underdetermined systems, we need to size y based on
        # num_coefs (probably due to how lapack overwrites "b" in
        # place).
        y = np.resize(y, num_coefs)
    # print('solving:', np.shape(X), np.shape(y))
    # print('rank:', np.linalg.matrix_rank(X))
    lqr, coefs, info = scipy.linalg.lapack.dgels(X, y)
    if info == 0 and max(np.abs(coefs)) > 10000:
        # Check for a numerically ill-conditioned solution.  An
        # alternative would be to check np.linalg.matrix_rank.  In
        # principle this should be caught by dgels, but have seen a
        # case where it is not.
        info = 1
    if info < 0:
        raise LeastSquaresError(f"error in lapack.dgels, info={info}")
    elif info == 0:
        coefs = coefs[:num_coefs]
        if underdetermined:
            XXinv = None
        else:
            XXinv = lqr[:num_coefs, :]
            XXinv, info = scipy.linalg.lapack.dpotri(XXinv, lower=0, overwrite_c=1)
            # Copy upper to lower triangle
            i_lower = np.tril_indices(len(XXinv), -1)
            XXinv[i_lower] = XXinv.T[i_lower]
            # Now have inv(XX')
    else:
        XXinv = None
        _, coefs, _, rank, _, info = scipy.linalg.lapack.dgelss(X, y)
        # print('Warning, rank deficient: rank={}, nteam={}'.format(rank, np.size(X,1)))
        coefs = coefs[:num_coefs]
        if info != 0:
            raise LeastSquaresError(f"Error in dgelss: {info}")

    return coefs, XXinv


def test_weight_func(games):
    return np.repeat(0.2, len(games))
    # weights = np.ones(len(games))
    # weights[:len(games)//2] = .5
    # return weights


def test_weight_func2(games):
    weights = np.ones(len(games))
    weights[np.abs(games["points"] - games["opponent_points"]) > 17] = 0.3
    return weights


class LeastSquares(RatingSystem):
    def __init__(
        self,
        league,
        homecourt=False,
        game_outcome_measure=None,
        weight_function=None,
        train_interval=None,
        test_interval=None,
    ):
        """
        Parameters
        ----------
        league : League
            League class instance containing the data to be used for
            the ratings
        homecourt : bool
            Whether to account for homecourt advantage
        game_outcome_measure : callable or None
            Callable that accepts an array of point differences and
            returns corresponding array of game outcome measures.  May
            be an instance of a GameOutcomeMeasure subclass.
        weight_function : callable or None
            Function that accepts a games data frame and ratings
            array, and returns an array of weights
        train_interval : int or None
            See RatingSystem
        test_interval : int or None
            See RatingSystem
        """
        super().__init__(
            league, train_interval=train_interval, test_interval=test_interval
        )
        self.homecourt = homecourt
        if game_outcome_measure is None:
            self.gom = gom.PointDifference()
        else:
            self.gom = game_outcome_measure
        self.weight_function = weight_function

        self.fit_ratings()

    def fit_ratings(self):
        """Determine the ratings using least squares

        Solves for the ratings and updates various attrributes of
        self, including Rsquared, sigma, and residuals.  Overall,
        offense, and defense ratings are passed to parent method
        store_ratings.  Parent method store_predictions is called.
        Leave-one-out predicted results are also stored.
        """
        # Need game outcome measure and normalized score to be stored
        # in double_games.  Compute them directly for simplicity, even
        # though half of calc's are redundant.
        points = self.double_games["points"] - self.double_games["opponent_points"]
        self.double_games["GOM"] = self.gom(points)
        self.double_games = self.double_games.astype({"GOM": "float64"})

        # Copy is used here so that loo_predicted_results can be added:
        self.single_games = self.double_games[
            self.double_games["team_id"] < self.double_games["opponent_id"]
        ].copy()

        games = self.single_games[self.single_games["train"]]

        X = self.get_basis_matrix(games)
        if self.weight_function:
            weights = self.weight_function(self.single_games)
        else:
            weights = None
        ratings, self.XXinv = fit_linear_least_squares(
            X, games["GOM"].values, weights=weights
        )
        self.full_rank = self.XXinv is not None

        if self.weight_function:
            # Recalculate, since weights could depend on ratings
            self.weights = self.weight_function(self.single_games)
            # And for offense/defense calcs:
            double_weights = self.weight_function(self.double_games)
        else:
            double_weights = None

        residuals = games["GOM"] - np.dot(X, ratings)

        if self.homecourt:
            self.home_adv = ratings[-1]
            ratings = np.delete(ratings, -1)
        else:
            self.home_adv = None

        ratings = np.append(ratings, 0)  # Add 0 rating for last team
        ratings -= np.mean(ratings)  # Renormalize

        if self.gom.supports_off_def:
            offense, defense = self.get_offense_defense(
                self.double_games[self.double_games["train"]], ratings, double_weights
            )
        else:
            offense, defense = None, None

        # Store normalized score:
        self.double_games["normalized_score"] = (
            self.double_games["GOM"]
            + ratings[self.double_games["opponent_index"].values]
        )
        if self.homecourt:
            self.double_games["normalized_score"] -= (
                self.double_games["location"].map(loc_map) * self.home_adv
            )

        # Estimate R-squared and residual standard deviation, which
        # can be used in probability calculations.  Since the model
        # does not include an intercept term, we compute SST without
        # subtracting the mean of the dependent variable.
        weights = self.weights if self.weight_function else 1.0
        SST = sum(games["GOM"] ** 2 * weights)
        SSE = sum(residuals**2 * weights)
        self.Rsquared = 1.0 - SSE / SST

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
            dof = V2 / V1 * (V1**2 / V2 - nparam)
        else:
            dof = len(games) - nparam

        if dof > 0:
            self.sigma = np.sqrt(SSE / dof)

        if dof > 0 and self.full_rank:
            self.leverages = np.array([np.dot(np.dot(x, self.XXinv), x) for x in X])
            if self.weight_function:
                # This is essentially like putting the weights into
                # the above calculation (i.e., rescaling x just like
                # the basis was scaled prior to getting XXinv).
                self.leverages *= self.weights
        else:
            self.leverages = np.ones(len(games))
        self.residuals = residuals
        if self.full_rank:
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
        X = np.zeros((ngame, nvar), dtype="float64")
        X[np.arange(ngame), games["team_index"]] = 1
        X[np.arange(ngame), games["opponent_index"]] = -1
        # One rating is redundant, so assume rating of last team is 0
        X = np.delete(X, nteam - 1, axis=1)

        if self.homecourt:
            X[:, -1] = games["location"].map(loc_map)

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
        X = np.zeros((ngame, nvar), dtype="float64")
        X[np.arange(ngame), double_games["team_index"]] = 1
        X[np.arange(ngame), double_games["opponent_index"]] = 1
        return X

    def get_offense_defense(self, double_games, ratings, weights):
        X = self.get_basis_matrix_defense(double_games)
        # Create rhs. The relationship GOM=PF-PA is implied but will not
        # strictly hold if GOM is not defined this way. As a workaround, define
        # "psuedo points for" as GOM+PA.
        pseudo_pf = double_games["GOM"] + double_games["opponent_points"]
        # For consistency, half of the home court advantage needs to be
        # assigned to each of the two entries for each game.
        if self.homecourt:
            pseudo_pf -= 0.5 * self.home_adv * double_games["location"].map(loc_map)
        rhs = ratings[double_games["team_index"].values] - pseudo_pf

        defense, _ = fit_linear_least_squares(X, rhs.values, weights=weights)
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
        y = pd.Series(
            self.df_teams.loc[games["team_id"], "rating"].values
            - self.df_teams.loc[games["opponent_id"], "rating"].values,
            index=games.index,
        )
        if self.homecourt and "location" in games:
            y += games["location"].map(loc_map) * self.home_adv
        return y

    def predict_result(self, games):
        y = self.predict_game_outcome_measure(games)
        return y.apply(lambda gom: "W" if gom > 0.0 else "L")

    def predict_win_probability(self, games):
        """Predict win probability for each game in DataFrame

        For the least squares system, this is based on the predictive
        distribution (accounts for rating parameter uncertainty and
        residual variation in game outcome), but using a normal
        distribution instead of t distribution.

        """
        # Gracefully handle case where sigma is not defined:
        if not self.full_rank or not hasattr(self, "sigma"):
            return np.nan

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
            # This might happen if there are not enough games played
            raise LeastSquaresError("leverage value greater than 1")
        loo_resids = self.residuals / (1.0 - self.leverages)
        loo_gom_preds = (
            self.single_games.loc[self.single_games["train"], "GOM"] - loo_resids
        )
        return loo_gom_preds

    def store_leave_one_out_predicted_results(self):
        """Compute and store predicted results based on leave-one-out models"""
        try:
            loo_gom_preds = self.leave_one_out_predictions()
        except LeastSquaresError as e:
            print(e)
        else:
            loo_results = ["W" if gom > 0.0 else "L" for gom in loo_gom_preds]

            self.single_games.loc[
                self.single_games["train"], "loo_predicted_result"
            ] = loo_results

    def strength_of_schedule(self, ratings):
        return np.mean(ratings)

    def standard_normal_residuals_plot(self):
        """Plot empirical CDF of residuals on normal probability paper"""

        def get_ecdf(x):
            xs = np.sort(x)
            n = len(x)
            fhat = np.array([(i + 1.0 - 0.5) / n for i in range(n)])
            return xs, fhat

        xs, fhat = get_ecdf(self.residuals)
        z = scipy.stats.norm.isf(1.0 - fhat)
        f, ax = plt.subplots()
        ax.plot(xs, z, "o")
        ax.set_xlabel("Residual")
        ax.set_ylabel("Standard Normal")
        ax.grid()

    def plot_residuals_vs_predictions(self):
        f, ax = plt.subplots()
        preds = self.single_games["GOM"] - self.residuals
        ax.plot(preds, self.residuals, "o")
        ax.set_xlabel("Predicted GOM")
        ax.set_ylabel("Residual")
        ax.grid()
