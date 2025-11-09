"""Maximum likelihood ratings"""

import numpy as np
import pandas as pd
import scipy.optimize
from scipy.stats import gmean as geometric_mean

from .ratingbase import RatingSystem


def fixed_point_func(
    logr,
    double_games,
    get_win_count,
    get_available_win_array,
    func_count=[0],
    verbosity=0,
):
    """Function h(logr) = logr

    Parameters
    ----------
    logr : array
        Log of rating vector.  Length is nteam-1, so that the last
        rating is assumed to be 1
    """
    # Todo: vectorize or rewrite in C?
    if verbosity >= 2:
        print("r input:", np.exp(logr))
    result = np.empty(len(logr))
    ratings_full = np.exp(logr)
    for i, logri in enumerate(logr):
        df = double_games[double_games["team_index"] == i]
        rjs = ratings_full[df["opponent_index"].values]
        # denom = sum( 1.0 / (ri + rjs) )
        denom = sum(get_available_win_array(df) / (np.exp(logri) + rjs))
        # if debug:
        #     print('avail wins:', i, get_available_win_array(df))
        # num = sum( df['result'] == 'W' )
        num = get_win_count(df)
        # if debug:
        #     print('win count:', i, get_win_count(df))
        result[i] = num / denom
    result = np.log(result)
    if verbosity >= 2:
        print("r output:", np.exp(result))
    func_count[0] += 1
    return result


class Wins:
    def __init__(self, win_value=1.0):
        if win_value > 1.0 or win_value < 0.0:
            raise ValueError
        self.win_value = win_value

    def win_count(self, games):
        return sum(games["result"] == "W") * self.win_value + sum(
            games["result"] == "L"
        ) * (1.0 - self.win_value)

    def game_count_per_game(self, games):
        return 1.0


class WeibullWins:
    """Construct win function from Weibull CDF curve"""

    def __init__(self, shape, scale):
        self.shape = shape
        self.scale = scale

    def cdf(self, x):
        return 1.0 - np.exp(-((x / self.scale) ** self.shape))

    def __call__(self, x):
        return 0.5 + np.sign(x) * 0.5 * self.cdf(np.abs(x))

    def win_count(self, games):
        return sum(self(np.array(games["points"] - games["opponent_points"])))

    def game_count_per_game(self, games):
        return 1.0


class Points:
    def __init__(self):
        pass

    def win_count(self, games):
        return sum(games["points"])

    def game_count_per_game(self, games):
        return np.array(games["points"] + games["opponent_points"])


class MaximumLikelihood(RatingSystem):
    def __init__(
        self,
        league,
        method=Wins(),
        tol=1e-8,
        train_interval=None,
        test_interval=None,
        verbosity=0,
    ):
        """
        Parameters
        ----------
        league : League
            League class instance containing the data to be used for
            the ratings
        method : class instance
            Class instance that implements win_count and
            game_count_per_game methods
        tol : float
            Solution tolerance for ratings
        train_interval : int or None
            See RatingSystem
        test_interval : int or None
            See RatingSystem
        verbosity : int
            Control output verbosity
        """
        super().__init__(
            league, train_interval=train_interval, test_interval=test_interval
        )
        self.method = method
        self.verbosity = verbosity

        self.homecourt = False

        self.double_games_train = self.double_games[self.double_games["train"]]
        self.fit_ratings(tol)

    def _initialize_ratings(self):
        if isinstance(self.method, Points):
            # Just start with ones:
            return np.ones(len(self.df_teams))
        else:
            # Start with "modified" win/loss ratio:
            r0 = np.ones(len(self.df_teams))
            for i in range(len(self.df_teams)):
                df = self.double_games_train[self.double_games_train["team_index"] == i]
                w = self.method.win_count(df)
                l = len(df) - w
                r0[i] = w / l
            return r0

    def fit_ratings(self, tol):
        """Fit rating system using maximum likelihood

        Parent methods store_ratings and store_predictions are called
        to store the ratings and predictions.
        """
        # Copy used in case of modification
        self.single_games = self.double_games[
            self.double_games["team_id"] < self.double_games["opponent_id"]
        ].copy()

        r0 = self._initialize_ratings()
        func_count = [0]
        # Note that if all ratings are treated as variables, there is
        # not a unique solution because there is an arbitrary scale.
        # This could be resolved by fixing the last rating to 1 (say).
        # However, tests indicate that convergence is much faster if
        # all ratings are allowed to "float".
        #
        # Have seen problems when using the default solution method,
        # especially when adjusting the value per win (this was prior
        # to floating the last rating).
        logr = scipy.optimize.fixed_point(
            fixed_point_func,
            np.log(r0),
            args=[
                self.double_games_train,
                self.method.win_count,
                self.method.game_count_per_game,
                func_count,
                self.verbosity,
            ],
            xtol=tol,
            method="iteration",
            maxiter=1000,
        )
        r = np.exp(logr)
        self.function_count = func_count[0]
        if self.verbosity >= 1:
            print("function calls:", self.function_count)

        # Rescale to geometric mean of 1
        r /= geometric_mean(r)

        self.home_adv = None

        self.store_ratings(r)
        self.store_predictions()

    def strength_of_schedule(self, ratings):
        return geometric_mean(ratings)

    def predict_win_probability(self, games):
        """Predict win probability for each game in DataFrame"""
        ri = self.df_teams.loc[games["team_id"], "rating"].values
        p = ri / (ri + self.df_teams.loc[games["opponent_id"], "rating"].values)
        # Convert to Series for convenience
        return pd.Series(p, index=games.index)

    def predict_result(self, games):
        p = self.predict_win_probability(games)
        return p.apply(lambda pi: "W" if pi > 0.5 else "L")
