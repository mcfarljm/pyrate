import numpy as np
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
        neq = nteam+1 if self.homecourt else nteam
        XX = np.zeros((neq, neq))
        ratings = np.zeros(neq)
        for i, team in enumerate(self.teams):
            for game_id, game in team.games.iterrows():
                if not game['train']:
                    continue
                XX[i,game['opponent_index']] -= 1.0
                points = game['points'] - game['opponent_points']
                if self.score_cap is not None:
                    points = np.sign(points) * min(self.score_cap, abs(points)) # Truncates
                # Store "Game Outcome Measure":
                self.teams[i].games.loc[game_id,'GOM'] = points
                ratings[i] += points

                # Note: this assumes symmetry between home/away
                if self.homecourt:
                    if game['location'] == 'H': # Home game
                        XX[i,-1] += 1.0
                        # Home totals:
                        XX[-1,-1] += 1.0
                        ratings[-1] += points
                    elif game['location'] == 'A': # Away game
                        XX[i,-1] -= 1.0

            XX[i,i] = sum(self.teams[i].games['train'])
            #print('team {} games: {}'.format(i, XX[i,i]))

        if self.homecourt:
            # Copy the home counts from the last column into the last
            # row, to make the matrix symmetric
            XX[-1,:] = XX[:,-1]

        # Replace last team equation to force sum(ratings)=0:
        XX[nteam-1,:] = 0.0     # Catch the home court column
        XX[nteam-1,:nteam] = 1.0
        ratings[nteam-1] = 0.0

        ratings = np.linalg.solve(XX, ratings)

        # Add "Normalized Score" to team data.  This is essentially how
        # much was "earned" for each game.
        for iteam, team in enumerate(self.teams):
            for game_id, game in team.games.iterrows():
                if not game['train']:
                    continue
                opp_rating = ratings[game['opponent_index']]
                team.games.loc[game_id,'normalized_score'] = game['GOM'] + opp_rating
                team.games.loc[game_id,'predicted_GOM'] = ratings[iteam] - opp_rating
                if self.homecourt:
                    # Including home advantage in the normalized score is
                    # consistent with the rating being equal to the mean
                    # of the normalized scores.
                    loc = loc_map[game['location']] # numerical value
                    team.games.loc[game_id,'normalized_score'] -= loc*ratings[-1]
                    team.games.loc[game_id,'predicted_GOM'] += loc*ratings[-1]

        if self.homecourt:
            self.ratings = ratings[:-1]
            self.home_adv = ratings[-1]
        else:
            self.ratings = ratings
            self.home_adv = None
                    
        self.store_games()
        self.store_ratings()

        # Estimate R-squared and residual standard deviation, which
        # can be used in probability calculations
        all_games = self.single_games[self.single_games['train']]
        # Since the model does not include an intercept term, we
        # compute SST without subtracting the mean of the dependent
        # variable.
        SST = sum( all_games['GOM']**2 )
        residuals = all_games['GOM'] - all_games['predicted_GOM']
        SSE = sum(residuals**2)
        self.Rsquared = 1.0 - SSE/SST
        # DOF: number of observations less model parameters.  There is
        # one less parameter than teams because one rating is
        # arbitrary (sets the location).
        dof = len(all_games) - (len(self.teams) - 1)
        if self.homecourt:
            dof = dof -1
        self.sigma = np.sqrt( SSE / dof )

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

