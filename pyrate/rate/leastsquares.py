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

        # Estimate R-squared and residual standard deviation, which
        # can be used in probability calculations
        all_games = pd.concat([t.games for t in self.teams], ignore_index=True)
        all_games = all_games[all_games['train']]
        # Each game is represented twice, just choose 1.  (Believe
        # that this works the same in either "direction", although
        # e.g., mean(GOM) would not necssarily be the same if using a
        # score cap.)
        all_games = all_games[ all_games['team_id'] < all_games['opponent_id'] ]
        SST = sum( (all_games['GOM'] - np.mean(all_games['GOM']))**2 )
        residuals = all_games['GOM'] - all_games['predicted_GOM']
        SSE = sum( (residuals - np.mean(residuals))**2 )
        self.Rsquared = 1.0 - SSE/SST
        self.sigma = np.sqrt( SSE / (len(all_games)-1) )
        
        if self.homecourt:
            self.ratings = ratings[:-1]
            self.home_adv = ratings[-1]
        else:
            self.ratings = ratings
            self.home_adv = None

        self.store_ratings()

    def predict_win_probability(self, team1, team2, loc=None):
        """Predict win probability for team1 over team2

        loc: optionally, 1 for home, -1 for away

        For the least squares system, this is based on normally distributed residuals"""
        mu = team1.rating - team2.rating
        if self.homecourt and loc is not None:
            try:
                mu += loc * self.home_adv
            except AttributeError:
                pass # ratings do not include home_adv
        # 1-normcdf(0,mu) = normcdf(mu)
        return normcdf(mu, sigma=self.sigma)

