import numpy as np

import ratingbase as rb

def normcdf(x, mu=0, sigma=1):
    """Use np.erf so don't need scipy"""
    z = (x-mu)/sigma
    return (1.0 + np.erf(z / np.sqrt(2.0))) / 2.0

class LeastSquares(rb.RatingSystm):
    def __init__(self, max_date_train=None):
        super().__init__(max_date_train)
        self.fit_ratings()

    def fit_ratings(self, SCORE_CAP=15, HOME_ADV=True):
        nteam = len(self.teams)
        neq = nteam+1 if HOME_ADV else nteam
        XX = np.zeros((neq, neq))
        ratings = np.zeros(neq)
        for i, team in enumerate(self.teams):
            for game_id, game in team.games.iterrows():
                if not game['TRAIN']:
                    continue
                XX[i,game['OPP_IDX']] -= 1.0
                points = game['PTS'] - game['OPP_PTS']
                points = np.sign(points) * min(SCORE_CAP, abs(points)) # Truncates
                # Store "Game Outcome Measure":
                self.teams[i].games.loc[game_id,'GOM'] = points
                ratings[i] += points

                if HOME_ADV:
                    if game['LOC'] == 'H': # Home game
                        XX[i,-1] += 1.0
                        # Home totals:
                        XX[-1,-1] += 1.0
                        ratings[-1] += points
                    else: # Away game
                        XX[i,-1] -= 1.0

            XX[i,i] = sum(self.teams[i].games['TRAIN'])
            #print('team {} games: {}'.format(i, XX[i,i]))

        # Replace last team equation to force sum(ratings)=0:
        XX[nteam-1,:nteam] = 1.0
        ratings[nteam-1] = 0.0

        ratings = np.linalg.solve(XX, ratings)

        # Add "Normalized Score" to team data.  This is essentially how
        # much was "earned" for each game.
        for team in self.teams:
            for game_id, game in team.games.iterrows():
                if not game['TRAIN']:
                    continue
                opp_rating = ratings[game['OPP_IDX']]
                team.games.loc[game_id,'NS'] = game['GOM'] + opp_rating
                if HOME_ADV:
                    # Including home advantage in the normalized score is
                    # consistent with the rating being equal to the mean
                    # of the normalized scores.
                    loc = 1 if game['LOC']=='H' else -1
                    team.games.loc[game_id,'NS'] -= loc*ratings[-1]

        # Estimate residual standard deviation, which can be used in
        # probability calculations
        SS = sum([sum(t.games.loc[t.games['TRAIN'],'NS']**2) for t in self.teams]) / 2.0 # Divide by two b/c each game counted twice
        count = sum([sum(t.games['TRAIN']) for t in self.teams]) / 2
        self.sigma = np.sqrt(SS/count)

        if HOME_ADV:
            self.ratings = ratings[:-1]
            self.home_adv = ratings[-1]
        else:
            self.ratings = ratings

        # Store rating attribute for each team
        for rating,team in zip(self.ratings, self.teams):
            team.rating = rating

    def predict_win_probability(self, team1, team2, loc=None):
        """Predict win probability for team1 over team2

        loc: optionally, 1 for home, -1 for away

        For the least squares system, this is based on normally distributed residuals"""
        mu = team1.rating - team2.rating
        if loc is not None:
            try:
                mu += loc * self.home_adv
            except AttributeError:
                pass # ratings do not include home_adv
        # 1-normcdf(0,mu) = normcdf(mu)
        return normcdf(mu, sigma=self.sigma)
        

if __name__ == '__main__':
    import pandas as pd
    import datetime

    lsq = LeastSquares(datetime.datetime(2018,1,1))
    lsq.evaluate_predicted_wins(True)
    ratings = pd.DataFrame({'rating':lsq.ratings}, index=[t.name for t in lsq.teams])
    ratings = ratings.sort_values(by='rating', ascending=False)
    
