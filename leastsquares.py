import pandas as pd
import numpy as np
import nba_py
import nba_py.team
import nba_py.game

SEASON='2017-18'

# Define internal ordering for computations:
TEAM_ABBRVS = sorted(nba_py.constants.TEAMS)
IDS = [nba_py.constants.TEAMS[t]['id'] for t in TEAM_ABBRVS]
SPURS_IDX = TEAM_ABBRVS.index('SAS')

def normcdf(x, mu=0, sigma=1):
    """Use np.erf so don't need scipy"""
    z = (x-mu)/sigma
    return (1.0 + np.erf(z / np.sqrt(2.0))) / 2.0


class Team:
    def __init__(self, id):
        self.id = id
        self.info = nba_py.constants.TEAMS[TEAM_ABBRVS[IDS.index(id)]]
        self.name = self.info['name']
        self.load_games()

    def load_games(self):
        df = nba_py.team.TeamGameLogs(self.id,SEASON).info()

        self.games = df.loc[:,['Game_ID','GAME_DATE','MATCHUP','WL','PTS']]
        self.games = self.games.set_index('Game_ID')

class League:
    def __init__(self):
        self.teams = [Team(id) for id in IDS]
        self.fill_scores()
        
    def fill_scores(self):
        """Fill in opponent scores by matching up game id's

        Much faster than getting these by pulling down all of the game ID's"""
        for team in self.teams:
            for game_id in team.games.index:
                for other_team in self.teams:
                    if team is other_team:
                        continue
                    try:
                        pts = other_team.games.loc[game_id,'PTS']
                    except KeyError:
                        pass
                    else:
                        team.games.loc[game_id,'OPP_ID'] = other_team.id
                        team.games.loc[game_id,'OPP_PTS'] = pts
                        break
            team.games['OPP_PTS'] = team.games['OPP_PTS'].astype(int)

class RatingSystm(League):
    def __init__(self):
        super().__init__()

    def evaluate_predicted_wins(self):
        """Evaluate how many past games are predicted correctly"""
        count = 0
        correct = 0
        for team in self.teams:
            for game_id, game in team.games.iterrows():
                count += 1
                opp_idx = IDS.index(game['OPP_ID'])
                loc = 1 if 'vs.' in game['MATCHUP'] else 0
                pred = 'W' if self.predict_win_probability(team, self.teams[opp_idx], loc) > 0.5 else 'L'
                if pred == game['WL']:
                    correct += 1
        count = count / 2
        correct = correct / 2
        print('Correct predictions: {} / {}'.format(correct, count))
                    
class LeastSquares(RatingSystm):
    def __init__(self):
        super().__init__()
        self.fit_ratings()

    def fit_ratings(self, SCORE_CAP=15, HOME_ADV=True):
        nteam = len(self.teams)
        neq = nteam+1 if HOME_ADV else nteam
        XX = np.zeros((neq, neq))
        ratings = np.zeros(neq)
        for i, team_id in enumerate(IDS):
            for game_id, game in self.teams[i].games.iterrows():
                opp_idx = IDS.index(game['OPP_ID'])
                XX[i,opp_idx] -= 1.0
                points = game['PTS'] - game['OPP_PTS']
                points = np.sign(points) * min(SCORE_CAP, abs(points)) # Truncates
                # Store "Game Outcome Measure":
                self.teams[i].games.loc[game_id,'GOM'] = points
                ratings[i] += points

                if HOME_ADV:
                    if 'vs.' in game['MATCHUP']: # Home game
                        self.teams[i].games.loc[game_id,'LOC'] = 1.0
                        XX[i,-1] += 1.0
                        # Home totals:
                        XX[-1,-1] += 1.0
                        ratings[-1] += points
                    else: # Away game
                        self.teams[i].games.loc[game_id,'LOC'] = -1.0
                        XX[i,-1] -= 1.0

            XX[i,i] = len(self.teams[i].games)

        # Replace last team equation to force sum(ratings)=0:
        XX[nteam-1,:nteam] = 1.0
        ratings[nteam-1] = 0.0

        ratings = np.linalg.solve(XX, ratings)

        # Add "Normalized Score" to team data.  This is essentially how
        # much was "earned" for each game.
        for team in self.teams:
            for game_id, game in team.games.iterrows():
                opp_idx = IDS.index(game['OPP_ID'])
                opp_rating = ratings[opp_idx]
                team.games.loc[game_id,'NS'] = game['GOM'] + opp_rating
                if HOME_ADV:
                    # Including home advantage in the normalized score is
                    # consistent with the rating being equal to the mean
                    # of the normalized scores.
                    team.games.loc[game_id,'NS'] -= game['LOC']*ratings[-1]

        # Estimate residual standard deviation, which can be used in
        # probability calculations
        SS = sum([sum(t.games['NS']**2) for t in self.teams]) / 2.0 # Divide by two b/c each game counted twice
        count = sum([len(t.games) for t in self.teams]) / 2
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
    lsq = LeastSquares()
    lsq.evaluate_predicted_wins()
    ratings = pd.DataFrame({'rating':lsq.ratings}, index=[t.name for t in lsq.teams])
    ratings = ratings.sort_values(by='rating', ascending=False)
    
