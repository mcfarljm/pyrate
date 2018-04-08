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

def fill_scores(teams):
    """Fill in opponent scores by matching up game id's

    Much faster than getting these by pulling down all of the game ID's"""
    for team in teams:
        for game_id in team.games.index:
            for other_team in teams:
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

def least_squares(teams, SCORE_CAP=15, HOME_ADV=True):
    nteam = len(teams)
    neq = nteam+1 if HOME_ADV else nteam
    XX = np.zeros((neq, neq))
    ratings = np.zeros(neq)
    for i, team_id in enumerate(IDS):
        for game_id, game in teams[i].games.iterrows():
            opp_idx = IDS.index(game['OPP_ID'])
            XX[i,opp_idx] -= 1.0
            points = game['PTS'] - game['OPP_PTS']
            points = np.sign(points) * min(SCORE_CAP, abs(points)) # Truncates
            # Store "Game Outcome Measure":
            teams[i].games.loc[game_id,'GOM'] = points
            ratings[i] += points

            if HOME_ADV:
                if 'vs.' in game['MATCHUP']: # Home game
                    teams[i].games.loc[game_id,'LOC'] = 1.0
                    XX[i,-1] += 1.0
                    # Home totals:
                    XX[-1,-1] += 1.0
                    ratings[-1] += points
                else: # Away game
                    teams[i].games.loc[game_id,'LOC'] = -1.0
                    XX[i,-1] -= 1.0
                    
        XX[i,i] = len(teams[i].games)

    # Replace last team equation to force sum(ratings)=0:
    XX[nteam-1,:nteam] = 1.0
    ratings[nteam-1] = 0.0

    ratings = np.linalg.solve(XX, ratings)

    # Add "Normalized Score" to team data.  This is essentially how
    # much was "earned" for each game.
    for team in teams:
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
    SS = sum([sum(t.games['NS']**2) for t in teams]) / 2.0 # Divide by two b/c each game counted twice
    count = sum([len(t.games) for t in teams]) / 2
    sigma = np.sqrt(SS/count)

    if HOME_ADV:
        return ratings[:-1], ratings[-1], sigma
    else:
        return ratings, None, sigma
        

if __name__ == '__main__':
    teams = [Team(id) for id in IDS]
    fill_scores(teams)
    rating_vals, home_adv, sigma = least_squares(teams)
    ratings = pd.DataFrame({'rating':rating_vals}, index=[t.name for t in teams])
    ratings = ratings.sort_values(by='rating', ascending=False)
    

            
        
