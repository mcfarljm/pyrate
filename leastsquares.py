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

def least_squares(teams, SCORE_CAP=15):
    XX = np.zeros((len(teams), len(teams)))
    ratings = np.zeros(len(teams))
    for i, team_id in enumerate(IDS):
        for game_id, game in teams[i].games.iterrows():
            opp_idx = IDS.index(game['OPP_ID'])
            XX[i,opp_idx] -= 1.0
            points = game['PTS'] - game['OPP_PTS']
            points = np.sign(points) * min(SCORE_CAP, abs(points)) # Truncates
            # Store "Game Outcome Measure":
            teams[i].games.loc[game_id,'GOM'] = points
            ratings[i] += points
        XX[i,i] = len(teams[i].games)

    # Replace last equation to force sum(ratings)=0:
    XX[-1,:] = 1.0
    ratings[-1] = 0.0

    ratings = np.linalg.solve(XX, ratings)

    # Add "Normalized Score" to team data.  This is essentially how much was "earned" for each game
    for team in teams:
        for game_id, game in team.games.iterrows():
            opp_idx = IDS.index(game['OPP_ID'])
            opp_rating = ratings[opp_idx]
            team.games.loc[game_id,'NS'] = game['GOM'] + opp_rating
    
    return ratings
        

if __name__ == '__main__':
    teams = [Team(id) for id in IDS]
    fill_scores(teams)
    ratings = least_squares(teams)
    ratings = pd.DataFrame({'rating':ratings}, index=[t.name for t in teams])
    ratings = ratings.sort_values(by='rating', ascending=False)
    

            
        