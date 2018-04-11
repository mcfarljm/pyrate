import pandas as pd
import numpy as np
import datetime
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
        self.games['LOC'] = self.games.apply(lambda row: 'H' if 'vs.' in row['MATCHUP'] else 'A', axis=1)
        # Default to use all games for model training
        self.games['TRAIN'] = True

    def set_train_flag_by_date(self, max_date):
        """Set training flag by maximum date"""
        self.games['TRAIN'] = self.games['GAME_DATE'].apply(lambda d: datetime.datetime.strptime(d, '%b %d, %Y') <= max_date)

class League:
    def __init__(self, max_date_train=None):
        self.teams = [Team(id) for id in IDS]
        self.fill_scores()
        if max_date_train is not None:
            for team in self.teams:
                team.set_train_flag_by_date(max_date_train)
            print('Training on {} games'.format(sum([sum(t.games['TRAIN']) for t in self.teams]) / 2))
                
        
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
                        team.games.loc[game_id,'OPP_IDX'] = IDS.index(other_team.id)
                        team.games.loc[game_id,'OPP_PTS'] = pts

                        if team.games.loc[game_id,'LOC'] == other_team.games.loc[game_id,'LOC']:
                            print('Location mismatch:', game_id)
                        break
            team.games['OPP_PTS'] = team.games['OPP_PTS'].astype(int)
            team.games['OPP_IDX'] = team.games['OPP_IDX'].astype(int)

class RatingSystm(League):
    def __init__(self, max_date_train=None):
        super().__init__(max_date_train)

    def evaluate_predicted_wins(self, exclude_train=False):
        """Evaluate how many past games are predicted correctly"""
        count = 0
        correct = 0
        pred_win_count = 0
        for team in self.teams:
            for game_id, game in team.games.iterrows():
                if exclude_train and game['TRAIN']:
                    continue
                count += 1
                loc = 1 if game['LOC']=='H' else -1
                pred = 'W' if self.predict_win_probability(team, self.teams[game['OPP_IDX']], loc) > 0.5 else 'L'
                if pred == 'W':
                    pred_win_count += 1
                if pred == game['WL']:
                    correct += 1
        count = count / 2
        correct = correct / 2
        print('Correct predictions: {} / {}'.format(correct, count))
        if pred_win_count != count:
            print('Mismatch for predicted win count:', pred_win_count)

    def evaluate_coverage_probability(self, exclude_train=False):
        pvals = np.array([0.5, 0.6, 0.7, 0.8, 0.9])
        counts = np.zeros(len(pvals), dtype=int)
        correct = np.zeros(len(pvals), dtype=int)

        total_count = 0 # Sanity check
        for team in self.teams:
            for game_id, game in team.games.iterrows():
                if exclude_train and game['TRAIN']:
                    continue                
                loc = 1 if game['LOC']=='H' else -1
                pred_prob = self.predict_win_probability(team, self.teams[game['OPP_IDX']], loc)
                pred_outcome = 'W' if pred_prob > 0.5 else 'L'
                if pred_prob > 0.5:
                    total_count += 1
                    # Determine interval
                    interval = np.where(pred_prob>pvals)[0][-1]
                    counts[interval] += 1
                    if pred_outcome == game['WL']:
                        correct[interval] += 1

        print("Total count:", total_count)
        for i,p in enumerate(pvals):
            print('Coverage for {}: {} / {} ({:.2})'.format(p, correct[i], counts[i], float(correct[i])/counts[i]))

    def scoreboard_predict(self, offset=0):
        board = nba_py.Scoreboard(offset=offset).game_header()
        for idx, game in board.iterrows():
            home_team = self.teams[IDS.index(str(game['HOME_TEAM_ID']))]
            vis_team = self.teams[IDS.index(str(game['VISITOR_TEAM_ID']))]
            home_percent = self.predict_win_probability(home_team, vis_team, 1.0) * 100.0
            print('({:.0f}%) {:15} @ ({:.0f}%) {:15}'.format(100.0-home_percent, vis_team.name, home_percent, home_team.name))
