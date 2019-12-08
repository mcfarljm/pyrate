import pandas as pd
import numpy as np
import datetime

import team as teammodule

class League:
    def __init__(self, teams, max_date_train=None):
        """Create League instance
        """
        self.teams = teams
        self.team_ids = [team.id for team in teams]
        for team in self.teams:
            team.games['TRAIN'] = True
        # if max_date_train is not None:
        #     for team in self.teams:
        #         team.set_train_flag_by_date(max_date_train)
        #     print('Training on {} games'.format(sum([sum(t.games['TRAIN']) for t in self.teams]) / 2))

    @classmethod
    def from_hyper_table(cls, df, max_date_train=None):
        """Set up league from hyper table format

        Parameters
        ----------
        df : pandas Data frame
            A data frame with league data, containing at least the
            following columns: 'GAME_ID', 'TEAM_ID', 'PTS'.
            Optional columns are 'DATE', 'LOC' (1 for home, -1
            for away, 0 for neutral).
        """
        team_ids = df['TEAM_ID'].unique()
        teams = [teammodule.Team.from_hyper_table(df, id) for id in team_ids]
        teammodule.fill_hyper_scores(teams)
        return cls(teams, max_date_train=max_date_train)

    @classmethod
    def from_games_table(cls, df, max_date_train=None):
        """Set up league from games table format

        Parameters
        ----------
        df : pandas Data frame
            A data frame with league data, containing at least the
            following columns: 'TEAM_ID', 'PTS', 'OPP_ID', 'OPP_PTS'.
            Optional columns are 'DATE', 'LOC', and 'OPP_LOC' (1 for
            home, -1 for away, 0 for neutral).
        """
        team_ids = np.unique(np.concatenate((df['TEAM_ID'], df['OPP_ID'])))
        teams = [teammodule.Team.from_games_table(df, id) for id in team_ids]
        return cls(teams, max_date_train=max_date_train)    

    @classmethod
    def from_massey_hyper_csv(cls, filename):
        df = pd.read_csv(filename, names=['days','date','GAME_ID','RESULT_ID','TEAM_ID','LOC','PTS'], header=None)
        return League.from_hyper_table(df)

    @classmethod
    def from_massey_games_csv(cls, filename):
        df = pd.read_csv(filename, names=['days','date','TEAM_ID','LOC', 'PTS','OPP_ID','OPP_LOC','OPP_PTS'], header=None)
        # Ignore location for now, due to inconsistency in string vs int:
        df.drop(['LOC','OPP_LOC'], axis=1, inplace=True)
        return League.from_games_table(df)    
        

class RatingSystm:
    def __init__(self, league):
        self.league = league
        self.teams = league.teams

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
        count = count // 2
        correct = correct // 2
        if pred_win_count != count:
            print('Mismatch for predicted win count:', pred_win_count)
        return correct, count

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
