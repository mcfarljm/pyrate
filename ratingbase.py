import pandas as pd
import numpy as np
import datetime

import team as teammodule

class League:
    def __init__(self, teams, team_names=None, max_date_train=None):
        """Create League instance

        Parameters
        ----------
        team_names: dict-like
            maps team id to name
        """
        self.teams = teams
        self.team_ids = [team.id for team in teams]
        if team_names is not None:
            for team in teams:
                team.name = team_names[team.id]
            self.team_dict = {t.name: t for t in teams}
        for team in self.teams:
            team.games['TRAIN'] = True
        # if max_date_train is not None:
        #     for team in self.teams:
        #         team.set_train_flag_by_date(max_date_train)
        #     print('Training on {} games'.format(sum([sum(t.games['TRAIN']) for t in self.teams]) / 2))

    @classmethod
    def from_hyper_table(cls, df, team_names=None, max_date_train=None):
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
        return cls(teams, team_names=team_names, max_date_train=max_date_train)

    @classmethod
    def from_games_table(cls, df, team_names=None, max_date_train=None):
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
        return cls(teams, team_names=team_names, max_date_train=max_date_train)


class RatingSystm:
    def __init__(self, league):
        self.league = league
        self.teams = league.teams
        try:
            self.team_dict = league.team_dict
        except AttributeError:
            pass

    def store_ratings(self):
        "After child method is called, organize rating data into DataFrame"""
        ratings = self.ratings
        self.get_strength_of_schedule()
        sos = [t.sos for t in self.teams]

        try:
            index = [t.name for t in self.teams]
        except AttributeError:
            index = [t.id for t in self.teams]
        self.ratings = pd.DataFrame({'rating': ratings,
                                     'SoS': sos},
                                    index=index)

    def get_strength_of_schedule(self):
        """Compute strength of schedule as average of opponent rating

        For now, does not account for home court"""
        for team in self.teams:
            team.sos = np.mean([self.teams[idx].rating for idx in team.games['OPP_IDX']])

    def display_ratings(self, n=10):
        print(self.ratings.sort_values(by='rating', ascending=False).head(n))    

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
