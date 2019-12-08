import pandas as pd
import numpy as np
import datetime

import team

class League:
    def __init__(self, teams, max_date_train=None):
        self.teams = teams
        self.team_ids = [team.id for team in teams]
        self.fill_scores()
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
        teams = [team.Team.from_hyper_table(df, id) for id in team_ids]
        return cls(teams, max_date_train)
        
    def fill_scores(self):
        """Fill in opponent scores by matching up game id's"""
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
                        team.games.loc[game_id,'OPP_IDX'] = self.team_ids.index(other_team.id)
                        team.games.loc[game_id,'OPP_PTS'] = pts

                        if 'LOC' in team.games:
                            if team.games.loc[game_id,'LOC'] == other_team.games.loc[game_id,'LOC']:
                                print('Location mismatch:', game_id)
                        break
            if any(team.games['OPP_IDX'].isnull()):
                raise ValueError('incomplete opponent data for team {}'.format(team.id))
            team.games.loc[:,'OPP_IDX'] = team.games['OPP_IDX'].astype(int)
            team.games.loc[:,'OPP_PTS'] = team.games['OPP_PTS'].astype(int)

            # Set WL
            def get_wl(row):
                if row['PTS'] > row['OPP_PTS']:
                    return 'W'
                elif row['PTS'] < row['OPP_PTS']:
                    return 'L'
                else:
                    return 'T'
            team.games['WL'] = team.games.apply(get_wl, axis=1)
        

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
