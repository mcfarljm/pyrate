import pandas as pd
import numpy as np

class Team:
    def __init__(self, id, games):
        self.id = id
        self.games = games
        #print('setup team:', id, games)

    @classmethod
    def from_hyper_table(cls, df, team_id):
        """Set up an incomplete team definition from a hypertable

        The league data will need to be parsed to fill in opponent
        score information

        Parameters
        ----------
        df : pandas Data frame
            A data frame with league data, containing at least the
            following columns: 'GAME_ID', 'TEAM_ID', 'PTS'.
            Optional columns are 'DATE', 'LOC' (1 for home, -1
            for away, 0 for neutral).

        """
        if 'TEAM_ID' not in df:
            raise ValueError("expected 'TEAM_ID' column")
        elif 'GAME_ID' not in df:
            raise ValueError("expected 'GAME_ID' column")
        elif 'PTS' not in df:
            raise ValueError("expected 'PTS' column")
        
        games = df[df['TEAM_ID'] == team_id].copy()
        games.set_index('GAME_ID', inplace=True)
        return cls(team_id, games)

    @classmethod
    def from_games_table(cls, df, team_id):
        """Set up complete team definition from games table

        In the games table, each game is represented by one row that
        identifies both teams and their scores

        Parameters
        ----------
        df : pandas Data frame
            A data frame with league data, containing at least the
            following columns: 'TEAM_ID', 'PTS', 'OPP_ID', 'OPP_PTS'.
            Optional columns are 'DATE', 'LOC', and 'OPP_LOC' (1 for
            home, -1 for away, 0 for neutral).

        """
        if 'TEAM_ID' not in df:
            raise ValueError("expected 'TEAM_ID' column")
        elif 'PTS' not in df:
            raise ValueError("expected 'PTS' column")
        elif 'OPP_ID' not in df:
            raise ValueError("expected 'OPP_ID' column")
        elif 'OPP_PTS' not in df:
            raise ValueError("expected 'OPP_PTS' column")
        
        games_lhs = df[df['TEAM_ID'] == team_id].copy()

        games_rhs = df[df['OPP_ID'] == team_id].copy()

        # games_rhs is "backwards" and needs to be flipped
        games_rhs.rename(columns={'TEAM_ID':'OPP_ID', 'OPP_ID':'TEAM_ID',
                                  'PTS':'OPP_PTS', 'OPP_PTS':'PTS'},
                         inplace=True)
        if 'LOC' in games_rhs:
            games_rhs.rename(columns={'LOC':'OPP_LOC', 'OPP_LOC':'LOC'},
                             inplace=True)
        games = pd.concat((games_lhs, games_rhs), sort=False)

        team_ids = list(np.unique(np.concatenate((df['TEAM_ID'], df['OPP_ID']))))
        games['OPP_IDX'] = games['OPP_ID'].apply(lambda x: team_ids.index(x))
        return cls(team_id, games)    

def fill_hyper_scores(teams):
    """Fill in opponent scores for "hyper" format data

    In this data format, every game is loaded directly, but the
    opponent id and score are missing and must be filled in

    """
    team_ids = [team.id for team in teams]
    
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
                    team.games.loc[game_id,'OPP_IDX'] = team_ids.index(other_team.id)
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
