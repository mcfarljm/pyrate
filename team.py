import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()

class Team:
    def __init__(self, id, games):
        self.id = id

        # Split up into games and schedule
        unplayed = games['PTS'].isnull()
        
        self.games = games[~unplayed].copy()
        self.scheduled = games[unplayed].copy()
        _fill_win_loss(self.games)

    @classmethod
    def from_hyper_table(cls, df, team_id):
        """Set up an incomplete team definition from a hypertable

        The league data will need to be parsed to fill in opponent
        score information

        Parameters
        ----------
        df : pandas Data frame
            A data frame with league data, containing at least the
            following columns: 'GAME_ID', 'TEAM_ID', 'PTS'.  Optional
            columns are 'DATE', 'LOC' (1 for home, -1 for away, 0 for
            neutral).  Scheduled (unplayed) games can be represented
            by using np.nan for PTS.
        """
        if 'TEAM_ID' not in df:
            raise ValueError("expected 'TEAM_ID' column")
        elif 'GAME_ID' not in df:
            raise ValueError("expected 'GAME_ID' column")
        elif 'PTS' not in df:
            raise ValueError("expected 'PTS' column")
        
        games = df[df['TEAM_ID'] == team_id].copy()
        games.set_index('GAME_ID', inplace=True)

        opp_games = df[ (df['GAME_ID'].isin(games.index)) & (df['TEAM_ID'] != team_id) ].copy()
        opp_games.rename(columns={'TEAM_ID':'OPP_ID', 'PTS':'OPP_PTS', 'LOC':'OPP_LOC'}, inplace=True )
        opp_games.set_index('GAME_ID', inplace=True)

        games = games.join(opp_games)

        team_ids = list(df['TEAM_ID'].unique())
        games['OPP_IDX'] = games['OPP_ID'].apply(lambda x: team_ids.index(x))

        # For compatibility with Massey data, treat 0 points as
        # scheduled game (could be added as a flag)
        scheduled = (games['PTS'] == 0) & (games['OPP_PTS'] == 0)
        games.loc[scheduled,'PTS'] = np.nan # Flag scheduled games        
        
        return cls(team_id, games)

    @classmethod
    def from_games_table(cls, df, team_id):
        """Set up complete team definition from games table

        In the games table, each game is represented by one row that
        identifies both teams and their scores.

        Parameters
        ----------
        df : pandas Data frame
            A data frame with league data, containing at least the
            following columns: 'TEAM_ID', 'PTS', 'OPP_ID', 'OPP_PTS'.
            Optional columns are 'DATE', 'LOC', and 'OPP_LOC' (1 for
            home, -1 for away, 0 for neutral). Scheduled (unplayed)
            games can be represented by using np.nan for PTS.
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
        _fill_win_loss(games)
        return cls(team_id, games)

    def plot(self, by='NS'):
        f, ax = plt.subplots()
        ax.plot(self.games['date'], self.games[by], 'o')

        xl = ax.get_xlim()
        yl = ax.get_ylim()
        ym = max(abs(np.array(yl)))
        ax.plot(xl, [0,0], 'k--', lw=1)
        ax.set_xlim(xl)
        ax.set_ylim(-ym,ym)

        ax.set_title(self.name.replace('_',' '))
        ax.set_xlabel('Date')
        ax.set_ylabel(by)
        

def _fill_win_loss(games):
    def get_wl(row):
        if row['PTS'] > row['OPP_PTS']:
            return 'W'
        elif row['PTS'] < row['OPP_PTS']:
            return 'L'
        else:
            return 'T'
    games['WL'] = games.apply(get_wl, axis=1)    

        
