import pandas as pd
import numpy as np
try:
    import matplotlib.pyplot as plt
    from pandas.plotting import register_matplotlib_converters
except ImportError:
    pass
else:
    register_matplotlib_converters()

class Team:
    def __init__(self, id, games):
        self.id = id

        # Split up into games and schedule
        unplayed = (games['points'].isnull() | games['opponent_points'].isnull())
        
        self.games = games[~unplayed].copy()
        self.scheduled = games[unplayed].copy()
        del games # Prevent accidental access to old version
        self.games = self.games.astype({'points':'int32', 'opponent_points':'int32'})
        self.games['score'] = self.games['points'].astype(str).str.cat(self.games['opponent_points'].astype(str),sep='-')
        fill_win_loss(self.games)
        if 'date' in self.games:
            self.games.sort_values(by='date', inplace=True)

        self.wins = sum(self.games['points'] > self.games['opponent_points'])
        self.losses = len(self.games) - self.wins

    @classmethod
    def from_hyper_table(cls, df, team_id):
        """Set up an incomplete team definition from a hypertable

        The league data will need to be parsed to fill in opponent
        score information

        Parameters
        ----------
        df : pandas Data frame
            A data frame with league data, containing at least the
            following columns: 'game_id', 'team_id', 'points'.
            Optional columns are 'date', 'location' (1 for home, -1
            for away, 0 for neutral).  Scheduled (unplayed) games can
            be represented by using np.nan for 'points'.
        """
        if 'team_id' not in df:
            raise ValueError("expected 'team_id' column")
        elif 'game_id' not in df:
            raise ValueError("expected 'game_id' column")
        elif 'points' not in df:
            raise ValueError("expected 'points' column")
        
        games = df[df['team_id'] == team_id].copy()
        games.set_index('game_id', inplace=True)

        opp_games = df[ (df['game_id'].isin(games.index)) & (df['team_id'] != team_id) ].copy()
        opp_games.rename(columns={'team_id':'opponent_id', 'points':'opponent_points', 'location':'opponent_location'}, inplace=True )
        opp_games.set_index('game_id', inplace=True)

        games = games.join(opp_games)

        team_ids = list(df['team_id'].unique())
        games['opponent_index'] = games['opponent_id'].apply(lambda x: team_ids.index(x))
        games['team_index'] = games['team_id'].apply(lambda x: team_ids.index(x))

        # For compatibility with Massey data, treat 0 points as
        # scheduled game (could be added as a flag)
        scheduled = (games['points'] == 0) & (games['opponent_points'] == 0)
        games.loc[scheduled,['points','opponent_points']] = np.nan # Flag scheduled games
        
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
            following columns: 'team_id', 'points', 'opponent_id',
            'opponent_points'.  Optional columns are 'date',
            'location', and 'opponent_location' (1 for home, -1 for
            away, 0 for neutral). Scheduled (unplayed) games can be
            represented by using np.nan for 'points'.
        """
        if 'team_id' not in df:
            raise ValueError("expected 'team_id' column")
        elif 'points' not in df:
            raise ValueError("expected 'points' column")
        elif 'opponent_id' not in df:
            raise ValueError("expected 'opponent_id' column")
        elif 'opponent_points' not in df:
            raise ValueError("expected 'opponent_points' column")
        
        games_lhs = df[df['team_id'] == team_id].copy()

        games_rhs = df[df['opponent_id'] == team_id].copy()

        # games_rhs is "backwards" and needs to be flipped
        games_rhs.rename(columns={'team_id':'opponent_id',
                                  'opponent_id':'team_id',
                                  'points':'opponent_points',
                                  'opponent_points':'points'},
                         inplace=True)
        if 'location' in games_rhs:
            games_rhs.rename(columns={'location':'opponent_location',
                                      'opponent_location':'location'},
                             inplace=True)
        games = pd.concat((games_lhs, games_rhs), sort=False)

        team_ids = list(np.unique(np.concatenate((df['team_id'], df['opponent_id']))))
        games['opponent_index'] = games['opponent_id'].apply(lambda x: team_ids.index(x))
        games['team_index'] = games['team_id'].apply(lambda x: team_ids.index(x))

        fill_win_loss(games)
        return cls(team_id, games)

    def plot(self, by='normalized_score'):
        f, ax = plt.subplots()
        ax.plot(self.games['date'], self.games[by], 'o')

        xl = ax.get_xlim()
        yl = ax.get_ylim()
        ym = max(abs(np.array(yl)))
        ax.plot(xl, [0,0], 'k--', lw=1)
        ax.set_xlim(xl)
        ax.set_ylim(-ym,ym)

        ax.set_title(self.name.replace('_',' '))
        ax.set_xlabel('date')
        ax.set_ylabel(by)
        

def fill_win_loss(games):
    def get_wl(row):
        if row['points'] > row['opponent_points']:
            return 'W'
        elif row['points'] < row['opponent_points']:
            return 'L'
        else:
            return 'T'
    games['result'] = games.apply(get_wl, axis=1)    

        
