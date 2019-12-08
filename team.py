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

        The league data will need to be parsed to fill in opponent score information

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
        
