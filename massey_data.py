import pandas as pd
import numpy as np

import ratingbase

loc_map = {1: 'H', -1: 'A', 0: 'N'}

class MasseyURL:
    """URL builder for Massey web data"""
    base_url = 'https://www.masseyratings.com/scores.php?s={league}{sub}&all=1&mode={mode}{scheduled}&format={format}'
    # Format: 0="text", 1="matlab games", 2="matlab teams", 3="matlab hyper"
    
    def __init__(self, league, mode=3, scheduled=True, ncaa_d1=False):
        """
        Parameters
        ----------
        league : str
            For example, 'nba2020', 'cb2020', cf2019'
        mode : int
            1=inter, 2=intra, 3=all
        scheduled : bool
            whether to include scheduled games
        ncaa_d1 : bool
            whether to limit to NCAA D1 teams
        """
        self.league = league
        self.mode = mode
        if scheduled:
            self.scheduled = '&sch=on'
        else:
            self.scheduled = ''
        if ncaa_d1:
            self.sub = '&sub=11590'
        else:
            self.sub = ''

    def get_league_data(self):
        """Retrieve data from URL and process into League class"""
        return league_from_massey_games_csv(self.games_url(), self.teams_url())

    def teams_url(self):
        return self.base_url.format(league=self.league, mode=self.mode, scheduled=self.scheduled, format=2, sub=self.sub)
        
    def games_url(self):
        return self.base_url.format(league=self.league, mode=self.mode, scheduled=self.scheduled, format=1, sub=self.sub)


def league_from_massey_hyper_csv(games_file, teams_file):
    df = pd.read_csv(games_file, names=['days','date','GAME_ID','RESULT_ID','TEAM_ID','LOC','PTS'], header=None)
    df['LOC'] = df['LOC'].map(loc_map)
    df['date'] = pd.to_datetime(df['date'].astype(str))
    df.drop(columns='days', inplace=True)
    names = pd.read_csv(teams_file, index_col=0, squeeze=True, header=None, skipinitialspace=True)
    return ratingbase.League.from_hyper_table(df, team_names=names)

def league_from_massey_games_csv(games_file, teams_file):
    df = pd.read_csv(games_file, names=['days','date','TEAM_ID','LOC', 'PTS','OPP_ID','OPP_LOC','OPP_PTS'], header=None)
    df['LOC'] = df['LOC'].map(loc_map)
    df['OPP_LOC'] = df['OPP_LOC'].map(loc_map)
    df['date'] = pd.to_datetime(df['date'].astype(str))
    df.drop(columns='days', inplace=True)
    names = pd.read_csv(teams_file, index_col=0, squeeze=True, header=None, skipinitialspace=True)
    scheduled = (df['PTS'] == 0) & (df['OPP_PTS'] == 0)
    df.loc[scheduled,['PTS','OPP_PTS']] = np.nan # Flag scheduled games
    return ratingbase.League.from_games_table(df, team_names=names)
        
