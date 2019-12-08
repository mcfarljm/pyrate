import pandas as pd

import ratingbase

def from_massey_hyper_csv(filename, teams_files):
    df = pd.read_csv(filename, names=['days','date','GAME_ID','RESULT_ID','TEAM_ID','LOC','PTS'], header=None)
    names = pd.read_csv(teams_file, index_col=0, squeeze=True, header=None, skipinitialspace=True)
    return ratingbase.League.from_hyper_table(df, team_names=names)

def from_massey_games_csv(filename, teams_file):
    df = pd.read_csv(filename, names=['days','date','TEAM_ID','LOC', 'PTS','OPP_ID','OPP_LOC','OPP_PTS'], header=None)
    # Ignore location for now, due to inconsistency in string vs int:
    df.drop(['LOC','OPP_LOC'], axis=1, inplace=True)
    names = pd.read_csv(teams_file, index_col=0, squeeze=True, header=None, skipinitialspace=True)
    return ratingbase.League.from_games_table(df, team_names=names)
        
