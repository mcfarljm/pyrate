import pandas as pd

import ratingbase

loc_map = {1: 'H', -1: 'A', 0: 'N'}

def from_massey_hyper_csv(filename, teams_files):
    df = pd.read_csv(filename, names=['days','date','GAME_ID','RESULT_ID','TEAM_ID','LOC','PTS'], header=None)
    df['LOC'] = df['LOC'].map(loc_map)
    names = pd.read_csv(teams_file, index_col=0, squeeze=True, header=None, skipinitialspace=True)
    return ratingbase.League.from_hyper_table(df, team_names=names)

def from_massey_games_csv(filename, teams_file):
    df = pd.read_csv(filename, names=['days','date','TEAM_ID','LOC', 'PTS','OPP_ID','OPP_LOC','OPP_PTS'], header=None)
    df['LOC'] = df['LOC'].map(loc_map)
    df['OPP_LOC'] = df['OPP_LOC'].map(loc_map)
    names = pd.read_csv(teams_file, index_col=0, squeeze=True, header=None, skipinitialspace=True)
    return ratingbase.League.from_games_table(df, team_names=names)
        
