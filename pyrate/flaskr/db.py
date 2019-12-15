import pandas as pd
import numpy as np
import sqlalchemy

from flask import current_app, g
from flask.cli import with_appcontext


def get_db():
    if 'db' not in g:
        print('creating')
        g.db = sqlalchemy.create_engine(current_app.config['DATABASE'])
    return g.db

def close_db(e=None):
    db = g.pop('db', None)
    print('dropping')

def init_app(app):
    app.teardown_appcontext(close_db)

def test_db():
    db = get_db()
    df = pd.read_sql_table('teams', db)
    print(df.head())

def get_teams_table():
    db = get_db()
    df = pd.read_sql_table('teams', db)
    df = df[['NAME', 'rank', 'rating', 'WINS', 'LOSSES', 'SoS']]
    df.rename(columns={'NAME':'Team', 'rank': 'Rank', 'rating': 'Rating', 'WINS':'W', 'LOSSES':'L'}, inplace=True)
    df.sort_values(by='Rating', ascending=False, inplace=True)
    return df

def get_games_table(team_name):
    db = get_db()

    # Todo: store team->id mapping so don't have to look up each time
    teams = pd.read_sql_table('teams', db, index_col='NAME')
    team_id = int(teams.loc[team_name,'TEAM_ID'])
    
    df = pd.read_sql_query('SELECT games.Date, teams.name, games.PTS, games.OPP_PTS, games.LOC, games.NS FROM games INNER JOIN teams WHERE games.TEAM_ID is ? AND games.PTS IS NOT NULL AND games.OPP_ID = teams.TEAM_ID', db, params=[team_id], parse_dates=['Date'])
    df.rename(columns={'NAME':'Opponent', 'PTS':'PF', 'OPP_PTS':'PA'}, inplace=True)

    print(df.dtypes)
    print(df)
    return df
