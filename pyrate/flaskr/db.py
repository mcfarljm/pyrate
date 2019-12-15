import pandas as pd
import numpy as np
import sqlalchemy

from flask import current_app, g, url_for
from flask.cli import with_appcontext

from pyrate.rate.team import fill_win_loss


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

def get_leagues():
    """Return list of available leagues"""
    db = get_db()
    df = pd.read_sql_table('leagues', db)
    return df['Name'].values

def add_link(m, league):
    """Replace team name with link"""
    t = m.group(0)
    url = url_for('team_page', league=league, team=t)
    return '<a href="{url}">{team}</a>'.format(url=url, team=t)

def get_teams_table(league):
    db = get_db()
    # Todo: subset out league
    df = pd.read_sql_table('teams', db)
    df = df[['NAME', 'rank', 'rating', 'WINS', 'LOSSES', 'SoS']]
    df.rename(columns={'NAME':'Team', 'rank': 'Rank', 'rating': 'Rating', 'WINS':'W', 'LOSSES':'L'}, inplace=True)

    func = lambda m: add_link(m, league)
    df['Team'] = df['Team'].str.replace('(.+)',func)
    
    df.sort_values(by='Rating', ascending=False, inplace=True)
    return df

def get_games_table(league, team_name):
    db = get_db()

    # Todo: store team->id mapping so don't have to look up each time
    teams = pd.read_sql_table('teams', db, index_col='NAME')
    team_id = int(teams.loc[team_name,'TEAM_ID'])
    
    df = pd.read_sql_query('SELECT games.Date, games.LOC, teams.name, games.PTS, games.OPP_PTS, games.NS FROM games INNER JOIN teams WHERE games.TEAM_ID is ? AND games.PTS IS NOT NULL AND games.OPP_ID = teams.TEAM_ID', db, params=[team_id], parse_dates=['Date'])

    fill_win_loss(df)
    
    df.rename(columns={'NAME':'Opponent',
                       'LOC':'Loc',
                       'PTS':'PF',
                       'WL':'Result',
                       'OPP_PTS':'PA'}, inplace=True)

    func = lambda m: add_link(m, league)
    df['Opponent'] = df['Opponent'].str.replace('(.+)', func)
    
    return df
