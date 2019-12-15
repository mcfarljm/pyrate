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

def get_teams_table():
    db = get_db()
    df = pd.read_sql_table('teams', db)
    df = df[['NAME', 'rank', 'rating', 'WINS', 'LOSSES', 'SoS']]
    df.rename(columns={'NAME':'Team', 'rank': 'Rank', 'rating': 'Rating', 'WINS':'W', 'LOSSES':'L'}, inplace=True)

    # Splice in link:
    def repl(m):
        t = m.group(0)
        url = url_for('team_page', team=t)
        return '<a href="{url}">{team}</a>'.format(url=url, team=t)
    df['Team'] = df['Team'].str.replace('(.+)',repl)
    
    df.sort_values(by='Rating', ascending=False, inplace=True)
    return df

def get_games_table(team_name):
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
    return df
