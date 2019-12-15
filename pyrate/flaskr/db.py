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

def date_updated():
    db = get_db()
    conn = db.connect()
    output = db.execute('SELECT Updated from properties;')
    date = pd.to_datetime(output.fetchone()[0])
    return date

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

    query = """SELECT t.rank, t.NAME, t.rating, t.WINS, t.LOSSES, t.SoS FROM teams t
    WHERE t.LEAGUE_ID IN (SELECT l.LEAGUE_ID FROM leagues l WHERE l.Name = ?);"""
    df = pd.read_sql_query(query, db, params=[league])
    df.rename(columns={'NAME':'Team', 'rank': 'Rank', 'rating': 'Rating', 'WINS':'W', 'LOSSES':'L'}, inplace=True)

    func = lambda m: add_link(m, league)
    df['Team'] = df['Team'].str.replace('(.+)',func)
    
    df.sort_values(by='Rating', ascending=False, inplace=True)
    return df

def get_games_table(league, team_name):
    db = get_db()

    # Todo: avoid having to look up team_id each time?
    conn = db.connect()
    output = db.execute('SELECT t.TEAM_ID FROM teams t WHERE t.NAME = ? AND t.LEAGUE_ID IN (SELECT l.LEAGUE_ID FROM leagues l WHERE l.Name = ?);', (team_name, league))
    team_id = output.fetchone()[0]

    query = """SELECT g.Date, g.LOC, t.name, g.PTS, g.OPP_PTS, g.NS FROM games g INNER JOIN teams t 
    WHERE g.LEAGUE_ID IN (SELECT l.LEAGUE_ID FROM leagues l WHERE l.Name = ?)
    AND t.LEAGUE_ID IN (SELECT l.LEAGUE_ID FROM leagues l WHERE l.Name = ?)
    AND g.TEAM_ID is ? AND g.PTS IS NOT NULL AND g.OPP_ID = t.TEAM_ID;"""
    
    df = pd.read_sql_query(query, db, params=[league, league, team_id], parse_dates=['Date'])

    fill_win_loss(df)
    
    df.rename(columns={'NAME':'Opponent',
                       'LOC':'Loc',
                       'PTS':'PF',
                       'WL':'Result',
                       'OPP_PTS':'PA'}, inplace=True)

    func = lambda m: add_link(m, league)
    df['Opponent'] = df['Opponent'].str.replace('(.+)', func)
    
    return df

def get_scheduled_games(league, team_name):
    db = get_db()

    # Todo: avoid having to look up team_id each time?
    conn = db.connect()
    output = db.execute('SELECT t.TEAM_ID FROM teams t WHERE t.NAME = ? AND t.LEAGUE_ID IN (SELECT l.LEAGUE_ID FROM leagues l WHERE l.Name = ?);', (team_name, league))
    team_id = output.fetchone()[0]

    query = """SELECT g.Date, g.LOC, t.name FROM games g INNER JOIN teams t 
    WHERE g.LEAGUE_ID IN (SELECT l.LEAGUE_ID FROM leagues l WHERE l.Name = ?)
    AND t.LEAGUE_ID IN (SELECT l.LEAGUE_ID FROM leagues l WHERE l.Name = ?)
    AND g.TEAM_ID is ? AND g.PTS IS NULL AND g.OPP_ID = t.TEAM_ID;"""
    
    df = pd.read_sql_query(query, db, params=[league, league, team_id], parse_dates=['Date'])

    df.rename(columns={'NAME':'Opponent',
                       'LOC':'Loc'}, inplace=True)

    df.sort_values(by='Date', inplace=True)

    func = lambda m: add_link(m, league)
    df['Opponent'] = df['Opponent'].str.replace('(.+)', func)
    
    return df
