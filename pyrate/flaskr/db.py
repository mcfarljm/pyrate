import pandas as pd
import numpy as np
import sqlalchemy

from flask import current_app, g, url_for
from flask.cli import with_appcontext


def get_db():
    if 'db' not in g:
        #print('creating')
        g.db = sqlalchemy.create_engine(current_app.config['DATABASE'])
    return g.db

def close_db(e=None):
    db = g.pop('db', None)
    #print('dropping')

def init_app(app):
    app.teardown_appcontext(close_db)

def date_updated():
    db = get_db()
    conn = db.connect()
    output = db.execute('SELECT Updated from properties;')
    date = pd.to_datetime(output.fetchone()[0])
    return date

def add_rating_link(m):
    """Replace rating name with link"""
    r = m.group(0)
    url = url_for('rating_system', rating=r)
    return '<a href="{url}">{rating}</a>'.format(url=url, rating=r)

def get_rating_systems():
    """Return list of rating system names"""
    db = get_db()
    df = pd.read_sql_table('ratings', db)

    df['name'] = df['name'].str.replace('(.+)', add_rating_link)

    df.rename(columns={'name':'League',
                       'home_advantage':'Home Advantage',
                       'r_squared':'R<sup>2</sup>',
                       'consistency':'Consistency',
                       'games_played':'GP',
                       'games_scheduled':'GS'},
              inplace=True)
    df = df[['League','Home Advantage','R<sup>2</sup>','Consistency','GP','GS']]
    
    return df

def add_link(m, rating):
    """Replace team name with link"""
    t = m.group(0)
    url = url_for('team_page', rating=rating, team=t)
    return '<a href="{url}">{team}</a>'.format(url=url, team=t)

def get_rating_table(rating):
    db = get_db()

    query = """
    SELECT t.rank, t.name, t.rating, t.wins, t.losses, t.strength_of_schedule_past, t.strength_of_schedule_future
    FROM teams t INNER JOIN ratings r ON t.rating_id = r.rating_id
    WHERE r.name = ?;"""

    df = pd.read_sql_query(query, db, params=[rating])
    df.rename(columns={'name':'Team',
                       'rank': 'Rank',
                       'rating': 'Rating',
                       'wins':'W',
                       'losses':'L',
                       'strength_of_schedule_past':'SoS(p)',
                       'strength_of_schedule_future':'SoS(f)',
                       'strength_of_schedule_all':'SoS(a)'},
              inplace=True)

    func = lambda m: add_link(m, rating)
    df['Team'] = df['Team'].str.replace('(.+)',func)
    
    df.sort_values(by='Rating', ascending=False, inplace=True)
    return df

def get_team_id(rating, team_name):
    db = get_db()

    conn = db.connect()
    output = db.execute("""
    SELECT t.team_id
    FROM teams t INNER JOIN ratings r ON t.rating_id = r.rating_id
    WHERE t.name = ? AND r.name = ?;""", (team_name, rating))

    team_id = output.fetchone()[0]
    return team_id

def get_team_data(rating, team_id):
    db = get_db()

    query = """
    SELECT t.rank, t.rating, t.wins, t.losses
    FROM teams t INNER JOIN ratings r ON t.rating_id = r.rating_id
    WHERE t.team_id = ? AND r.name = ?;"""
    with db.connect() as conn:
        output = conn.execute(query, (team_id, rating))
        result = output.fetchone()    

    return result

def get_games_table(rating, team_id):
    db = get_db()

    query = """
    SELECT g.date, g.location, t.name, t.rank, g.result, g.points_for, g.points_against, g.normalized_score
    FROM games g INNER JOIN teams t ON g.opponent_id = t.team_id
    INNER JOIN ratings r ON g.rating_id = r.rating_id
    WHERE r.name = ? and g.team_id = ? AND t.rating_id = r.rating_id AND g.result IS NOT NULL;
    """

    df = pd.read_sql_query(query, db, params=[rating, team_id], parse_dates=['date'])

    df.rename(columns={'name':'Opponent',
                       'location':'Loc',
                       'date':'Date',
                       'rank':'OR',
                       'points_for':'PF',
                       'result':'Result',
                       'points_against':'PA',
                       'normalized_score':'NS'},
              inplace=True)

    df.sort_values(by='Date', inplace=True)

    func = lambda m: add_link(m, rating)
    df['Opponent'] = df['Opponent'].str.replace('(.+)', func)
    
    return df

def get_scheduled_games(rating, team_id):
    db = get_db()

    query = """
    SELECT g.date, g.location, t.name, t.rank
    FROM games g INNER JOIN teams t ON g.opponent_id = t.team_id
    INNER JOIN ratings r ON g.rating_id = r.rating_id
    WHERE r.name = ? and g.team_id = ? AND t.rating_id = r.rating_id AND g.result IS NULL;
    """    

    df = pd.read_sql_query(query, db, params=[rating, team_id], parse_dates=['date'])

    df.rename(columns={'name':'Opponent',
                       'location':'Loc',
                       'date':'Date',
                       'rank':'OR'}, inplace=True)

    df.sort_values(by='Date', inplace=True)

    func = lambda m: add_link(m, rating)
    df['Opponent'] = df['Opponent'].str.replace('(.+)', func)
    
    return df
