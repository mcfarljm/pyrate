"""Example code to update ratings database using score and schedule data from www.masseyratings.com

This example fits least squares ratings models to data for college and
professional basketball and football.  The resulting ratings are
stored to the specified SQLite database, which can be used to drive a
website.

For a deployed website, a script like this could be scheduled to run
periodically, directing the database file to the pyrate/flaskr/static
directory used by the flask website.

Usage: pyrate_update.py [pyrate.db]

Specify the name (with path) of the database for storing the results.

"""

import argparse
import os

import sqlalchemy

from pyrate.rate import massey_data
from pyrate.rate.gom import CappedPointDifference, PointDifference
from pyrate.rate.leastsquares import LeastSquares, LeastSquaresError

parser = argparse.ArgumentParser(prog='pyrate-update')
parser.add_argument('db', nargs='?', default='pyrate.db', help='database to write to (default %(default)s)')
args = parser.parse_args()


db_path = os.path.abspath(args.db)
engine = sqlalchemy.create_engine(f'sqlite:///{db_path}')

def get_rating(name, massey_url, finished, score_cap=None): 
    """
    finished : bool
       Flag for whether the season is finished
    """
    if score_cap is not None:
        gom = CappedPointDifference(score_cap)
    else:
        gom = PointDifference()

    league = url.get_league_data()
    try:
        lsq = LeastSquares(league, homecourt=True, game_outcome_measure=gom)
    except LeastSquaresError as err:
        print(f'Least squares error for {name}: {err}')
        league.summarize()
    else:
        lsq.summarize()
        lsq.display_ratings(5)
        lsq.to_db(engine, name, finished)


url = massey_data.MasseyURL('nba2020')
get_rating('NBA 2020', url, False)

url = massey_data.MasseyURL('nfl2019')
get_rating('NFL 2019', url, True)

url = massey_data.MasseyURL('cb2020', ncaa_d1=True)
get_rating('College Basketball 2020', url, False)

url = massey_data.MasseyURL('cf2019', ncaa_d1=True)
get_rating('College Football 2019', url, True)






