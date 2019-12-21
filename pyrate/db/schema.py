schema = """
CREATE TABLE IF NOT EXISTS ratings (
rating_id INTEGER PRIMARY KEY, name TEXT UNIQUE, league TEXT, year TEXT, home_advantage REAL, r_squared REAL, consistency REAL, games_played INTEGER, games_scheduled INTEGER, description TEXT );

CREATE TABLE IF NOT EXISTS teams (
rating_id INTEGER, team_id INTEGER, name TEXT, wins INTEGER, losses INTEGER, rating REAL, rank INTEGER, strength_of_schedule_past REAL, strength_of_schedule_future REAL, strength_of_schedule_all REAL );

CREATE TABLE IF NOT EXISTS games (
rating_id INTEGER, team_id INTEGER, opponent_id INTEGER, points_for INTEGER, points_against INTEGER, result TEXT, date TEXT, location TEXT, normalized_score REAL, weight REAL, win_probability REAL );
"""
