import pandas as pd
from sportsipy.nba.teams import Schedule as NbaSchedule
from sportsipy.nba.teams import Teams as NbaTeams
from sportsipy.nfl.teams import Schedule as NflSchedule
from sportsipy.nfl.teams import Teams as NflTeams

from . import ratingbase

# Todo: check neutral coding
loc_map = {"Home": "H", "Away": "A", "Neutral": "N"}


def nba_from_sportsipy(year):
    return _league_from_sportsipy(NbaTeams, NbaSchedule, year)


def nfl_from_sportsipy(year):
    return _league_from_sportsipy(NflTeams, NflSchedule, year)


def _league_from_sportsipy(Teams, Schedule, year):
    """Construct league from sportsipy Teams and Schedule classes"""
    teams = Teams()
    team_data = [(t.abbreviation, t.name) for t in teams]
    df_teams = pd.DataFrame(team_data, columns=["id", "name"])
    df_teams.set_index("id", inplace=True)

    team_games = [_get_team_games(Schedule, tid, year) for tid in df_teams.index]
    df_games = pd.concat(team_games, ignore_index=True)

    return ratingbase.League(df_games, df_teams=df_teams, duplicated_games=True)


def _get_team_games(Schedule, team_id, year):
    schedule = Schedule(team_id, year)
    game_data = [
        (g.opponent_abbr, g.points_scored, g.points_allowed, g.datetime, g.location)
        for g in schedule
    ]
    df = pd.DataFrame(
        game_data,
        columns=["opponent_id", "points", "opponent_points", "date", "location"],
    )
    df["location"] = df["location"].map(loc_map)
    df["team_id"] = team_id
    return df
