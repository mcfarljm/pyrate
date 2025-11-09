"""Utilities for retrieving score and schedule data from www.masseyratings.com"""

import numpy as np
import pandas as pd

from . import ratingbase

loc_map = {1: "H", -1: "A", 0: "N"}


class MasseyURL:
    """URL builder for Massey web data"""

    base_url = "https://www.masseyratings.com/scores.php?s={league}{sub}&all=1&mode={mode}{scheduled}&format={format}"
    # Format: 0="text", 1="matlab games", 2="matlab teams", 3="matlab hyper"

    def __init__(self, league, mode=3, scheduled=True, ncaa_d1=False):
        """
        Parameters
        ----------
        league : str
            For example, 'nba2020', 'cb2020', cf2019'
        mode : int
            1=inter, 2=intra, 3=all
        scheduled : bool
            whether to include scheduled games
        ncaa_d1 : bool
            whether to limit to NCAA D1 teams
        """
        self.league = league
        self.mode = mode
        if scheduled:
            self.scheduled = "&sch=on"
        else:
            self.scheduled = ""
        if ncaa_d1:
            self.sub = "&sub=11590"
        elif "mlb" in league.lower():
            self.sub = "&sub=14342"
        else:
            self.sub = ""

    def get_league_data(self):
        """Retrieve data from URL and process into League class"""
        return _league_from_massey_games_csv(self.games_url(), self.teams_url())

    def teams_url(self):
        return self.base_url.format(
            league=self.league,
            mode=self.mode,
            scheduled=self.scheduled,
            format=2,
            sub=self.sub,
        )

    def games_url(self):
        return self.base_url.format(
            league=self.league,
            mode=self.mode,
            scheduled=self.scheduled,
            format=1,
            sub=self.sub,
        )


def _league_from_massey_hyper_csv(games_file, teams_file):
    """Construct a League instance from game and team data

    Parameters
    ----------
    games_file : file like
    teams_file : file like
    """
    df = pd.read_csv(
        games_file,
        names=["days", "date", "game_id", "result_id", "team_id", "location", "points"],
        header=None,
    )
    df["location"] = df["location"].map(loc_map)
    df["date"] = pd.to_datetime(df["date"].astype(str))
    df.drop(columns="days", inplace=True)
    df_teams = pd.read_csv(
        teams_file, index_col=0, header=None, names=["name"], skipinitialspace=True
    )
    return ratingbase.League.from_hyper_table(df, df_teams=df_teams)


def _league_from_massey_games_csv(games_file, teams_file):
    df = pd.read_csv(
        games_file,
        names=[
            "days",
            "date",
            "team_id",
            "location",
            "points",
            "opponent_id",
            "opponent_location",
            "opponent_points",
        ],
        header=None,
    )
    df["location"] = df["location"].map(loc_map)
    df["opponent_location"] = df["opponent_location"].map(loc_map)
    df["date"] = pd.to_datetime(df["date"].astype(str))
    df.drop(columns="days", inplace=True)
    scheduled = (df["points"] == 0) & (df["opponent_points"] == 0)
    df.loc[scheduled, ["points", "opponent_points"]] = np.nan  # Flag scheduled games
    df_teams = pd.read_csv(
        teams_file, index_col=0, header=None, names=["name"], skipinitialspace=True
    )
    return ratingbase.League(df, df_teams=df_teams, duplicated_games=False)
