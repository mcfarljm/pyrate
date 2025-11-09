"""Interface with the SQLite database to extract ratings data"""

import pandas as pd
import sqlalchemy
from flask import current_app, g, request, url_for
from sqlalchemy import text

from pyrate.rate.ratingbase import rank_array


def get_db():
    if "db" not in g:
        # print('creating')
        g.db = sqlalchemy.create_engine(current_app.config["DATABASE"])
    return g.db


def close_db(e=None):
    g.pop("db", None)
    # print('dropping')


def init_app(app):
    app.teardown_appcontext(close_db)


def date_updated():
    db = get_db()
    with db.connect() as conn:
        output = conn.execute(text("SELECT Updated from properties;"))
    date = pd.to_datetime(output.fetchone()[0])
    return date


def add_rating_link(m):
    """Replace rating name with link"""
    r = m.group(0)
    url = url_for("rating_system", rating=r)
    return f'<a href="{url}">{r}</a>'


def get_most_recent_game(rating):
    """Get date of most recent game played"""
    db = get_db()

    query = """
    SELECT g.date
    FROM games g INNER JOIN ratings r on g.rating_id = r.rating_id
    WHERE r.name = :rating AND g.result IS NOT NULL
    ORDER by g.date DESC
    LIMIT 1;"""

    with db.connect() as conn:
        output = conn.execute(text(query), {"rating": rating})
        result = output.fetchone()
    return result[0]


def get_rating_system_names():
    """Get list of rating system names

    Used to populate the "Leagues" menu, so do not include those that
    are finished"""
    db = get_db()
    with db.connect() as conn:
        query = """
        SELECT name FROM ratings
        WHERE finished = 0
        ORDER BY rowid DESC;"""
        output = conn.execute(text(query))
        results = [r[0] for r in output.fetchall()]
    return results


def get_rating_systems():
    """Get table with rating systm information"""
    db = get_db()
    df = pd.read_sql_table("ratings", db)

    df["Through"] = df["name"].apply(get_most_recent_game)

    df["name"] = df["name"].str.replace("(.+)", add_rating_link, regex=True)

    df.rename(
        columns={
            "name": "League",
            "home_advantage": "Home Adv",
            "r_squared": "R<sup>2</sup>",
            "consistency": "Consist",
            "games_played": "GP",
            "games_scheduled": "GR",
        },
        inplace=True,
    )

    # Reverse sort by rating_id to show the most recent leagues first:
    df.sort_values(by="rating_id", ascending=False, inplace=True)

    df = df[
        [
            "League",
            "Through",
            "Home Adv",
            "R<sup>2</sup>",
            "Consist",
            "GP",
            "GR",
            "finished",
        ]
    ]

    return df


def add_link(m, rating):
    """Replace team name with link"""
    t = m.group(0)
    url = url_for("team_page", rating=rating, team=t)
    return f'<a href="{url}">{t}</a>'


def get_rating_table(rating):
    db = get_db()

    query = """
    SELECT t.rank, t.name, t.wins, t.losses, t.rating, t.offense_rank, t.defense_rank, t.offense, t.defense, t.strength_of_schedule_past, t.strength_of_schedule_future
    FROM teams t INNER JOIN ratings r ON t.rating_id = r.rating_id
    WHERE r.name = :rating;"""

    df = pd.read_sql_query(text(query), db, params={"rating": rating})
    df.rename(
        columns={
            "name": "Team",
            "rank": "Rank",
            "rating": "Rating",
            "wins": "W",
            "losses": "L",
            "strength_of_schedule_past": "SoS(p)",
            "strength_of_schedule_future": "SoS(f)",
            "strength_of_schedule_all": "SoS(a)",
            "offense_rank": "Off",
            "defense_rank": "Def",
        },
        inplace=True,
    )

    if request.args.get("mode") == "rating":
        # Switch in offense/defense rating columns:
        df.drop(columns=["Off", "Def"], inplace=True)
        df.rename(columns={"offense": "Off", "defense": "Def"}, inplace=True)
    else:
        # Offense/defense rankings are already retrieved, just drop
        # the corresopnding rating columns:
        df.drop(columns=["offense", "defense"], inplace=True)

        # Compute SoS ranks:
        df["SoS(p)"] = rank_array(df["SoS(p)"].values)
        # Note: originally used custom rank_array implementation that
        # calls np.argsort, which gets overridden for a series,
        # causing the custom algorithm not to work.
        #
        # There is a tradeoff with how to handle missing values.
        # Without filling, 'nan' will show up in the table, but that
        # prevents numerical sort from working.  Filling with a
        # numerical value enables sorting but is still not ideal.  For
        # example, add ".fillna(999)" to the end of the call
        df["SoS(f)"] = df["SoS(f)"].rank(ascending=False, method="min")

    if any(df["Off"].isnull()):
        df.drop(columns="Off", inplace=True)
    if any(df["Def"].isnull()):
        df.drop(columns="Def", inplace=True)

    def func(m):
        return add_link(m, rating)

    df["Team"] = df["Team"].str.replace("(.+)", func, regex=True)

    df.sort_values(by="Rating", ascending=False, inplace=True)
    return df


def get_team_id(rating, team_name):
    db = get_db()

    query = """
    SELECT t.team_id
    FROM teams t INNER JOIN ratings r ON t.rating_id = r.rating_id
    WHERE t.name = :team_name AND r.name = :rating;"""

    with db.connect() as conn:
        output = conn.execute(text(query), {"team_name": team_name, "rating": rating})

    team_id = output.fetchone()[0]
    return team_id


def get_team_data(rating, team_id):
    db = get_db()

    query = """
    SELECT t.rank, t.rating, t.wins, t.losses, t.expected_wins, t.expected_losses, t.offense_rank, t.defense_rank
    FROM teams t INNER JOIN ratings r ON t.rating_id = r.rating_id
    WHERE t.team_id = :team_id AND r.name = :rating;"""
    with db.connect() as conn:
        output = conn.execute(text(query), {"team_id": team_id, "rating": rating})
        result = output.fetchone()

    # Converting result to a dict makes it editable (and dict works
    # better than pd.Series here because it doesn't coerce all values
    # into same format)
    return result._asdict()


def get_games_table(rating, team_id):
    db = get_db()

    query = """
    SELECT g.date, g.location, t.name, t.rank, g.result, g.points_for, g.points_against, g.normalized_score
    FROM games g INNER JOIN teams t ON g.opponent_id = t.team_id
    INNER JOIN ratings r ON g.rating_id = r.rating_id
    WHERE r.name = :rating and g.team_id = :team_id AND t.rating_id = r.rating_id AND g.result IS NOT NULL;
    """

    df = pd.read_sql_query(
        text(query),
        db,
        params={"rating": rating, "team_id": team_id},
        parse_dates=["date"],
    )

    df.rename(
        columns={
            "name": "Opponent",
            "location": "Loc",
            "date": "Date",
            "rank": "OR",
            "points_for": "PF",
            "result": "Result",
            "points_against": "PA",
            "normalized_score": "NS",
        },
        inplace=True,
    )

    if any(df["NS"].isnull()):
        df.drop(columns="NS", inplace=True)

    df.sort_values(by="Date", inplace=True)

    def func(m):
        return add_link(m, rating)

    df["Opponent"] = df["Opponent"].str.replace("(.+)", func, regex=True)

    return df


def get_scheduled_games(rating, team_id):
    db = get_db()

    query = """
    SELECT g.date, g.location, t.name, t.rank, g.win_probability
    FROM games g INNER JOIN teams t ON g.opponent_id = t.team_id
    INNER JOIN ratings r ON g.rating_id = r.rating_id
    WHERE r.name = :rating and g.team_id = :team_id AND t.rating_id = r.rating_id AND g.result IS NULL;
    """

    df = pd.read_sql_query(
        text(query),
        db,
        params={"rating": rating, "team_id": team_id},
        parse_dates=["date"],
    )

    df.rename(
        columns={
            "name": "Opponent",
            "location": "Loc",
            "date": "Date",
            "rank": "OR",
            "win_probability": "Result",
        },
        inplace=True,
    )

    df["Result"] *= 100.0  # Probability to percent

    df.sort_values(by="Date", inplace=True)

    def func(m):
        return add_link(m, rating)

    df["Opponent"] = df["Opponent"].str.replace("(.+)", func, regex=True)

    return df
