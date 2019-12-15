import pandas as pd
import numpy as np
import sqlalchemy.types as sqlt

from .team import Team

def rank_array(a, descending=True):
    """Rank array counting from 1"""
    temp = np.argsort(a)
    if descending:
        temp = temp[::-1]
    ranks = np.empty_like(temp)
    ranks[temp] = np.arange(1,len(a)+1)
    return ranks

class League:
    def __init__(self, teams, team_names=None, max_date_train=None):
        """Create League instance

        Parameters
        ----------
        team_names: dict-like
            maps team id to name
        """
        self.teams = teams
        self.team_ids = [team.id for team in teams]
        if team_names is not None:
            for team in teams:
                team.name = team_names[team.id]
                team.games['Opponent'] = team.games['OPP_ID'].map(team_names)
            self.team_dict = {t.name: t for t in teams}
        for team in self.teams:
            team.games['TRAIN'] = True
        # if max_date_train is not None:
        #     for team in self.teams:
        #         team.set_train_flag_by_date(max_date_train)
        #     print('Training on {} games'.format(sum([sum(t.games['TRAIN']) for t in self.teams]) / 2))

    @classmethod
    def from_hyper_table(cls, df, team_names=None, max_date_train=None):
        """Set up league from hyper table format

        Parameters
        ----------
        df : pandas Data frame
            A data frame with league data, containing at least the
            following columns: 'GAME_ID', 'TEAM_ID', 'PTS'.
            Optional columns are 'DATE', 'LOC' (1 for home, -1
            for away, 0 for neutral).
        """
        team_ids = df['TEAM_ID'].unique()
        teams = [Team.from_hyper_table(df, id) for id in team_ids]
        return cls(teams, team_names=team_names, max_date_train=max_date_train)

    @classmethod
    def from_games_table(cls, df, team_names=None, max_date_train=None):
        """Set up league from games table format

        Parameters
        ----------
        df : pandas Data frame
            A data frame with league data, containing at least the
            following columns: 'TEAM_ID', 'PTS', 'OPP_ID', 'OPP_PTS'.
            Optional columns are 'DATE', 'LOC', and 'OPP_LOC' (1 for
            home, -1 for away, 0 for neutral).
        """
        team_ids = np.unique(np.concatenate((df['TEAM_ID'], df['OPP_ID'])))
        teams = [Team.from_games_table(df, id) for id in team_ids]
        return cls(teams, team_names=team_names, max_date_train=max_date_train)


class RatingSystem:
    def __init__(self, league):
        self.league = league
        self.teams = league.teams
        try:
            self.team_dict = league.team_dict
        except AttributeError:
            pass

    def summarize(self):
        print('{} played games'.format(sum([len(t.games) for t in self.teams])//2))
        num_sched = sum([len(t.scheduled) for t in self.teams])//2
        if num_sched > 0:
            print('{} scheduled games'.format(num_sched))

    def store_ratings(self):
        "After child method is called, organize rating data into DataFrame"""
        ratings = self.ratings

        # Store rating attribute for each team
        for rating,team in zip(self.ratings, self.teams):
            team.rating = rating
        
        self.get_strength_of_schedule()
        sos = [t.sos for t in self.teams]

        try:
            index = [t.name for t in self.teams]
        except AttributeError:
            index = [t.id for t in self.teams]
        self.ratings = pd.DataFrame({'rating': ratings,
                                     'rank': rank_array(ratings),
                                     'SoS': sos},
                                    index=index)

    def get_strength_of_schedule(self):
        """Compute strength of schedule as average of opponent rating

        For now, does not account for home court"""
        for team in self.teams:
            team.sos = np.mean([self.teams[idx].rating for idx in team.games['OPP_IDX']])

    def display_ratings(self, n=10):
        print(self.ratings.sort_values(by='rating', ascending=False).head(n))    

    def evaluate_predicted_wins(self, exclude_train=False):
        """Evaluate how many past games are predicted correctly"""
        count = 0
        correct = 0
        pred_win_count = 0
        for team in self.teams:
            for game_id, game in team.games.iterrows():
                if exclude_train and game['TRAIN']:
                    continue
                count += 1
                loc = 1 if game['LOC']=='H' else -1
                pred = 'W' if self.predict_win_probability(team, self.teams[game['OPP_IDX']], loc) > 0.5 else 'L'
                if pred == 'W':
                    pred_win_count += 1
                if pred == game['WL']:
                    correct += 1
        count = count // 2
        correct = correct // 2
        if pred_win_count != count:
            print('Mismatch for predicted win count:', pred_win_count)
        return correct, count

    def evaluate_coverage_probability(self, exclude_train=False):
        pvals = np.array([0.5, 0.6, 0.7, 0.8, 0.9])
        counts = np.zeros(len(pvals), dtype=int)
        correct = np.zeros(len(pvals), dtype=int)

        total_count = 0 # Sanity check
        for team in self.teams:
            for game_id, game in team.games.iterrows():
                if exclude_train and game['TRAIN']:
                    continue                
                loc = 1 if game['LOC']=='H' else -1
                pred_prob = self.predict_win_probability(team, self.teams[game['OPP_IDX']], loc)
                pred_outcome = 'W' if pred_prob > 0.5 else 'L'
                if pred_prob > 0.5:
                    total_count += 1
                    # Determine interval
                    interval = np.where(pred_prob>pvals)[0][-1]
                    counts[interval] += 1
                    if pred_outcome == game['WL']:
                        correct[interval] += 1

        print("Total count:", total_count)
        for i,p in enumerate(pvals):
            print('Coverage for {}: {} / {} ({:.2})'.format(p, correct[i], counts[i], float(correct[i])/counts[i]))

    def to_db(self, engine, league_name):
        """Write to database

        Create "teams" and "games" tables.  The "games" table also
        includes scheduled games.  Each game in the games table is
        represented twice, once for each team in the TEAM_ID
        position

        Parameters
        ----------
        league_name : str
            A unique name for the league, used to differentiate
            leagues within the database
        """

        ### leagues table
        conn = engine.connect()
        conn.execute('CREATE TABLE IF NOT EXISTS leagues ( LEAGUE_ID INTEGER PRIMARY KEY, Name TEXT UNIQUE);')

        # Check whether league exists:
        output = conn.execute('SELECT LEAGUE_ID FROM leagues WHERE Name=?', (league_name,))
        result = output.fetchone()
        if result:
            league_id = result[0]
        else:
            conn.execute('INSERT INTO leagues (Name) VALUES (?);', (league_name,))
            output = conn.execute('SELECT last_insert_rowid();')
            league_id = output.fetchone()[0]
        
        ### ratings table
        team_names = [t.name for t in self.teams]
        df = self.ratings.copy()
        df['LEAGUE_ID'] = league_id
        df['TEAM_ID'] = self.league.team_ids
        df['WINS'] = [t.wins for t in self.teams]
        df['LOSSES'] = [t.losses for t in self.teams]
        df['NAME'] = df.index

        df.set_index('TEAM_ID', inplace=True)

        # First delete previous entries for this league:
        if _table_exists(conn, 'teams'):
            conn.execute('DELETE FROM teams WHERE LEAGUE_ID=?;', (league_id,))

        df.to_sql("teams", engine, if_exists='append', index=True,
                  dtype = {'TEAM_ID': sqlt.Integer,
                           'LEAGUE_ID': sqlt.Integer,                           
                           'NAME': sqlt.Text,
                           'rating': sqlt.Float,
                           'rank': sqlt.Integer,
                           'SoS': sqlt.Float,
                           'WINS': sqlt.Integer,
                           'LOSSES': sqlt.Integer})

        ### games table
        df = pd.concat([t.games for t in self.teams])
        df = df.loc[:,['TEAM_ID','OPP_ID','PTS','OPP_PTS','LOC','Date','NS']]
        df['LEAGUE_ID'] = league_id

        # First delete previous entries for this league:
        if _table_exists(conn, 'games'):
            conn.execute('DELETE FROM games WHERE LEAGUE_ID=?;', (league_id,))

        df.to_sql("games", engine, if_exists='append', index=False,
                  dtype = {'TEAM_ID': sqlt.Integer,
                           'LEAGUE_ID': sqlt.Integer,
                           'OPP_ID': sqlt.Integer,
                           'PTS': sqlt.Integer,
                           'OPP_PTS': sqlt.Integer,
                           'Date': sqlt.Date,
                           'NS': sqlt.Float})

        # scheduled games
        df = pd.concat([t.scheduled for t in self.teams])
        df = df.loc[:,['TEAM_ID','OPP_ID','LOC','Date']]
        df['LEAGUE_ID'] = league_id

        df.to_sql("games", engine, if_exists='append', index=False,
                  dtype = {'TEAM_ID': sqlt.Integer,
                           'LEAGUE_ID': sqlt.Integer,
                           'OPP_ID': sqlt.Integer,
                           'Date': sqlt.Date})

def _table_exists(conn, table_name):
    output = conn.execute('SELECT count(*) FROM sqlite_master WHERE type=? AND name=?;', ('table',table_name,))
    return output.fetchone()[0]
    
