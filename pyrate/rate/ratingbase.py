import pandas as pd
import numpy as np
import sqlalchemy.types as sqlt

from .team import Team
from pyrate.db.schema import schema

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
                team.games['opponent'] = team.games['opponent_id'].map(team_names)
            self.team_dict = {t.name: t for t in teams}
        for team in self.teams:
            team.games['train'] = True
        # if max_date_train is not None:
        #     for team in self.teams:
        #         team.set_train_flag_by_date(max_date_train)
        #     print('Training on {} games'.format(sum([sum(t.games['train']) for t in self.teams]) / 2))

    @classmethod
    def from_hyper_table(cls, df, team_names=None, max_date_train=None):
        """Set up league from hyper table format

        Parameters
        ----------
        df : pandas Data frame
            A data frame with league data, containing at least the
            following columns: 'game_id', 'team_id', 'points'.
            Optional columns are 'date', 'location' (1 for home, -1
            for away, 0 for neutral).
        """
        team_ids = df['team_id'].unique()
        teams = [Team.from_hyper_table(df, id) for id in team_ids]
        return cls(teams, team_names=team_names, max_date_train=max_date_train)

    @classmethod
    def from_games_table(cls, df, team_names=None, max_date_train=None):
        """Set up league from games table format

        Parameters
        ----------
        df : pandas Data frame
            A data frame with league data, containing at least the
            following columns: 'team_id', 'points', 'opponent_id',
            'opponent_points'.  Optional columns are 'date',
            'location', and 'opponent_location' (1 for home, -1 for
            away, 0 for neutral).
        """
        team_ids = np.unique(np.concatenate((df['team_id'], df['opponent_id'])))
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
        if self.homecourt:
            print('home advantage: {:.1f}'.format(self.home_adv))

    def store_games(self):
        """Extract team-by-team game data into league-wide table

        This is done after the ratings are fit, since the fitting
        process might add attributes to the team game data
        """
        # double_games stores each game twice (one with each team in
        # the first position)
        self.double_games = pd.concat([t.games for t in self.teams], ignore_index=True)
        # Each game is represented twice, just choose 1.  (Believe
        # that this works the same in either "direction", although
        # e.g., mean(GOM) would not necssarily be the same if using a
        # score cap.)
        self.single_games = self.double_games[ self.double_games['team_id'] < self.double_games['opponent_id'] ]

    def store_ratings(self):
        """After child method is called, organize rating data into DataFrame"""
        ratings = self.ratings

        # Store rating attribute for each team
        for rating,team in zip(self.ratings, self.teams):
            team.rating = rating
        
        self.get_strength_of_schedule()
        sos_past = [t.sos_past for t in self.teams]
        sos_future = [t.sos_future for t in self.teams]
        sos_all = [t.sos_all for t in self.teams]

        # Note: although for interactive use, indexing by name is
        # convenient, currently index by id to cover case where names
        # are not provided (and maybe it provides faster lookup?)
        index = [t.id for t in self.teams]
        self.ratings = pd.DataFrame({'rating': ratings,
                                     'rank': rank_array(ratings),
                                     'strength_of_schedule_past': sos_past,
                                     'strength_of_schedule_future': sos_future,
                                     'strength_of_schedule_all': sos_all},
                                    index=index)

    def get_strength_of_schedule(self):
        """Compute strength of schedule as average of opponent rating

        For now, does not account for home court"""
        for team in self.teams:
            team.sos_past = np.mean([self.teams[idx].rating for idx in team.games['opponent_index']])
            if len(team.scheduled) > 0:
                team.sos_future = np.mean([self.teams[idx].rating for idx in team.scheduled['opponent_index']])
            else:
                # Prevent warning
                team.sos_future = np.nan
            team.sos_all = np.mean([self.teams[idx].rating for idx in np.concatenate((team.games['opponent_index'],team.scheduled['opponent_index']))])

    def display_ratings(self, n=10):
        print(self.ratings.sort_values(by='rating', ascending=False).head(n))

    def store_predictions(self):
        self.double_games['predicted_result'] = self.predict_result(self.double_games)
        self.consistency = sum(self.double_games['predicted_result']==self.double_games['result']) / float(len(self.double_games))

    def evaluate_predicted_wins(self, exclude_train=False):
        """Evaluate how many past games are predicted correctly"""
        if exclude_train:
            idx = ~ self.double_games['train']
        else:
            # Set idx to all True
            idx = self.double_games['train'].notnull()

        correct = sum( self.double_games.loc[idx,'predicted_result'] == self.double_games.loc[idx,'result'] ) // 2
        total = sum(idx) // 2

        return correct, total

    def evaluate_coverage_probability(self, exclude_train=False):
        pvals = np.array([0.5, 0.6, 0.7, 0.8, 0.9])
        counts = np.zeros(len(pvals), dtype=int)
        correct = np.zeros(len(pvals), dtype=int)

        total_count = 0 # Sanity check
        for team in self.teams:
            for game_id, game in team.games.iterrows():
                if exclude_train and game['train']:
                    continue                
                loc = 1 if game['location']=='H' else -1
                pred_prob = self.predict_win_probability(team, self.teams[game['opponent_index']], loc)
                pred_outcome = 'W' if pred_prob > 0.5 else 'L'
                if pred_prob > 0.5:
                    total_count += 1
                    # Determine interval
                    interval = np.where(pred_prob>pvals)[0][-1]
                    counts[interval] += 1
                    if pred_outcome == game['result']:
                        correct[interval] += 1

        print("Total count:", total_count)
        for i,p in enumerate(pvals):
            print('Coverage for {}: {} / {} ({:.2})'.format(p, correct[i], counts[i], float(correct[i])/counts[i]))

    def to_db(self, engine, rating_name):
        """Write to database

        Create "teams" and "games" tables.  The "games" table also
        includes scheduled games.  Each game in the games table is
        represented twice, once for each team in the team_id position

        Parameters
        ----------
        rating_name : str
            A unique name for the rating

        """

        with engine.connect() as conn:
            for s in schema.split('\n\n'):
                conn.execute(s)

            ## properties table (general info)
            today = pd.to_datetime(pd.datetime.today())
            df = pd.DataFrame({'Updated':[today]})
            df.to_sql("properties", engine, if_exists='replace', index=False)

            ### ratings table
            # Needs to be handled carefully because previous rating_id
            # needs to be used to remove associated game/team entries,
            # but we also need to update the rating if that row
            # already exists
            
            # Check whether rating exists:
            output = conn.execute('SELECT rating_id FROM ratings WHERE name=?', (rating_name,))
            result = output.fetchone()
            if result:
                rating_id = result[0]
            else:
                conn.execute('INSERT INTO ratings (name) VALUES (?);', (rating_name,))
                output = conn.execute('SELECT last_insert_rowid();')
                rating_id = output.fetchone()[0]

            # Now update rating_id with new data.  Re-using the
            # rating_id prevents the rating_id values from continuuing
            # to increase (alternatively, could just delete all
            # rating_id entries from games and teams here and then get
            # a new arbitrary rating_id before adding the data)
            n_games = sum([len(t.games) for t in self.teams]) // 2
            n_scheduled = sum([len(t.scheduled) for t in self.teams]) // 2
            conn.execute('UPDATE ratings SET home_advantage = ?, r_squared = ?, consistency=?, games_played = ?, games_scheduled = ? WHERE rating_id = ?;', (self.home_adv, self.Rsquared, self.consistency, n_games, n_scheduled, rating_id))

            ### teams table
            df = self.ratings.copy()
            df['rating_id'] = rating_id
            df['team_id'] = df.index
            df['wins'] = [t.wins for t in self.teams]
            df['losses'] = [t.losses for t in self.teams]
            df['name'] = [t.name for t in self.teams]

            # First delete previous entries for this league:
            conn.execute('DELETE FROM teams WHERE rating_id=?;', (rating_id,))

            df.to_sql("teams", engine, if_exists='append', index=False,
                      dtype = {'team_id': sqlt.Integer,
                               'rating_id': sqlt.Integer,
                               'name': sqlt.Text,
                               'rating': sqlt.Float,
                               'rank': sqlt.Integer,
                               'strength_of_schedule': sqlt.Float,
                               'wins': sqlt.Integer,
                               'losses': sqlt.Integer})

            ### games table
            df = self.double_games.loc[:,['team_id','opponent_id','points','opponent_points','location','date','normalized_score','result']]
            df['rating_id'] = rating_id

            df.rename(columns={'points':'points_for',
                               'opponent_points':'points_against'},
                      inplace=True)

            # First delete previous entries for this league:
            conn.execute('DELETE FROM games WHERE rating_id=?;', (rating_id,))

            df.to_sql("games", engine, if_exists='append', index=False,
                      dtype = {'team_id': sqlt.Integer,
                               'rating_id': sqlt.Integer,
                               'opponent_id': sqlt.Integer,
                               'points_for': sqlt.Integer,
                               'points_against': sqlt.Integer,
                               'date': sqlt.Date,
                               'normalized_score': sqlt.Float})

            # scheduled games
            df = pd.concat([t.scheduled for t in self.teams])
            df = df.loc[:,['team_id','opponent_id','location','date']]
            df['rating_id'] = rating_id

            df.to_sql("games", engine, if_exists='append', index=False,
                      dtype = {'team_id': sqlt.Integer,
                               'rating_id': sqlt.Integer,
                               'opponent_id': sqlt.Integer,
                               'date': sqlt.Date})
