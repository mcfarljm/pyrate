"""Define League class and RatingSystem base class"""

import pandas as pd
import numpy as np
import sqlalchemy.types as sqlt
import datetime

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
    """Data structure to hold score and schedule data"""
    def __init__(self, df_games, df_teams=None, duplicated_games=True):
        """Create League instance

        Parameters
        ----------
        df_games : DataFrame
            DataFrame containing at least columns 'team_id',
            'opponent_id', 'points', and 'opponent_points'
        df_teams : DataFrame
            DataFrame containing at least an index of team id's.
        duplicated_games : bool
            Whether each game is represented twice, once for each team
        """
        if 'team_id' not in df_games:
            raise ValueError("expected 'team_id' column")
        elif 'points' not in df_games:
            raise ValueError("expected 'points' column")
        elif 'opponent_id' not in df_games:
            raise ValueError("expected 'opponent_id' column")
        elif 'opponent_points' not in df_games:
            raise ValueError("expected 'opponent_points' column")

        if not duplicated_games:
            df2 = df_games.rename(columns={'team_id':'opponent_id', 'opponent_id':'team_id',
                                           'points':'opponent_points','opponent_points':'points',
                                           'location':'opponent_location','opponent_location':'location'})
            df_games = pd.concat((df_games,df2), join='inner')

        # Note: although for interactive use, indexing by name is
        # convenient, currently index by id to cover case where names
        # are not provided (and maybe it provides faster lookup?)
        if df_teams is None:
            self.teams = pd.DataFrame(index=np.sort(df_games['team_id'].unique()))
        else:
            self.teams = df_teams

        team_ids = list(self.teams.index)
        df_games = df_games.copy()
        df_games['team_index'] = df_games['team_id'].apply(lambda x: team_ids.index(x))
        df_games['opponent_index'] = df_games['opponent_id'].apply(lambda x: team_ids.index(x))

        # Split up into games and schedule
        unplayed = (df_games['points'].isnull() | df_games['opponent_points'].isnull())
        self.double_games = df_games[~unplayed].copy()
        self.double_schedule = df_games[unplayed].copy()
        self.double_games = self.double_games.astype({'points':'int32', 'opponent_points':'int32'})

        def get_wl(row):
            if row['points'] > row['opponent_points']:
                return 'W'
            elif row['points'] < row['opponent_points']:
                return 'L'
            else:
                return 'T'
        self.double_games['result'] = self.double_games.apply(get_wl, axis=1)

        self.teams['wins'] = [sum(self.double_games.loc[self.double_games['team_id'] == tid, 'result'] == 'W') for tid in self.teams.index]
        self.teams['losses'] = [sum(self.double_games['team_id']==tid) - self.teams.at[tid,'wins'] for tid in self.teams.index]

    def summarize(self):
        print('League summary: {} teams, {} games'.format(len(self.teams), len(self.double_games)//2))


    @classmethod
    def from_hyper_table(cls, df_games, df_teams=None):
        """Set up league from hyper table format

        Parameters
        ----------
        df_games : pandas Data frame
            A data frame with league data, containing at least the
            following columns: 'game_id', 'team_id', 'points'.
            Optional columns are 'date', 'location' (1 for home, -1
            for away, 0 for neutral).
        """
        if 'team_id' not in df_games:
            raise ValueError("expected 'team_id' column")
        elif 'game_id' not in df_games:
            raise ValueError("expected 'game_id' column")
        elif 'points' not in df_games:
            raise ValueError("expected 'points' column")

        num_rows = len(df_games)
        if num_rows % 2 != 0:
            raise ValueError("expected even number of rows")

        games = df_games.set_index('game_id')
        games = games.join(games, rsuffix='2')
        # Each index was originally represented twice, so after join
        # now appears 4 times, 2 combinations of which are not valid
        games = games[games['team_id'] != games['team_id2']]
        # Now have double-games format; rename the columns
        games = games.rename(columns={'team_id2':'opponent_id',
                                      'points2':'opponent_points',
                                      'location2':'opponent_location'})
        if len(games) != num_rows:
            # If there is a mismatch, game_id's not appearing twice
            # get dropped by the join
            raise ValueError("game_id mismatch")

        # For compatibility with Massey data, treat 0 points as
        # scheduled game (could be added as a flag)
        scheduled = (games['points'] == 0) & (games['opponent_points'] == 0)
        games.loc[scheduled,['points','opponent_points']] = np.nan # Flag scheduled games

        return cls(games, df_teams)


class RatingSystem:
    """Base class for rating system"""
    def __init__(self, league, train_interval=None, test_interval=None):
        """Base class initialization called by child classes

        For train/test split, only one of train_interval or
        test_interval may be provided.

        Parameters
        ----------
        league : League instance
        train_interval : int or None
            Interval used to define train/test split of games.  Use a
            value of 1 to train on every game.  A value of 2 will
            train on every other game, etc.
        test_interval : int or None
            Interval used to define train/test split of games.  Use a
            value of 2 to test on every other game, 3 to test on every
            3rd game, etc.
        """
        if train_interval and test_interval:
            raise ValueError
        self.df_teams = league.teams
        # The double_games table may be modified by the fitting
        # routine, so we let the fitting routine store single_games
        # once that is done.
        self.double_games = league.double_games
        self.double_schedule = league.double_schedule
        self.single_schedule = self.double_schedule[ self.double_schedule['team_id'] < self.double_schedule['opponent_id'] ]

        # Flagging the train/test set is a little tricky because we
        # work with double games.  To handle this, we can set it based
        # on the DataFrame index, which is unique by game.
        if train_interval:
            self.double_games['train'] = False
            self.double_games.loc[self.double_games.index.unique()[::train_interval], 'train'] = True
        elif test_interval:
            self.double_games['train'] = True
            self.double_games.loc[self.double_games.index.unique()[::test_interval], 'train'] = False
        else:
            self.double_games['train'] = True

    def summarize(self):
        """Print summary information to screen"""
        cv_flag = (not self.double_games['train'].all())

        print('{} played games'.format(len(self.double_games)//2))
        if cv_flag:
            print('{} trained games'.format(sum(self.double_games['train'])//2))
        num_sched = len(self.double_schedule)//2
        if num_sched > 0:
            print('{} scheduled games'.format(num_sched))

        if self.homecourt:
            print('home advantage: {:.1f}'.format(self.home_adv))

        print('Consistency: {:.3f}'.format(self.consistency))
        if hasattr(self, 'loo_consistency'):
            print('LOO consistency: {:.3f}'.format(self.loo_consistency))
        if cv_flag:
            correct, total = self.evaluate_predicted_wins(exclude_train=True)
            print('CV consistency: {:.3f}'.format(correct/total))
        print('Log lhood: {:.3f}'.format(self.log_likelihood()))
        if cv_flag:
            print('CV log lhood: {:.3f}'.format(self.log_likelihood(exclude_train=True)))
        

    def store_ratings(self, ratings, offense=None, defense=None):
        """After child method is called, organize rating data into DataFrame"""
        self.df_teams['rating'] = ratings
        self.df_teams['rank'] = rank_array(ratings)
        if offense is not None:
            self.df_teams['offense'] = offense
            self.df_teams['offense_rank'] = rank_array(offense)
        if defense is not None:
            self.df_teams['defense'] = defense
            self.df_teams['defense_rank'] = rank_array(defense)

        self.get_strength_of_schedule()

    def get_strength_of_schedule(self):
        """Compute strength of schedule as average of opponent rating

        Call into child class method to compute schedule strength from
        array of opponent ratings.

        For now, does not account for home court"""
        self.df_teams['strength_of_schedule_past'] = np.nan
        self.df_teams['strength_of_schedule_future'] = np.nan
        self.df_teams['strength_of_schedule_all'] = np.nan
        for team_id,team in self.df_teams.iterrows():
            games = self.double_games[self.double_games['team_id'] == team_id]
            schedule = self.double_schedule[self.double_schedule['team_id'] == team_id]
            self.df_teams.at[team_id,'strength_of_schedule_past'] = self.strength_of_schedule(self.df_teams.loc[games['opponent_id'],'rating'])
            if len(schedule) > 0:
                self.df_teams.at[team_id,'strength_of_schedule_future'] = self.strength_of_schedule(self.df_teams.loc[schedule['opponent_id'],'rating'])
            else:
                self.df_teams.at[team_id,'strength_of_schedule_future'] = np.nan
            self.df_teams.at[team_id,'strength_of_schedule_all'] = self.strength_of_schedule(self.df_teams.loc[np.concatenate((games['opponent_id'],schedule['opponent_id'])),'rating'])

    def display_ratings(self, n=10):
        print(self.df_teams.sort_values(by='rating', ascending=False).head(n))

    def store_predictions(self):
        """Compute and store predictions for scheduled games"""
        self.double_games['predicted_result'] = self.predict_result(self.double_games)
        self.double_games['win_probability'] = self.predict_win_probability(self.double_games)
        self.double_schedule['win_probability'] = self.predict_win_probability(self.double_schedule)
        self.consistency = sum(self.double_games['predicted_result']==self.double_games['result']) / float(len(self.double_games))
        if hasattr(self, 'single_games') and 'loo_predicted_result' in self.single_games:
            games = self.single_games[self.single_games['train']]
            self.loo_consistency = sum(games['loo_predicted_result']==games['result']) / float(len(games))

        # Expected wins, losses:
        exp_wins = [int(round(sum(self.double_schedule.loc[self.double_schedule['team_id']== tid,'win_probability']))) + self.df_teams.at[tid,'wins'] for tid in self.df_teams.index]

        self.df_teams['expected_losses'] = [sum(self.double_games['team_id']==tid) + sum(self.double_schedule['team_id']==tid) - exp_wins[i] for i,tid in enumerate(self.df_teams.index)]
        self.df_teams['expected_wins'] = exp_wins

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

    def log_likelihood(self, exclude_train=False):
        """Evaluate log of likelihood of outcomes based on predicted win probabilities"""
        games = self.double_games[ self.double_games['team_id'] < self.double_games['opponent_id'] ]
        if exclude_train:
            games = games[~ games['train']]

        pvals = games.apply(lambda r: r['win_probability'] if r['result']=='W' else 1.0 - r['win_probability'], axis=1)
        return sum(np.log(pvals))

    def evaluate_coverage_probability(self, exclude_train=False):
        pvals = np.array([0.5, 0.6, 0.7, 0.8, 0.9])
        counts = np.zeros(len(pvals), dtype=int)
        correct = np.zeros(len(pvals), dtype=int)

        total_count = 0 # Sanity check

        # Use double games for book-keeping simplicity...
        if exclude_train:
            games = self.double_games[self.double_games['train']]
        else:
            games = self.double_games

        pred_probs = self.predict_win_probability(games)
        pred_outcomes = ['W' if p>0.5 else 'L' for p in pred_probs]
        for p,wl,(index,game) in zip(pred_probs,pred_outcomes,games.iterrows()):
            if p > 0.5:
                total_count += 1
                # Determine interval
                interval = np.where(p>pvals)[0][-1]
                counts[interval] += 1
                if wl == game['result']:
                    correct[interval] += 1

        print("Total count:", total_count)
        for i,p in enumerate(pvals):
            print('Coverage for {}: {} / {} ({:.2})'.format(p, correct[i], counts[i], float(correct[i])/counts[i]))

    def to_db(self, engine, rating_name, finished=False):
        """Write to database

        Create "teams" and "games" tables.  The "games" table also
        includes scheduled games.  Each game in the games table is
        represented twice, once for each team in the team_id position

        Parameters
        ----------
        rating_name : str
            A unique name for the rating
        finished : bool
            Flag for whether the season is finished, used by website.
        """

        with engine.connect() as conn:
            for s in schema.split('\n\n'):
                conn.execute(s)

            ## properties table (general info)
            today = pd.to_datetime(datetime.datetime.today())
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
            n_games = len(self.double_games) // 2
            n_scheduled = len(self.double_schedule) // 2
            Rsquared = self.Rsquared if hasattr(self, 'Rsquared') else None
            conn.execute('UPDATE ratings SET home_advantage = ?, r_squared = ?, consistency=?, games_played = ?, games_scheduled = ?, finished = ? WHERE rating_id = ?;', (self.home_adv, Rsquared, self.consistency, n_games, n_scheduled, finished, rating_id))

            ### teams table
            df = self.df_teams.copy()
            df['rating_id'] = rating_id
            df['team_id'] = df.index

            # First delete previous entries for this league:
            conn.execute('DELETE FROM teams WHERE rating_id=?;', (rating_id,))

            df.to_sql("teams", engine, if_exists='append', index=False,
                      dtype = {'team_id': sqlt.Integer,
                               'rating_id': sqlt.Integer,
                               'name': sqlt.Text,
                               'rating': sqlt.Float,
                               'rank': sqlt.Integer,
                               'wins': sqlt.Integer,
                               'losses': sqlt.Integer,
                               'expected_wins': sqlt.Integer,
                               'expected_losses': sqlt.Integer,
                               'offense_rank': sqlt.Integer,
                               'defense_rank': sqlt.Integer})

            ### games table
            # Using reindex both selects columns and creates NA
            # columns if not present (using .loc for this will trigger
            # a warning if requested columns are not present)
            df = self.double_games.reindex(columns=['team_id','opponent_id','points','opponent_points','location','date','normalized_score','result','win_probability'])
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
            df = self.double_schedule.loc[:,['team_id','opponent_id','location','date','win_probability']]
            df['rating_id'] = rating_id

            df.to_sql("games", engine, if_exists='append', index=False,
                      dtype = {'team_id': sqlt.Integer,
                               'rating_id': sqlt.Integer,
                               'opponent_id': sqlt.Integer,
                               'date': sqlt.Date})
