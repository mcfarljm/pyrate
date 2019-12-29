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
        
        if df_teams is None:
            self.teams = pd.DataFrame(index=games['team_id'].unique())
        else:
            self.teams = df_teams

        if not duplicated_games:
            df2 = df_games.rename(columns={'team_id':'opponent_id', 'opponent_id':'team_id',
                                           'points':'opponent_points','opponent_points':'points',
                                           'location':'opponent_location','opponent_location':'location'})
            df_games = pd.concat((df_games,df2))
            

        # Split up into games and schedule
        unplayed = (df_games['points'].isnull() | df_games['opponent_points'].isnull())
        self.double_games = df_games[~unplayed].copy()
        self.double_scheduled = df_games[unplayed].copy()
        self.double_games = self.doublegames.astype({'points':'int32', 'opponent_points':'int32'})
        self.double_games['train'] = True
        

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

        games = df_games.set_index('game_id')
        games = games.join(games, rsuffix='2')
        # Each index was originally represented twice, so after join
        # now appears 4 times, 2 combinations of which are not valid
        games = games[games['team_id'] != games['team_id2']]
        # Now have double-games format; rename the columns
        games = games.rename(columns={'team_id2':'opponent_id',
                                      'points2':'opponent_points',
                                      'location2':'opponent_location'})
        
        # For compatibility with Massey data, treat 0 points as
        # scheduled game (could be added as a flag)
        scheduled = (games['points'] == 0) & (games['opponent_points'] == 0)
        games.loc[scheduled,['points','opponent_points']] = np.nan # Flag scheduled games

        return cls(games, df_teams)


class RatingSystem:
    def __init__(self, league):
        self.league = league

        # The double_games table may be modified by the fitting
        # routine, so we let the fitting routine store single_games
        # once that is done.
        self.double_games = pd.concat([t.games for t in self.teams], ignore_index=True)
        self.double_schedule = pd.concat([t.scheduled for t in self.teams], ignore_index=True)

    def summarize(self):
        print('{} played games'.format(sum([len(t.games) for t in self.teams])//2))
        num_sched = sum([len(t.scheduled) for t in self.teams])//2
        if num_sched > 0:
            print('{} scheduled games'.format(num_sched))
        if self.homecourt:
            print('home advantage: {:.1f}'.format(self.home_adv))

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
        self.double_games['win_probability'] = self.predict_win_probability(self.double_games)
        self.double_schedule['win_probability'] = self.predict_win_probability(self.double_schedule)
        self.consistency = sum(self.double_games['predicted_result']==self.double_games['result']) / float(len(self.double_games))

        # Expected wins, losses:
        exp_wins = [int(round(sum(self.double_schedule.loc[self.double_schedule['team_id']==t.id,'win_probability']))) + t.wins for t in self.teams]
        self.ratings['expected_losses'] = [len(t.games) + len(t.scheduled) - exp_wins[i] for i,t in enumerate(self.teams)]
        self.ratings['expected_wins'] = exp_wins

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
                               'losses': sqlt.Integer,
                               'expected_wins': sqlt.Integer,
                               'expected_losses': sqlt.Integer})

            ### games table
            df = self.double_games.loc[:,['team_id','opponent_id','points','opponent_points','location','date','normalized_score','result','win_probability']]
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
