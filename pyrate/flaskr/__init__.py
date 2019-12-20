import os

from flask import Flask, render_template

from . import db


def create_app(test_config=None):
    # create and configure the app
    db_path = os.path.join(os.path.dirname(__file__), 'static', 'pyrate.db')
    db_uri = 'sqlite:///{}'.format(db_path)
    
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_mapping(
        DATABASE=db_uri,
    )

    if test_config is None:
        # load the instance config, if it exists, when not testing
        app.config.from_pyfile('config.py', silent=True)
    else:
        # load the test config if passed in
        app.config.from_mapping(test_config)

    @app.route('/')
    def home():
        ratings = db.get_rating_systems()
        updated = db.date_updated().strftime('%Y-%m-%d %H:%M')
        return render_template('home.html', ratings=ratings, updated=updated)

    @app.route('/<rating>')
    def rating_system(rating):
        print('rankings page:', rating)
        df = db.get_rating_table(rating)
        updated = db.date_updated().strftime('%Y-%m-%d %H:%M')
        fmts = {'Rating': '{:.2f}',
                'SoS': '{:.2f}'}
        return render_template('ratings.html', rating=rating, updated=updated, table=df.style.hide_index().format(fmts).set_table_attributes('class="dataframe"').render(escape=False))

    @app.route('/<rating>/<team>')
    def team_page(rating, team):
        team_id = db.get_team_id(rating, team)
        df = db.get_games_table(rating, team_id)
        df_sched = db.get_scheduled_games(rating, team_id)

        td = db.get_team_data(rating, team_id)

        fmts = {'Date': lambda x: "{}".format(x.strftime('%m/%d')),
                'NS': '{:.0f}'}

        def color_outcome(s):
            return ['color: green' if v=='W' else 'color: red' for v in s]

        games = df.style.hide_index().format(fmts).apply(color_outcome, subset='Result').set_properties(subset=['NS'], **{'text-align':'center'}).bar(subset=['NS'], align='zero').render(escape=False)
        scheduled = df_sched.style.hide_index().format(fmts).render(escape=False)
        
        return render_template('team.html', rating=rating, team=team, team_data=td, table=games, scheduled=scheduled)


    db.init_app(app)

    return app
