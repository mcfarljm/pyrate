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
    def leagues():
        leagues = db.get_leagues()
        print('leagues:', leagues)
        return render_template('leagues.html', leagues=leagues)

    @app.route('/<league>')
    def rankings(league):
        print('rankings page:', league)
        df = db.get_teams_table(league)
        fmts = {'Rating': '{:.2f}',
                'SoS': '{:.2f}'}
        return render_template('teams.html', table=df.style.hide_index().format(fmts).set_table_attributes('class="dataframe"').render(escape=False))

    @app.route('/<league>/<team>')
    def team_page(league, team):
        df = db.get_games_table(league, team)

        fmts = {'Date': lambda x: "{}".format(x.strftime('%m/%d')),
                'NS': '{:.0f}'}

        wins = sum(df['Result'] == 'W')
        losses = sum(df['Result'] == 'L')

        def color_outcome(s):
            return ['color: green' if v=='W' else 'color: red' for v in s]
        
        return render_template('games.html', league=league, team=team, wins=wins, losses=losses, table=df.style.hide_index().format(fmts).apply(color_outcome, subset='Result').set_properties(subset=['NS'], **{'text-align':'center'}).bar(subset=['NS'], align='zero').render(escape=False))


    db.init_app(app)

    return app