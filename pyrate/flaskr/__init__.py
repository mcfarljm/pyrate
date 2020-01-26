import os

from flask import Flask, render_template, request

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
        ratings = db.get_rating_system_names()
        # print("names:", ratings)
        ratings_summary = db.get_rating_systems()
        updated = db.date_updated().strftime('%Y-%m-%d %H:%M')
        fmts = {'Home Adv':'{:.1f}',
                'R<sup>2</sup>':'{:.2f}',
                'Consist':'{:.2f}',}
        return render_template('home.html', ratings=ratings, updated=updated, ratings_summary=ratings_summary.style.hide_index().format(fmts).set_properties(subset=['Home Adv','Consist'], **{'text-align':'center'}).render(escape=False))

    @app.route('/<rating>')
    def rating_system(rating):
        ratings = db.get_rating_system_names()
        df = db.get_rating_table(rating)
        updated = db.date_updated().strftime('%Y-%m-%d %H:%M')
        fmts = {'Rating': '{:.2f}'}

        if request.args.get('mode') == 'rating':
            fmts.update({
                'SoS(p)': '{:.1f}',
                'SoS(f)': '{:.1f}',
                'SoS(a)': '{:.1f}',
                'Off': '{:.1f}',
                'Def': '{:.1f}'})
            
        return render_template('ratings.html', rating=rating, ratings=ratings, updated=updated, table=df.style.hide_index().format(fmts).set_table_attributes('class="dataframe"').set_uuid('ratingTable').render(escape=False))

    @app.route('/<rating>/<team>')
    def team_page(rating, team):
        ratings = db.get_rating_system_names()
        team_id = db.get_team_id(rating, team)
        df = db.get_games_table(rating, team_id)
        df_sched = db.get_scheduled_games(rating, team_id)

        td = db.get_team_data(rating, team_id)

        fmts = {'Date': lambda x: "{}".format(x.strftime('%Y-%m-%d')),
                'NS': '{:.0f}'}
        # Try to be smart about formatting for NS.  When using just
        # wins, the values are small and .0f is not good.
        if max(df['NS'].abs()) < 4:
            fmts['NS'] = '{:.2f}'

        fmts_sched = fmts.copy() # 'Result' format is different
        fmts_sched['Result'] = '{:.0f}%'

        def color_outcome(s):
            return ['color: green' if v=='W' else 'color: red' for v in s]

        games = df.style.hide_index().format(fmts).apply(color_outcome, subset='Result').set_uuid('gameTable')
        if 'NS' in df:
            games = games.set_properties(subset=['NS'], **{'text-align':'center'}).bar(subset=['NS'], align='zero') 
        scheduled = df_sched.style.hide_index().format(fmts_sched).set_uuid('scheduleTable').render(escape=False)
        
        return render_template('team.html', rating=rating, team=team, ratings=ratings, team_data=td, table=games.render(escape=False), scheduled=scheduled)


    db.init_app(app)

    return app
