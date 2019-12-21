# PyRate

PyRate is a Python package for sports ratings.  It is intended to
provide a simple interface for testing and evaluating rating systems
such as least-squares rating.  It is capable of reading in team and
game data from basic CSV file formats and storing resulting rating
data to a database.  It also includes a Flask module that creates a
website using data from the database.

## About least squares ratings

The objective of the rating system is to assign quantitative ratings
to a set of teams based on a given set of historical outcomes.  Least
squares sports ratings are attractive because they are simple to
implement, easy to interpret, and allow for a variety of modifications
such as inclusion of a home field advantage, custom game outcome
measures, and game weighting schemes.

The least squares method begins with the concept of a "Game Outcome
Measure" that is used to quantify the results of a particular game.  A
simple approach is to define the game outcome measure as the
difference in points, although other measures are possible.  The basic
idea is to attempt to explain the game outcome measure for each game
in terms of the difference in ratings between the two teams.  As an
example, suppose that Team A has a rating of 10, Team B has a rating
of 6, and Team C has a rating of 0.  Assuming that points are used as
the game outcome measure, these ratings imply that we would expect
Team A to beat Team B by 4 points, Team B to beat Team C by 6 points,
and Team A to beat Team C by 10 points.  Note that these are only the
*expected* results, and that actual outcomes can vary.

The objective is to determine the rating values for each team based on
a set of past outcomes.  Of course, the ratings will not perfectly
explain every outcome, so it is necessary to establish some definition
of what constitutes the "best" ratings.  Note that for any particular
game, we can make a comparison between the actual outcome and what is
predicted by the team ratings.  Let *y<sub>i</sub>* denote the actual
outcome for a game between teams A and B.  The expected value of the
outcome is equal to *r<sub>A</sub> - r<sub>B</sub>*.  The difference
between the actual and expected outcome is *y<sub>i</sub> -
(r<sub>A</sub> - r<sub>B</sub>)*.  We seek ratings that minimize these
differences between expected and actual outcomes.  In least squares,
this is done by minimizing the sum of the squares of the differences
across all games.

Note that the predicted outcome of a game depends only on the
difference in team ratings.  Thus, a constant value *c* can be added
to all ratings without affecting the model.  This means that there is
not a unique solution to the least squares ratings.  This can be
easily resolved by imposing an additional constraint that fixes the
"location" of the overall set of rating values.  Typically, this is
done by constraining the sum (and hence the average) of the ratings to
be zero.  This is very convenient because it means that a rating value
of zero denotes an average team.  Taking the game outcomes together
with this additional constraint produces a system of equations that
can be solved directly using linear algebra: the result is the set of
rating values for the teams.

It was already noted that the difference in the ratings between two
teams can be interpreted as the expected value of the game outcome
measure (e.g., point difference).  Several other interpretations of
the least squares approach are possible.  For a given team, we can
compute a "*normalized score*" for each game as the sum of the game
outcome measure (expressed relative to that team, i.e., positive for a
win and negative for a loss) and the opponent's rating.  For example,
a 10-point win over a team with a -5.0 rating corresponds to a
normalized score of 5, whereas a 10-point win over a team with a 5.0
rating corresponds to a normalized score of 15.  Thus, the normalized
score measures the performance relative to the strength of the
opponent.  It turns out that due to the additive nature of the least
squares model, each team's rating is equal to the average of the
normalized scores from their games.

The ratings can also be used to evaluate strength of schedule.  A
simple metric for schedule strength is the average rating of a team's
opponents.  Using this metric, a strength of schedule of 0 indicates
that on average, the strength of the team's opponents is average.
Again, due to the additive nature of the system, the ratings can
further be interpreted in terms of schedule strength: each team's
rating is equal to the sum of their strength of schedule (average
opponent rating) and their average game outcome measure (e.g., point
difference).  This is another view that indicates how each team's
rating accounts for both the strength of the opposition and their
performance against that opposition.

It is also possible to account for the effect of home field advantage
within the least squares method.  This can be done by adding one new
parameter to the model, *h*, which represents the average home field
advantage across the entire league.  With this formulation, the
expected outcome for a particular game is given by *r<sub>A</sub> -
r<sub>B</sub> + h &middot; x<sub>h</sub>*, where *x<sub>h</sub>* is an
indicator that is assigned 1 for a home game and -1 for an away game.
The equations are modified accordingly, so that the solution produces
values for the team ratings and the universal home field advantage.
Note that home field advantage is expressed in terms of the game
outcome measure, which does not necessarily need to be defined as the
point difference.
