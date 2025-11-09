"""Helper classes for formulating game outcome measure"""

import numpy as np
import scipy.optimize
import scipy.stats

try:
    import matplotlib.pyplot as plt
except:
    pass


class GameOutcomeMeasure:
    """Base class for game outcome measure

    Derived class must implement __call__ method, which accepts array
    of point_differences and returns array of corresponding game
    outcome measures

    """

    def plot(self, ub=None, lb=0):
        if ub is None:
            if hasattr(self, 'max_point_diff'):
                ub = self.max_point_diff
            else:
                ub = 20

        xvals = np.linspace(lb, ub, 200)
        f, ax = plt.subplots()
        ax.plot(xvals, self(xvals))
        xpoints = np.arange(lb, ub+1)
        ax.plot(xpoints, self(xpoints), 'o')
        ax.grid()
        ax.set_xlabel('Point difference')
        ax.set_ylabel('Game outcome measure')

class PointDifference(GameOutcomeMeasure):
    supports_off_def = True
    def __init__(self):
        pass
    def __call__(self, point_diff):
        return point_diff

class CappedPointDifference(GameOutcomeMeasure):
    def __init__(self, cap=15):
        self.max_point_diff = cap
        if cap > 1:
            self.supports_off_def = True
        else:
            self.supports_off_def = False
    def __call__(self, point_diff):
        return np.sign(point_diff) * np.fmin(self.max_point_diff, np.abs(point_diff))

class BetaCurve(GameOutcomeMeasure):
    def __init__(self, max_point_diff=20, gom_at_1=3, max_gom=None):
        """
        Parameters
        ----------
        max_point_diff : int
            Limit after which game outcome measure no longer increases
        gom_at_1 : int
            Value of the game outcome measure when the point
            difference is 1 (i.e., the amount earned by winning)
        max_gom : int or None
            Value of the game outcome measure at max_point_diff.  If
            None, set equal to max_point_diff.
        """
        if max_gom is None:
            max_gom = max_point_diff

        normed_gom_at_1 = gom_at_1 / float(max_gom)
        xval = 1.0 / max_point_diff
        def root_func(alpha):
            return scipy.stats.beta.cdf(xval, alpha, 1.0/alpha) - normed_gom_at_1
        sol = scipy.optimize.root_scalar(root_func, bracket=[0.05, 10.0])

        self.alpha = sol.root
        self.beta = 1.0/self.alpha
        self.max_point_diff = max_point_diff
        self.max_gom = max_gom
        self.rv = scipy.stats.beta(self.alpha, self.beta, scale=self.max_point_diff)

    def __call__(self, point_diff):
        return np.sign(point_diff) * self.max_gom * self.rv.cdf(np.abs(point_diff))


if __name__ == '__main__':
    gom = BetaCurve()
    print('alpha:', gom.alpha)
    gom.plot()
    plt.show()
