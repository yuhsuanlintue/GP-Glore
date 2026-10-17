"""
Base class for logistic regression EMM quality functions in pysubgroup.
"""

import warnings
from abc import abstractmethod
from collections import namedtuple

import numpy as np
import pysubgroup as ps
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import PerfectSeparationError

class LogRegStats(namedtuple("LogRegStats", ["size_sg", "beta"])):
    # pysubgroup keeps results in a heap of (quality, subgroup, stats) tuples, and GP-growth
    # passes subgroup=None for single selectors. Two selectors with the same quality then
    # make Python compare the stats, and comparing numpy arrays raises. Order doesn't matter
    # on a tie, so treat all stats as equal.
    __slots__ = ()

    def __eq__(self, other):
        return True

    def __lt__(self, other):
        return False


def fit_logreg(X, y):
    """Logistic regression MLE, or None if it does not exist / fails to converge."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            res = sm.Logit(y, X).fit(disp=0)
        except (np.linalg.LinAlgError, PerfectSeparationError):
            return None
    beta = np.asarray(res.params)
    if not res.mle_retvals["converged"] or not np.all(np.isfinite(beta)):
        return None
    return beta


class LogRegQF(ps.AbstractInterestingnessMeasure):
    def __init__(self, y_name, x_names, n_min=30):
        self.y_name = y_name
        self.x_names = list(x_names)
        self.n_min = n_min
        self.X = None
        self.y = None
        self.beta_global = None
        self.has_constant_statistics = False
        self.required_stat_attrs = LogRegStats._fields
        self.gp_requires_cover_arr = False

    def calculate_constant_statistics(self, data, target):
        """Design matrix (with intercept), target and the global model.
        Subclasses may extend this (call super() first) to precompute phi."""
        self.X = sm.add_constant(data[self.x_names].to_numpy(dtype=float), has_constant="add")
        self.y = data[self.y_name].to_numpy(dtype=float)
        self.beta_global = fit_logreg(self.X, self.y)
        if self.beta_global is None:
            raise RuntimeError("Logistic regression on the full dataset did not converge.")
        self.has_constant_statistics = True

    # model configuration (V, (+), phi, chi): defined by subclasses
    @abstractmethod
    def gp_get_null_vector(self): ...

    @abstractmethod
    def gp_get_stats(self, row_index): ...

    @staticmethod
    @abstractmethod
    def gp_merge(left, right): ...

    @staticmethod
    @abstractmethod
    def gp_size_sg(v): ...

    @abstractmethod
    def chi(self, v): ...

    # GpGrowth hooks 
    def gp_get_params(self, cover_arr, v):
        n = self.gp_size_sg(v)
        if n < self.n_min:
            return LogRegStats(n, None)
        return LogRegStats(n, self.chi(v))

    def gp_to_str(self, v):
        return f"n={self.gp_size_sg(v)}"

    # generic pysubgroup interface (DFS, and GpGrowth's selector pass)
    def calculate_statistics(self, subgroup, target, data, statistics=None):
        """Build v^S = phi(i_1) (+) ... (+) phi(i_n) over the covered rows."""
        cover = subgroup if isinstance(subgroup, np.ndarray) else subgroup.covers(data)
        v = self.gp_get_null_vector()
        for i in np.flatnonzero(cover):
            self.gp_merge(v, self.gp_get_stats(i))
        return self.gp_get_params(None, v)

    def evaluate(self, subgroup, target, data, statistics=None):
        stats = self.ensure_statistics(subgroup, target, data, statistics)
        if stats.beta is None:
            return -np.inf
        return float(np.linalg.norm(stats.beta - self.beta_global))

    def optimistic_estimate(self, subgroup, target, data, statistics=None):
        return np.inf 
