"""
Logistic regression EMM with the *trivial valuation basis* for pysubgroup's GP-growth.

Lemmerich, Becker & Atzmueller (2012), "Generic Pattern Trees for Exhaustive
Exceptional Model Mining":
  - Sec. 3: the trivial valuation basis is the set of instances itself, so any
    model can be fitted from it.
  - Sec. 4: no exact single-pass algorithm is known for logistic regression, so
    it "relies on the trivial valuation basis".

    V        = sets of instances (stored as lists of row indices)
    phi(i)   = [i]
    v (+) u  = v + u                 (union; FP-tree branches are disjoint)
    chi(v)   = logistic regression MLE on the rows in v

Usage:
    qf = TrivialLogRegQF(y_name="y", x_names=["x1", "x2"], n_min=30)
    task = ps.SubgroupDiscoveryTask(data, ps.FITarget(), searchspace, qf,
                                    result_set_size=10, depth=2,
                                    constraints=[ps.MinSupportConstraint(30)])
    result = ps.GpGrowth().execute(task)
"""

from .logreg_qf import LogRegQF, fit_logreg


class TrivialLogRegQF(LogRegQF):
    def gp_get_null_vector(self):
        return []

    def gp_get_stats(self, row_index):  # phi
        return [row_index]

    @staticmethod
    def gp_merge(left, right):  # (+), in place
        left.extend(right)

    @staticmethod
    def gp_size_sg(v):
        return len(v)

    def chi(self, v):
        return fit_logreg(self.X[v], self.y[v])
