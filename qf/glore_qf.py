"""
GLORE-based logistic regression quality function for pysubgroup GP-growth.

LogRegQF (logreg_qf.py) gives us the global fit (self.X, self.y, self.beta_global),
the n_min check, the quality q(S) = ||chi(v) - beta_global||_2 and evaluate.
This file only adds the model configuration (V, (+), phi, chi):

    phi_lr(i) = ( 1, (y_i - pi_i) x_i, pi_i (1 - pi_i) x_i x_i^T ),  pi_i at beta_global
    v (+) u   = component-wise addition
    chi_lr(v) = beta_global + v_H^{-1} v_g     (one Newton-Raphson step)

A valuation basis v is one flat vector:

    [ n | g_0 .. g_m | upper triangle of H ]      length 1 + p + p(p+1)/2,  p = m + 1

Usage:
    qf = GloreLogRegQF(y_name="y", x_names=["x1", "x2"], n_min=30)
    result = ps.GpGrowth().execute(task)
"""

import numpy as np

from .logreg_qf import LogRegQF


def sigmoid(z):
    return 0.5 * (1.0 + np.tanh(0.5 * z))  # same as 1 / (1 + exp(-z)), no overflow


def solve_newton_step(H, g, cond_max):
    """H^{-1} g, or None if H is singular / too ill-conditioned to trust."""
    if not np.all(np.isfinite(H)) or np.linalg.cond(H) > cond_max:
        return None
    return np.linalg.solve(H, g)


class GloreLogRegQF(LogRegQF):
    def __init__(self, y_name, x_names, n_min=30, cond_max=1e12):
        super().__init__(y_name, x_names, n_min)
        self.cond_max = cond_max
        self.p = len(self.x_names) + 1  # coefficients, including the intercept
        self.triu = np.triu_indices(self.p)  # H is symmetric, so we only store this part
        self.dim = 1 + self.p + len(self.triu[0])
        self.phi = None  # phi_lr(i) for every row, shape (N, dim)

    def calculate_constant_statistics(self, data, target):
        super().calculate_constant_statistics(data, target)
        X, y = self.X, self.y
        pi = sigmoid(X @ self.beta_global)
        g = X * (y - pi)[:, None]
        H = (pi * (1 - pi))[:, None, None] * X[:, :, None] * X[:, None, :]
        self.phi = np.column_stack([np.ones(len(y)), g, H[:, self.triu[0], self.triu[1]]])

    # model configuration (V, (+), phi, chi)
    def gp_get_null_vector(self):
        return np.zeros(self.dim)

    def gp_get_stats(self, row_index):
        return self.phi[row_index].copy()  # copy: GP-growth adds into it in place

    @staticmethod
    def gp_merge(left, right):
        left += right

    @staticmethod
    def gp_size_sg(v):
        return v[0]

    def chi(self, v):
        _, g, H = self.unpack(v)
        step = solve_newton_step(H, g, self.cond_max)
        return None if step is None else self.beta_global + step

    def unpack(self, v):
        """v -> (n, g, H), with H rebuilt as the full symmetric matrix."""
        n, g = v[0], v[1 : 1 + self.p]
        H = np.zeros((self.p, self.p))
        H[self.triu] = v[1 + self.p :]
        H = H + np.triu(H, 1).T
        return n, g, H

    # DFS path: sum phi over the covered rows in one go instead of row by row
    def calculate_statistics(self, subgroup, target, data, statistics=None):
        cover = subgroup if isinstance(subgroup, np.ndarray) else subgroup.covers(data)
        return self.gp_get_params(None, self.phi[cover].sum(axis=0))
