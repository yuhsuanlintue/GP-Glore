"""
Sec. 4.3.1:
    DFS-Full     ps.DFS      + TrivialLogRegQF   exact fit per subgroup (quality reference)
    DFS-Simple   ps.DFS      + GloreLogRegQF     one GLORE step per subgroup, no tree
    GP-GLORE     ps.GpGrowth + GloreLogRegQF     our method

Usage:
    result, seconds = run_method("DFS-Full", load_dataset("credit-g"), depth=2)
"""

import time
import warnings

warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

import pysubgroup as ps

from qf.glore_qf import GloreLogRegQF
from qf.trivial_qf import TrivialLogRegQF

from .datasets import Y, load_dataset

METHODS = {
    "DFS-Full": (ps.DFS, TrivialLogRegQF),
    "DFS-Simple": (ps.DFS, GloreLogRegQF),
    "GP-GLORE": (ps.GpGrowth, GloreLogRegQF),
}


def make_task(ds, qf, depth, k, n_min):
    return ps.SubgroupDiscoveryTask(
        ds.data, ps.FITarget(), ds.selectors, qf,
        result_set_size=k, depth=depth,
        constraints=[ps.MinSupportConstraint(n_min)],
    )


def run_method(name, ds, depth=2, k=10, n_min=30):
    """Run one method; returns (SubgroupDiscoveryResult, wall-clock seconds).
    The time includes fitting the global model (done inside execute)."""
    algo_cls, qf_cls = METHODS[name]
    task = make_task(ds, qf_cls(Y, ds.x_names, n_min=n_min), depth, k, n_min)
    t0 = time.perf_counter()
    result = algo_cls().execute(task)
    return result, time.perf_counter() - t0


if __name__ == "__main__":
    ds = load_dataset("credit-g")
    print(ds)
    result, secs = run_method("DFS-Simple", ds, depth=2, k=5)
    print(f"\nDFS-Simple: {secs:.2f}s")
    print(result.to_dataframe()[["subgroup", "quality", "size_sg"]].to_string(index=False))
