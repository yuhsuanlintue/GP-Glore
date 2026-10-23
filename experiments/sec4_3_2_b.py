"""
Sec. 4.3.2 b
uv run python -m experiments.sec4_3_2_b
"""

import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import pysubgroup as ps

from qf.glore_qf import GloreLogRegQF

from .datasets import Y, load_dataset
from .methods import make_task
from .runner import RESULTS, append

# ---- config -----------------------------------------------------------------
DATASET = "credit-g"
X_NAMES = ("duration", "credit_amount")
DEPTHS = [2, 3, 4, 5]
REPEATS = 20  # runs take seconds, so a median over a few is cheap
K = 10
N_MIN = 30
OUT = RESULTS / "sec4_3_2_b.csv"

FIELDS = ["method", "depth", "repeat", "seconds"]


def run_all(depths: list[int], repeats: int, k: int, n_min: int, out: Path) -> None:
    ds = load_dataset(DATASET, x_names=X_NAMES)
    models = {
        "FP": ps.CountQF,
        "LR": lambda: GloreLogRegQF(Y, ds.x_names, n_min=n_min),
    }
    for model, depth, rep in product(models, sorted(depths), range(repeats)):
        qf = models[model]()
        # no pruning (for FP)
        qf.optimistic_estimate = lambda *args, **kwargs: np.inf
        task = make_task(ds, qf, depth, k, n_min)
        t0 = time.perf_counter()
        ps.GpGrowth().execute(task)
        secs = time.perf_counter() - t0
        row = {"method": model, "depth": depth, "repeat": rep, "seconds": secs}
        append(row, out, FIELDS)


def summarize(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    table = df.groupby(["depth", "method"]).seconds.median().unstack("method")
    table["LR / FP"] = table["LR"] / table["FP"]
    print("median seconds per depth, and how many times slower LR is than FP:")
    print(table.round(3).to_string())
    return table


if __name__ == "__main__":
    run_all(DEPTHS, REPEATS, K, N_MIN, OUT)
    summarize(OUT)
