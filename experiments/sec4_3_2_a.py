"""
uv run python -m experiments.sec4_3_2_a
"""

from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from .datasets import SPECS
from .methods import METHODS
from .runner import RESULTS, append, run_one

# ---- config -----------------------------------------------------------------
DATASETS = list(SPECS)
METHOD_NAMES = list(METHODS)
DEPTHS = [2, 3, 4,5]
REPEATS = 1
K = 10
N_MIN = 30
TIMEOUT = 1800 # 30mins
OUT = RESULTS / "sec4_3_2_a.csv"

FIELDS = ["dataset", "method", "depth", "repeat", "status", "seconds",
          "n_rows", "n_selectors", "k", "n_min", "timeout", "best_quality"]


def run_all(datasets: list[str], methods: list[str], depths: list[int], repeats: int,
            k: int, n_min: int, timeout: float, out: Path) -> None:
    for dataset, method in product(datasets, methods):
        timed_out = False
        for depth in sorted(depths):
            for rep in range(repeats):
                row = {"dataset": dataset, "method": method, "depth": depth, "repeat": rep,
                       "k": k, "n_min": n_min, "timeout": timeout}
                if timed_out:
                    row["status"] = "timeout (skipped)"
                    append(row, out, FIELDS)
                    break
                row.update(run_one(dataset, method, depth, k, n_min, timeout))
                append(row, out, FIELDS)
                if row["status"] == "timeout":
                    timed_out = True
                    break


def summarize(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    timed_out = df.status.str.startswith("timeout")
    df.loc[timed_out, "seconds"] = df.loc[timed_out, "timeout"]  # at least this long
    df = df[(df.status == "ok") | timed_out]
    keys = ["dataset", "depth", "method"]

    table = df.groupby(keys).seconds.median().unstack("method")
    is_bound = df.groupby(keys).status.apply(lambda s: s.str.startswith("timeout").all()).unstack("method")
    for col in ["DFS-Full", "DFS-Simple"]:
        if col in table and "GP-GLORE" in table:
            table[f"speedup vs {col}"] = table[col] / table["GP-GLORE"]

    shown = table.round(2).astype(object)
    for col in is_bound.columns:  
        # mark timeouts as lower bounds: ">21600"
        shown.loc[is_bound[col].fillna(False).astype(bool), col] = (
            ">" + table.loc[is_bound[col].fillna(False).astype(bool), col].round(0).astype(int).astype(str))
    print("median seconds ('>' = timed out, speedups against those are lower bounds):")
    print(shown.to_string())

    # geometric mean of the speedups over datasets, per depth
    for col in [c for c in table.columns if c.startswith("speedup")]:
        gm = table[col].groupby("depth").apply(lambda s: float(np.exp(np.log(s.dropna()).mean())))
        print(f"\ngeometric mean {col}:")
        print(gm.round(2).to_string())
    return table


if __name__ == "__main__":
    run_all(DATASETS, METHOD_NAMES, DEPTHS, REPEATS, K, N_MIN, TIMEOUT, OUT)
    summarize(OUT)
