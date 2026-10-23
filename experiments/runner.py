"""
Shared bits for running experiments: one run per child process (so it can be
killed at the time limit) and appending each result row to a CSV right away.

Usage:
    row = {"dataset": "credit-g", "method": "DFS-Full", "depth": 2, ...}
    row.update(run_one("credit-g", "DFS-Full", depth=2, k=10, n_min=30, timeout=3600))
    append(row, out, fields)
"""

import multiprocessing as mp
from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pysubgroup as ps

from .datasets import Dataset, load_dataset
from .methods import run_method

RESULTS = Path(__file__).resolve().parent.parent / "results"

# pulls extra fields out of a finished run; must be a module-level function
# (it gets sent to the child process)
Extract = Callable[[ps.SubgroupDiscoveryResult, Dataset], dict]


def best_quality(result: ps.SubgroupDiscoveryResult, ds: Dataset) -> dict:
    return {"best_quality": result.results[0][0] if result.results else float("nan")}


def _worker(queue: mp.Queue, dataset: str, method: str, depth: int, k: int, n_min: int,
            frac: float, seed: int, extract: Extract) -> None:
    """Runs in the child. Loading the data is not timed, only run_method is."""
    try:
        ds = load_dataset(dataset, frac=frac, seed=seed)
        result, secs = run_method(method, ds, depth=depth, k=k, n_min=n_min)
        row = {"status": "ok", "seconds": secs, "n_rows": len(ds.data),
               "n_selectors": len(ds.selectors)}
        row.update(extract(result, ds))
        queue.put(row)
    except Exception as e:  # report it as a row instead of crashing the whole experiment
        queue.put({"status": f"error: {type(e).__name__}: {e}"})


def run_one(dataset: str, method: str, depth: int, k: int, n_min: int, timeout: float,
            frac: float = 1.0, seed: int = 0, extract: Extract = best_quality) -> dict:
    """Run once in a child process; returns the measured fields of a CSV row."""
    queue = mp.Queue()
    proc = mp.Process(target=_worker,
                      args=(queue, dataset, method, depth, k, n_min, frac, seed, extract))
    proc.start()
    proc.join(timeout)
    if proc.is_alive():
        proc.terminate()
        proc.join()
        return {"status": "timeout"}
    return queue.get()


def append(row: dict, out: Path, fields: list[str]) -> None:
    """Add one row to the CSV right away, so a crash or Ctrl+C keeps the finished runs."""
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        header = out.open().readline().strip().split(",")
        if header != fields:  # appending would misalign the columns
            raise ValueError(f"{out} has columns {header}, expected {fields}; delete or rename it")
    pd.DataFrame([row], columns=fields).to_csv(out, mode="a", header=not out.exists(), index=False)
    print(f"{row.get('dataset', ''):10} {row['method']:11} depth={row['depth']} rep={row['repeat']} "
          f"{row.get('status', ''):17} {row.get('seconds', float('nan')):.2f}s")
