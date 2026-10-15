"""
Dataset loading

Usage:
    ds = load_dataset("credit-g")
    ds.data, ds.x_names, ds.selectors
"""

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import pysubgroup as ps
from pysubgroup.datasets import get_credit_data
from sklearn.datasets import fetch_openml

Y = "y"
DATA_HOME = Path(__file__).resolve().parent.parent / "data"


def _openml(name: str, version: int) -> pd.DataFrame:
    """This is generate with LLM
    Download (first time only, cached in data/) and return the full table."""
    df = fetch_openml(name, version=version, as_frame=True, data_home=DATA_HOME).frame
    cats = df.select_dtypes("category").columns
    df[cats] = df[cats].astype(object) 
    return df


@dataclass(frozen=True)
class DatasetSpec:
    load: Callable[[], pd.DataFrame]  # returns the raw table
    target: str  # column we predict
    positive: str  # the target value that becomes y = 1
    x_names: tuple[str, ...]  # numeric columns used as X in the logistic regression


SPECS: dict[str, DatasetSpec] = {
    "credit-g": DatasetSpec(
        get_credit_data,  # ships with pysubgroup, nothing to download
        "class", "bad", ("duration", "credit_amount"),
    ),
    "adults": DatasetSpec(
        partial(_openml, "adult", 2),
        "class", ">50K", ("age", "hours-per-week"),
    ),
    "diabetes": DatasetSpec(
        partial(_openml, "diabetes", 1),
        "class", "tested_positive", ("plas", "mass"),
    ),
    "spambase": DatasetSpec(
        partial(_openml, "spambase", 1),
        "class", "1", ("char_freq_%21", "capital_run_length_average"),
    ),
}


@dataclass
class Dataset:
    name: str
    data: pd.DataFrame
    x_names: list[str]
    selectors: list[ps.SelectorBase]

    def __repr__(self) -> str:
        return (f"Dataset({self.name!r}, n={len(self.data)}, x={self.x_names}, "
                f"{len(self.selectors)} selectors, positives={self.data[Y].mean():.1%})")


def _decode_bytes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns[df.dtypes == object]:
        df[col] = df[col].map(lambda v: v.decode() if isinstance(v, bytes) else v)
    return df


def load_dataset(name: str, frac: float = 1.0, seed: int = 0) -> Dataset:
    """Load a dataset and turn the target into a 0/1 column `y`.
    get subsample with frac < 1.
    """
    spec = SPECS[name]
    df = _decode_bytes(spec.load())

    df[Y] = (df.pop(spec.target) == spec.positive).astype(int)
    x_names = list(spec.x_names)
    df = df.dropna(subset=x_names).reset_index(drop=True)

    # every other column is used for subgroup descriptions; numbers get 5 equal-frequency bins
    selectors = ps.create_selectors(df, nbins=5, ignore=[Y, *x_names])

    if frac < 1.0:
        rng = np.random.default_rng(seed)
        keep = np.sort(rng.choice(len(df), size=round(frac * len(df)), replace=False))
        df = df.iloc[keep].reset_index(drop=True)

    return Dataset(name, df, x_names, selectors)


if __name__ == "__main__":
    for name in SPECS:
        print(load_dataset(name))
