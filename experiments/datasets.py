"""
Dataset loading

Usage:
    ds = load_dataset("credit-g")
    ds.data, ds.x_names, ds.selectors
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pysubgroup as ps
from pysubgroup.datasets import get_credit_data
from sklearn.datasets import fetch_openml

Y = "y"
DATA_HOME = Path(__file__).resolve().parent.parent / "data"


def _openml(data_id: int) -> pd.DataFrame:
    """This is generate with LLM
    Download (first time only, cached in data/) and return the full table."""
    df = fetch_openml(data_id=data_id, as_frame=True, data_home=DATA_HOME).frame
    cats = df.select_dtypes("category").columns
    df[cats] = df[cats].astype(object) 
    return df


@dataclass(frozen=True)
class DatasetSpec:
    openml_id: int | None  # None = credit-g that ships with pysubgroup
    target: str  # column we predict
    positive: str  # the target value that becomes y = 1
    x_names: tuple[str, ...]  # numeric columns used as X in the logistic regression
    drop: tuple[str, ...] = ()  # columns left out of the subgroup descriptions


SPECS: dict[str, DatasetSpec] = {
    "adults": DatasetSpec(
        1590,
        "class", ">50K", ("age", "education-num", "hours-per-week", "capital-gain"),
        drop=("education",),  # same information as education-num (see note.md)
    ),
    "credit-g": DatasetSpec(
        None,  # ships with pysubgroup
        "class", "bad", ("duration", "credit_amount", "age", "installment_commitment"),
    ),
    "credit-a": DatasetSpec(
        29,
        "class", "+", ("A2", "A3", "A8", "A11"),
    ),
    "heart-h": DatasetSpec(
        51,
        "num", ">50_1", ("age", "trestbps", "chol", "thalach"),
    ),
    "diabetes": DatasetSpec(
        37,
        "class", "tested_positive", ("plas", "mass", "age", "pedi"),
    ),
    "breast-w": DatasetSpec(
        15,
        "Class", "malignant",
        ("Clump_Thickness", "Cell_Size_Uniformity", "Marginal_Adhesion", "Bland_Chromatin"),
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
    df = get_credit_data() if spec.openml_id is None else _openml(spec.openml_id)
    df = _decode_bytes(df)

    df[Y] = (df.pop(spec.target) == spec.positive).astype(int)
    x_names = list(spec.x_names)
    df = df.drop(columns=list(spec.drop)).dropna(subset=x_names).reset_index(drop=True)

    # standardization
    X = df[x_names].astype(float)
    df[x_names] = (X - X.mean()) / X.std()

    # see note.md
    selectors = ps.create_selectors(df, nbins=5, ignore=[Y, *x_names])

    if frac < 1.0:
        rng = np.random.default_rng(seed)
        keep = np.sort(rng.choice(len(df), size=round(frac * len(df)), replace=False))
        df = df.iloc[keep].reset_index(drop=True)

    return Dataset(name, df, x_names, selectors)


if __name__ == "__main__":
    for name in SPECS:
        print(load_dataset(name))
