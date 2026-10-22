"Datasets preperation for Logistic Regression Experiments"
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import pysubgroup as ps
from pysubgroup.datasets import get_credit_data
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler

Y = "y"
DATA_HOME = Path(__file__).resolve().parent.parent / "data"

def _openml(name: str, version: int) -> pd.DataFrame:
    try:
        df = fetch_openml(name, version=version, as_frame=True, data_home=DATA_HOME).frame
    except ValueError:
        bunch = fetch_openml(name, version=version, as_frame=False, data_home=DATA_HOME)
        X = bunch.data.toarray() if hasattr(bunch.data, "toarray") else bunch.data
        df = pd.DataFrame(X, columns=bunch.feature_names)
        if bunch.target is not None:
            df["target"] = bunch.target

    cats = df.select_dtypes("category").columns
    df[cats] = df[cats].astype(object)
    return df


def _heart_h_positive(s: pd.Series) -> pd.Series:
    str_vals = s.astype(str).str.strip().str.lower()
    # Check string patterns like '>0', '1', '2', '3', '4', 'present', 'positive', 'sick', etc.
    is_neg = str_vals.isin(["0", "0.0", "<50", "0: <50", "negative", "absent", "buff", "false", "no", "n"])
    numeric_vals = pd.to_numeric(str_vals, errors="coerce")
    
    is_pos_num = numeric_vals.notna() & (numeric_vals > 0)
    is_pos_str = (~is_neg) & (str_vals != "") & (str_vals != "nan") & (str_vals != "?")
    
    return is_pos_num | is_pos_str


@dataclass(frozen=True)
class DatasetSpec:
    load: Callable[[], pd.DataFrame]
    target: str
    positive_fn: Callable[[pd.Series], pd.Series]
    x_names: tuple[str, ...]


SPECS: dict[str, DatasetSpec] = {
    "adults": DatasetSpec(
        partial(_openml, "adult", 2),
        "class",
        lambda s: s.astype(str).str.strip().str.lower() == ">50k",
        ("age", "fnlwgt", "education_num", "hours_per_week"),
    ),
    "credit-g": DatasetSpec(
        get_credit_data,
        "class",
        lambda s: s.astype(str).str.strip().str.lower() == "bad",
        ("duration", "credit_amount", "installment_commitment", "residence_since"),
    ),
    "credit-a": DatasetSpec(
        partial(_openml, "Australian", 1),
        "class",
        lambda s: s.astype(str).str.strip() == "1",
        ("a2", "a3", "a7", "a10"),
    ),
    "diabetes": DatasetSpec(
        partial(_openml, "diabetes", 1),
        "class",
        lambda s: s.astype(str).str.strip().str.lower() == "tested_positive",
        ("plas", "pres", "skin", "mass"),
    ),
    "breast-w": DatasetSpec(
        partial(_openml, "breast-w", 1),
        "class",
        lambda s: s.astype(str).str.strip().str.lower() == "malignant",
        ("clump_thickness", "cell_size", "cell_shape", "marginal_adhesion"),
    ),
    "heart-h": DatasetSpec(
        partial(_openml, "heart-h", 1),
        "num",
        _heart_h_positive,
        ("age", "trestbps", "chol", "thalach"),
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
    spec = SPECS[name]
    df = _decode_bytes(spec.load())

    df.columns = [str(c).strip().lower().replace("-", "_") for c in df.columns]

    if name == "credit-a" and "a2" not in df.columns:
        col_map = {c: f"a{i}" for i, c in enumerate(df.columns[:-1], 1)}
        col_map[df.columns[-1]] = "class"
        df = df.rename(columns=col_map)

    if name == "breast-w":
        col_renames = {}
        for col in df.columns:
            if "clump" in col:
                col_renames[col] = "clump_thickness"
            elif "size" in col and "cell" in col and "shape" not in col:
                col_renames[col] = "cell_size"
            elif "shape" in col and "cell" in col:
                col_renames[col] = "cell_shape"
            elif "adhesion" in col:
                col_renames[col] = "marginal_adhesion"
        df = df.rename(columns=col_renames)

    cols = list(df.columns)
    counts = {}
    new_cols = []
    for c in cols:
        if c in counts:
            counts[c] += 1
            new_cols.append(f"{c}_{counts[c]}")
        else:
            counts[c] = 0
            new_cols.append(c)
    df.columns = new_cols

    target_col = spec.target.lower().replace("-", "_")
    if target_col not in df.columns:
        candidates = [c for c in df.columns if c in ["class", "num", "target", "diagnosis", "hd"]]
        target_col = candidates[0] if candidates else df.columns[-1]

    x_names = [x.lower().replace("-", "_") for x in spec.x_names]

    target_series = df.pop(target_col)
    df[Y] = spec.positive_fn(target_series).astype(int)

    for col in x_names:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if name in ["heart-h", "diabetes"]:
            df[col] = df[col].replace(0, np.nan)

    df = df.dropna(subset=[Y, *x_names]).reset_index(drop=True)

    scaler = StandardScaler()
    df[x_names] = scaler.fit_transform(df[x_names])

    selectors = ps.create_selectors(df, nbins=5, ignore=[Y, *x_names])

    if frac < 1.0:
        rng = np.random.default_rng(seed)
        keep = np.sort(rng.choice(len(df), size=round(frac * len(df)), replace=False))
        df = df.iloc[keep].reset_index(drop=True)

    return Dataset(name, df, x_names, selectors)


if __name__ == "__main__":
    for name in SPECS:
        print(load_dataset(name))