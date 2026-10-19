# Notes

## Selectors — `experiments/datasets.py`

```python
selectors = ps.create_selectors(df, nbins=5, ignore=[Y, *x_names])
```

Builds the search space: the yes/no conditions that get combined (with AND, up to
`depth` of them) into subgroup descriptions like `purpose=='used car' AND job=='skilled'`.

- **Which columns**: every column except `y` and the X columns. Model attributes are kept
  out of the descriptions, as in Lemmerich et al. (2012).
- **Nominal column**: one selector per value, e.g. `purpose=='radio/tv'`.
- **Numeric column**: 5 equal-frequency intervals, e.g. `residence_since<2.0`.
- **Built before subsampling**, so every subsample in the scalability runs uses the same
  selectors and only the number of rows changes.

## Dropping `education` from adults

`education` is the same information as the model attribute `education-num` (each value
maps to one number, e.g. `HS-grad` = 9). As a selector it would make `education-num`
constant inside the subgroup, so it can't be told apart from the intercept: the Hessian
is singular and the logistic regression can't be fitted. It's the same rule as keeping
X out of the descriptions, for a column that is X under another name.
