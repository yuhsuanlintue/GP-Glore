# GP-Glore

Code for *Extending GP-Growth to Logistic Regression for Exhaustive EMM* (Group 11, 2AMM20).

## Setup

Install [uv](https://docs.astral.sh/uv/), then:

```
uv sync
```

## Experiments

Settings are at the top of each script; results go to `results/`.

Comparison with depth-first search (Sec. 4.3.2 a):

```
uv run python -m experiments.sec4_3_2_a
```

Comparison of model classes (Sec. 4.3.2 b):

```
uv run python -m experiments.sec4_3_2_b
```
