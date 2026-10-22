"Logistic Regression experiments"

from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from experiments.datasets_glore import SPECS, load_dataset, Y

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)
OUT_CSV = RESULTS_DIR / "sec5_3_2_outcomes.csv"

def fit_global_lr(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    n, p = X.shape
    X_design = np.hstack([np.ones((n, 1)), X])

    def loss_and_grad(w):
        z = X_design @ w
        p_hat = 1.0 / (1.0 + np.exp(-z))
        eps = 1e-15
        p_hat = np.clip(p_hat, eps, 1.0 - eps)
        loss = -np.sum(y * np.log(p_hat) + (1.0 - y) * np.log(1.0 - p_hat))
        grad = X_design.T @ (p_hat - y)
        return loss, grad

    w0 = np.zeros(p + 1)
    res = minimize(loss_and_grad, w0, jac=True, method="L-BFGS-B")
    if not res.success:
        raise RuntimeError("Global LR fit did not converge.")
    return res.x


def evaluate_subgroup_lr(X_sub: np.ndarray, y_sub: np.ndarray, b_global: np.ndarray):
    n_sub, p = X_sub.shape
    if n_sub < 5:
        return None, None, 0, False, True

    X_design = np.hstack([np.ones((n_sub, 1)), X_sub])

    z_global = X_design @ b_global
    pi_global = 1.0 / (1.0 + np.exp(-z_global))
    eps = 1e-15
    pi_global = np.clip(pi_global, eps, 1.0 - eps)

    grad = X_design.T @ (y_sub - pi_global)
    weights = pi_global * (1.0 - pi_global)
    H = X_design.T @ (weights[:, None] * X_design)

    try:
        delta = np.linalg.solve(H, grad)
        b_approx = b_global + delta
        singular = False
    except np.linalg.LinAlgError:
        b_approx = b_global
        singular = True

    b_curr = b_global.copy()
    nr_iters = 0
    max_iters = 25
    tol = 1e-6
    converged = False

    for it in range(1, max_iters + 1):
        nr_iters = it
        z = X_design @ b_curr
        pi = 1.0 / (1.0 + np.exp(-z))
        pi = np.clip(pi, eps, 1.0 - eps)

        g_iter = X_design.T @ (y_sub - pi)
        w_iter = pi * (1.0 - pi)
        H_iter = X_design.T @ (w_iter[:, None] * X_design)

        try:
            step = np.linalg.solve(H_iter, g_iter)
        except np.linalg.LinAlgError:
            break

        b_curr += step
        if np.max(np.abs(step)) < tol:
            converged = True
            break

    separated = not converged
    b_conv = b_curr if converged else b_approx

    return b_approx, b_conv, nr_iters, separated, singular


def get_subgroup_mask(df, selector_combo):
    mask = np.ones(len(df), dtype=bool)
    for sel in selector_combo:
        mask &= sel.covers(df)
    return mask


def run_sec5_3_2(depths=(1, 2, 3), top_k=10):
    records = []

    for depth in depths:
        for name in SPECS:
            print(f"\n---> Evaluating Section 5.3.2 LR Outcomes: {name} (depth={depth}, k={top_k})")
            ds = load_dataset(name)

            X = ds.data[ds.x_names].values
            y = ds.data[Y].values

            try:
                b_global = fit_global_lr(X, y)
            except Exception as e:
                print(f"     Error on {name}: {e}")
                continue

            subgroup_mads = []
            nr_iters_list = []
            separated_count = 0
            singular_count = 0
            evaluated_count = 0

            base_selectors = ds.selectors[:min(20, len(ds.selectors))]
            candidate_combos = list(combinations(base_selectors, depth))

            for combo in candidate_combos:
                mask = get_subgroup_mask(ds.data, combo)
                if mask.sum() < 30:
                    continue

                X_sub = X[mask]
                y_sub = y[mask]

                b_approx, b_conv, iters, sep, sing = evaluate_subgroup_lr(X_sub, y_sub, b_global)
                if b_approx is None:
                    continue

                evaluated_count += 1
                if sing:
                    singular_count += 1
                if sep:
                    separated_count += 1
                else:
                    subgroup_mads.append(np.mean(np.abs(b_approx - b_conv)))

                nr_iters_list.append(iters)

                if evaluated_count >= top_k:
                    break

            if evaluated_count == 0:
                continue

            overall_mad = np.mean(subgroup_mads) if subgroup_mads else np.nan
            avg_nr_iters = np.mean(nr_iters_list) if nr_iters_list else np.nan
            sep_rate = (separated_count / evaluated_count) * 100.0
            sing_rate = (singular_count / evaluated_count) * 100.0

            records.append({
                "depth": depth,
                "dataset": name,
                "n_attributes": len(ds.x_names),
                "top_k": top_k,
                "evaluated_subgroups": evaluated_count,
                "overall_mad": overall_mad,
                "avg_nr_iters": avg_nr_iters,
                "separation_rate_pct": sep_rate,
                "singular_rate_pct": sing_rate,
            })

    df_res = pd.DataFrame(records)
    print("\n========================================================")
    print(" SECTION 5.3.2 TEST RESULTS Across Search Depths d in {1, 2, 3}")
    print("========================================================")
    print(df_res.to_string(index=False))

    df_res.to_csv(OUT_CSV, index=False)
    print(f"\nSaved test run results to: {OUT_CSV}")


if __name__ == "__main__":
    run_sec5_3_2()