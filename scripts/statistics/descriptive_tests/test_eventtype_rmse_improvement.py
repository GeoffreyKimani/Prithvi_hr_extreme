import sys
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from scipy import stats

BASE_ROOT = Path("~/scratch/prithvi_hr_extreme/outputs").expanduser()

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.hrx_var_names import HRX_VAR_NAMES


def load_eventtype_var_rmse(exp_name: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load event_type × variable RMSE for one experiment.

    Returns
    -------
    event_types : np.ndarray of shape (N_types,)
    rmse        : np.ndarray of shape (N_types, N_vars)
    """
    path = BASE_ROOT / exp_name / "eval_test" / "rmse_per_variable_per_event_test.npz"
    data = np.load(path, allow_pickle=True)
    event_types = data["event_types"]          # (N_types,)
    rmse = data["rmse"]                        # (N_types, N_vars)
    return event_types, rmse


def bootstrap_mean_ci(
    diffs: np.ndarray,
    n_boot: int = 10000,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """
    Bootstrap (percentile) CI for the mean of diffs over event types.
    """
    if rng is None:
        rng = np.random.default_rng(seed=42)

    diffs = np.asarray(diffs)
    n = diffs.shape[0]
    boot_means = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        sample_idx = rng.integers(0, n, size=n)
        boot_means[b] = diffs[sample_idx].mean()

    lower = np.percentile(boot_means, 100 * (alpha / 2))
    upper = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return float(lower), float(upper)


def holm_bonferroni(p_vals: pd.Series, alpha: float = 0.05) -> pd.Series:
    """
    Holm–Bonferroni correction for multiple comparisons.

    Parameters
    ----------
    p_vals : pd.Series, index = tests (e.g., variable names)
    alpha  : float

    Returns
    -------
    pd.Series of adjusted p-values, same index as p_vals.
    """
    # Based on standard Holm procedure
    m = len(p_vals)
    sorted_idx = np.argsort(p_vals.values)
    sorted_p = p_vals.values[sorted_idx]

    adjusted = np.empty_like(sorted_p)
    for i, p in enumerate(sorted_p):
        adjusted[i] = min((m - i) * p, 1.0)

    # Enforce monotonicity
    for i in range(1, m):
        adjusted[i] = max(adjusted[i], adjusted[i - 1])

    # Map back to original order
    adj_series = pd.Series(index=p_vals.index, dtype=float)
    adj_series.iloc[sorted_idx] = adjusted
    return adj_series


def test_eventtype_rmse_improvement_for_variant(
    baseline: str,
    variant: str,
    var_indices: list[int],
    var_names: list[str] | None = None,
    n_boot: int = 10000,
) -> pd.DataFrame:
    """
    For a given variant, run paired tests across event types for given variable indices.

    Returns
    -------
    DataFrame with one row per variable index, columns including:
      variable, var_idx, n_types,
      mean_diff, ci_lower, ci_upper,
      t_stat, p_t,
      w_stat, p_w, p_w_adjusted
    """
    ev_base, rmse_base = load_eventtype_var_rmse(baseline)
    ev_var, rmse_var = load_eventtype_var_rmse(variant)

    assert np.array_equal(ev_base, ev_var), "Event types differ between experiments"
    assert rmse_base.shape == rmse_var.shape, "RMSE shapes differ between experiments"

    n_types, n_vars = rmse_base.shape

    if var_names is None:
        var_names = [f"var_{j}" for j in range(n_vars)]
    else:
        assert len(var_names) == n_vars, "var_names length must match number of variables"

    records = []
    p_wilcoxon = {}

    for var_idx in var_indices:
        rmse_b = rmse_base[:, var_idx]
        rmse_v = rmse_var[:, var_idx]
        diffs = rmse_v - rmse_b

        mean_diff = float(diffs.mean())
        ci_lower, ci_upper = bootstrap_mean_ci(diffs, n_boot=n_boot)

        # Paired t-test (two-sided)
        t_stat, p_t = stats.ttest_rel(rmse_v, rmse_b)

        # Wilcoxon signed-rank test (two-sided; you can interpret mainly for < 0)
        # Note: with zero diffs, scipy may warn; you can handle separately if needed.
        w_stat, p_w = stats.wilcoxon(diffs)

        variable_name = var_names[var_idx]
        p_wilcoxon[variable_name] = p_w

        records.append(
            {
                "variable": variable_name,
                "var_idx": var_idx,
                "n_types": n_types,
                "mean_diff": mean_diff,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "t_stat": float(t_stat),
                "p_t": float(p_t),
                "w_stat": float(w_stat),
                "p_w": float(p_w),
            }
        )

    df = pd.DataFrame.from_records(records)

    # Holm–Bonferroni over variables for this variant
    p_series = pd.Series(p_wilcoxon)
    p_adj = holm_bonferroni(p_series)
    df["p_w_adjusted"] = df["variable"].map(p_adj.to_dict())

    # Add a flag for "significant improvement" under Wilcoxon after correction
    # (interpreting the test as two-sided: improvement if mean_diff < 0 and p_adj < 0.05)
    df["significant_improvement"] = (df["mean_diff"] < 0.0) & (df["p_w_adjusted"] < 0.05)

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["unet_prithvi_mse", "unet_prithvi_tail", "unet_prithvi_exloss"],
    )
    parser.add_argument(
        "--var-names",
        nargs="+",
        type=str,
        required=True,
        default="all",
        help="Variable names to test (e.g., rain-related channels).",
    )

    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=10000,
        help="Number of bootstrap samples for mean diff CI.",
    )
    args = parser.parse_args()

    VAR_NAME_TO_IDX = {name: i for i, name in enumerate(HRX_VAR_NAMES)}

    if "all" in args.var_names:
        selected_indices = list(range(len(HRX_VAR_NAMES)))
    else:
        selected_indices = [VAR_NAME_TO_IDX[name] for name in args.var_names]
    
    

    out_dir = BASE_ROOT / "hypothesis_tests"
    out_dir.mkdir(parents=True, exist_ok=True)

    for variant in args.variants:
        df_results = test_eventtype_rmse_improvement_for_variant(
            baseline=args.baseline,
            variant=variant,
            var_indices=selected_indices,
            var_names=HRX_VAR_NAMES,
            n_boot=args.n_bootstrap,
        )

        out_path = out_dir / f"eventtype_rmse_tests_{args.baseline}_vs_{variant}.csv"
        df_results.to_csv(out_path, index=False)
        print(f"Saved hypothesis test results for {args.baseline} vs {variant} to {out_path}")


if __name__ == "__main__":
    main()