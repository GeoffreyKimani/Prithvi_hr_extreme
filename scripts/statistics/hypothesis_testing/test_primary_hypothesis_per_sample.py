"""
    Overview
    --------
    Run primary hypothesis tests comparing each Prithvi variant to the baseline.
    This script loads per-sample RMSE data for the baseline and each variant, aligns them by sample indices,
    and performs paired tests for each variable. Results are saved as CSV files for each variant comparison.

    Hypotheses Tested
    ---------------
    For each variable, we test:
    - H0: The prithvi variant has the same or worse RMSE than the baseline (mean difference >= 0)
    - H1: The prithvi variant has better RMSE than the baseline (mean difference < 0)

    Technical Details
    -----------------
    - Loads per-sample RMSE arrays from .npz files for the baseline and variant
    - Aligns samples by their indices to ensure proper pairing
    - For each variable:
        - Computes mean and median differences in RMSE
        - Computes bootstrap confidence intervals for the mean difference
        - Performs one-sided paired t-test and Wilcoxon signed-rank test (variant < baseline)
        - Calculates relative improvement percentage
    - Applies Holm-Bonferroni correction for multiple testing across variables
    - Saves results in a structured CSV format for downstream analysis and plotting

    Usage
    -----
    Run the script with optional arguments to specify baseline and variants:
    ```
    python test_primary_hypothesis_per_sample.py \
        --baseline unet_plain_mse \
        --variants unet_prithvi_mse unet_prithvi_tail unet_prithvi_exloss \
        --n-bootstrap 10000
    ```
"""

from pathlib import Path
import argparse
import sys
import numpy as np
import pandas as pd
from scipy import stats

BASE_ROOT = Path("~/scratch/prithvi_hr_extreme/outputs").expanduser()

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.hrx_var_names import HRX_VAR_NAMES


def load_per_sample_rmse(exp_name: str):
    """
    Load per-sample RMSE arrays for one experiment.

    Returns
    -------
    sample_indices : np.ndarray, shape (N,)
    event_types    : np.ndarray, shape (N,)
    rmse           : np.ndarray, shape (N, C)
    mean_rmse_all  : np.ndarray, shape (N,)
    """
    path = BASE_ROOT / exp_name / "eval_test" / "rmse_per_variable_per_sample_test.npz"
    data = np.load(path, allow_pickle=True)

    sample_indices = data["sample_indices"]
    event_types = data["event_types"]
    rmse = data["rmse"]
    mean_rmse_all = data["mean_rmse_all_vars"]

    return sample_indices, event_types, rmse, mean_rmse_all


def bootstrap_mean_ci(diffs, n_boot=10000, alpha=0.05, seed=42):
    rng = np.random.default_rng(seed)
    diffs = np.asarray(diffs, dtype=float)
    n = len(diffs)

    boot_means = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_means[i] = diffs[idx].mean()

    lower = np.percentile(boot_means, 100 * (alpha / 2))
    upper = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return float(lower), float(upper)


def holm_bonferroni(pvals: pd.Series) -> pd.Series:
    """
    Holm-Bonferroni adjusted p-values.
    """
    pvals = pvals.astype(float)
    m = len(pvals)

    order = np.argsort(pvals.values)
    sorted_p = pvals.values[order]

    adjusted = np.empty_like(sorted_p)
    for i, p in enumerate(sorted_p):
        adjusted[i] = min((m - i) * p, 1.0)

    for i in range(1, m):
        adjusted[i] = max(adjusted[i], adjusted[i - 1])

    result = pd.Series(index=pvals.index, dtype=float)
    result.iloc[order] = adjusted
    return result


def align_by_sample_indices(base_idx, base_event_types, base_rmse, var_idx, var_event_types, var_rmse):
    """
    Ensure both experiments are aligned on sample_indices.
    """
    df_base = pd.DataFrame({
        "sample_idx": base_idx,
        "event_type_base": base_event_types,
    })
    df_var = pd.DataFrame({
        "sample_idx": var_idx,
        "event_type_var": var_event_types,
    })

    merged = df_base.merge(df_var, on="sample_idx", how="inner")
    merged = merged.sort_values("sample_idx").reset_index(drop=True)

    base_order = merged["sample_idx"].map({k: i for i, k in enumerate(base_idx)}).to_numpy()
    var_order = merged["sample_idx"].map({k: i for i, k in enumerate(var_idx)}).to_numpy()

    aligned_base_event_types = base_event_types[base_order]
    aligned_var_event_types = var_event_types[var_order]

    if not np.array_equal(aligned_base_event_types, aligned_var_event_types):
        raise ValueError("Event types do not align after matching sample indices.")

    aligned_base_rmse = base_rmse[base_order]
    aligned_var_rmse = var_rmse[var_order]

    return merged["sample_idx"].to_numpy(), aligned_base_event_types, aligned_base_rmse, aligned_var_rmse


def test_variant_against_baseline(baseline: str, variant: str, n_boot=10000):
    base_idx, base_event_types, base_rmse, _ = load_per_sample_rmse(baseline)
    var_idx, var_event_types, var_rmse, _ = load_per_sample_rmse(variant)

    sample_idx, event_types, base_rmse, var_rmse = align_by_sample_indices(
        base_idx, base_event_types, base_rmse,
        var_idx, var_event_types, var_rmse
    )

    n_samples, n_vars = base_rmse.shape
    assert n_vars == len(HRX_VAR_NAMES), "Variable count does not match HRX_VAR_NAMES."

    rows = []
    wilcoxon_pvals = {}

    for j, var_name in enumerate(HRX_VAR_NAMES):
        x = var_rmse[:, j]   # Prithvi
        y = base_rmse[:, j]  # Baseline
        diffs = x - y

        mean_diff = float(diffs.mean())
        median_diff = float(np.median(diffs))
        ci_lower, ci_upper = bootstrap_mean_ci(diffs, n_boot=n_boot)

        # One-sided paired t-test: variant < baseline
        t_res = stats.ttest_rel(x, y, alternative="less")
        t_stat = float(t_res.statistic)
        p_t = float(t_res.pvalue)

        # One-sided Wilcoxon: variant < baseline
        try:
            w_res = stats.wilcoxon(x, y, alternative="less", zero_method="wilcox")
            w_stat = float(w_res.statistic)
            p_w = float(w_res.pvalue)
        except ValueError:
            # Can happen if all differences are zero
            w_stat = np.nan
            p_w = 1.0

        rel_improvement_pct = float(100.0 * (y.mean() - x.mean()) / y.mean()) if y.mean() != 0 else np.nan

        wilcoxon_pvals[var_name] = p_w

        rows.append({
            "variant": variant,
            "variable": var_name,
            "n_samples": n_samples,
            "mean_rmse_baseline": float(y.mean()),
            "mean_rmse_variant": float(x.mean()),
            "mean_diff": mean_diff,
            "median_diff": median_diff,
            "rel_improvement_pct": rel_improvement_pct,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "t_stat_less": t_stat,
            "p_t_less": p_t,
            "w_stat_less": w_stat,
            "p_w_less": p_w,
        })

    df = pd.DataFrame(rows)

    # Holm-Bonferroni correction across variables for this variant
    p_series = pd.Series(wilcoxon_pvals)
    p_adj = holm_bonferroni(p_series)
    df["p_w_less_holm"] = df["variable"].map(p_adj.to_dict())

    df["significant_improvement"] = (
        (df["mean_diff"] < 0) &
        (df["p_w_less_holm"] < 0.05)
    )

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", default="unet_plain_mse")
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["unet_prithvi_mse", "unet_prithvi_tail", "unet_prithvi_exloss"],
    )
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    args = parser.parse_args()

    out_dir = BASE_ROOT / "hypothesis_tests_primary"
    out_dir.mkdir(parents=True, exist_ok=True)

    for variant in args.variants:
        df = test_variant_against_baseline(
            baseline=args.baseline,
            variant=variant,
            n_boot=args.n_bootstrap,
        )
        out_path = out_dir / f"primary_hypothesis_{args.baseline}_vs_{variant}.csv"
        df.to_csv(out_path, index=False)
        print(f"Saved primary hypothesis test results to {out_path}")


if __name__ == "__main__":
    main()