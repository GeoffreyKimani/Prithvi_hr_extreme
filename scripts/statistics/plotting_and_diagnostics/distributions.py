import numpy as np
import argparse
from pathlib import Path
from scipy import stats
import matplotlib.pyplot as plt
from test_primary_hypothesis_per_sample import load_per_sample_rmse, align_by_sample_indices

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

    BASE_ROOT = Path("~/scratch/prithvi_hr_extreme/outputs").expanduser()
    out_dir = BASE_ROOT / "hypothesis_tests_primary"
    out_dir.mkdir(parents=True, exist_ok=True)

    base_idx, base_event_types, base_rmse, _ = load_per_sample_rmse(args.baseline)
    var_idx, var_event_types, var_rmse, _ = load_per_sample_rmse(args.variants[0])  # Assuming we're testing the first variant

    sample_idx, event_types, base_rmse, var_rmse = align_by_sample_indices(
        base_idx, base_event_types, base_rmse,
        var_idx, var_event_types, var_rmse
    )

    loss_name = str(args.variants[0]).split("_")[-1]  # Extract loss names from variant name
    print(f"Testing variant: {args.variants[0]} with losses: {loss_name}")
    print(f"Number of aligned samples: {len(sample_idx)}")

    # Compute paired differences
    d = var_rmse - base_rmse  # shape: (n_samples, n_vars)
    print(f"Len of paired differences: {len(d)}, shape: {d.shape}")

    d = d.mean(axis=1)  # shape: (n_events,)
    print(f"Mean of paired differences (averaged over variables): {d.mean():.4f}, Std: {d.std():.4f}")

    # --- Normality tests ---
    stat_sw, p_sw = stats.shapiro(d)
    print(f"Shapiro-Wilk:         W={stat_sw:.4f}, p={p_sw:.4e}")

    stat_dp, p_dp = stats.normaltest(d)
    print(f"D'Agostino-Pearson:   K²={stat_dp:.4f}, p={p_dp:.4e}")

    # Interpretation
    alpha = 0.05
    for name, p in [("Shapiro-Wilk", p_sw), ("D'Agostino-Pearson", p_dp)]:
        if p < alpha:
            print(f"  → {name}: REJECT normality (p={p:.4e} < {alpha}) → Wilcoxon is warranted")
        else:
            print(f"  → {name}: Cannot reject normality (p={p:.4e}) → t-test assumption holds")

    # --- Q-Q Plot ---
    fig, ax = plt.subplots(figsize=(5, 5))
    stats.probplot(d, dist="norm", plot=ax)

    # Override scipy's default labels
    ax.set_xlabel("Theoretical Normal Quantiles", fontsize=12)
    ax.set_ylabel("Sample Quantiles of Paired Differences (RMSE)", fontsize=12)
    ax.set_title(f"Q-Q Plot of Paired Differences\n(Prithvi − Baseline {loss_name})", fontsize=11)

    plt.tight_layout()
    plt.savefig(f"{out_dir}/qq_plot_paired_differences_{loss_name}.png", dpi=200)


if __name__ == "__main__":
    main()