from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

BASE_ROOT = Path("~/scratch/prithvi_hr_extreme/outputs").expanduser()

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.hrx_var_names import HRX_VAR_NAMES

# Define your rain proxy indices exactly as in your config
RAIN_PROXY_INDICES = {
    "2t": HRX_VAR_NAMES.index("2t"),
    "msl": HRX_VAR_NAMES.index("msl"),
    "q_1000": HRX_VAR_NAMES.index("q 100000"),
    "q_850": HRX_VAR_NAMES.index("q 85000"),
    "q_700": HRX_VAR_NAMES.index("q 70000"),
    "u_850": HRX_VAR_NAMES.index("u 85000"),
    "v_850": HRX_VAR_NAMES.index("v 85000"),
    "hgt_500": HRX_VAR_NAMES.index("hgtn 50000"),
}


def load_per_sample_rmse(exp_name: str):
    path = BASE_ROOT / exp_name / "eval_test" / "rmse_per_variable_per_sample_test.npz"
    data = np.load(path, allow_pickle=True)
    return data["sample_indices"], data["event_types"], data["rmse"]  # (N,), (N,), (N, C)


def align_rmse(baseline: str, variant: str):
    base_idx, base_et, base_rmse = load_per_sample_rmse(baseline)
    var_idx, var_et, var_rmse = load_per_sample_rmse(variant)

    # Align by sample index
    df_base = pd.DataFrame({"sample_idx": base_idx, "i_base": np.arange(len(base_idx))})
    df_var = pd.DataFrame({"sample_idx": var_idx, "i_var": np.arange(len(var_idx))})
    merged = df_base.merge(df_var, on="sample_idx", how="inner").sort_values("sample_idx")

    bi = merged["i_base"].to_numpy()
    vi = merged["i_var"].to_numpy()

    base_rmse_aligned = base_rmse[bi]
    var_rmse_aligned = var_rmse[vi]
    et_aligned = base_et[bi]

    return et_aligned, base_rmse_aligned, var_rmse_aligned


def plot_diagnostics_for_proxy(
    proxy_name: str,
    var_idx: int,
    event_types,
    rmse_base,
    rmse_var,
    out_dir: Path,
):
    x = rmse_var[:, var_idx]
    y = rmse_base[:, var_idx]
    diffs = y - x  # baseline - variant, so positive diffs mean Prithvi wins

    # Shapiro–Wilk normality test (advisory)
    sh_stat, sh_p = stats.shapiro(diffs)
    print(f"{proxy_name}: Shapiro-Wilk W={sh_stat:.3f}, p={sh_p:.3g}")

    stat, p = stats.wilcoxon(diffs, alternative='greater')  # H1: baseline > variant (Prithvi wins)
    print(f"{proxy_name}: Wilcoxon stat={stat:.1f}, p={p:.3g}")

    out_dir.mkdir(parents=True, exist_ok=True)

    # Histogram
    plt.figure(figsize=(6, 4))
    plt.hist(diffs, bins=15, edgecolor="k")
    plt.axvline(0, color="red", linestyle="--", label="0")
    plt.title(f"Paired differences (baseline − Prithvi) for {proxy_name}")
    plt.xlabel("RMSE difference")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"hist_diffs_{proxy_name}.png", dpi=300)
    plt.close()

    # Q–Q plot against normal
    plt.figure(figsize=(6, 4))
    stats.probplot(diffs, dist="norm", plot=plt)
    plt.title(f"Q-Q plot of diffs for {proxy_name}")
    plt.tight_layout()
    plt.savefig(out_dir / f"qq_diffs_{proxy_name}.png", dpi=300)
    plt.close()


def main():
    # or mse / tail
    baseline = "unet_plain_exloss"
    variant = "unet_prithvi_exloss"
    out_dir = BASE_ROOT / "diagnostics_rain_proxies" / f"{baseline}_vs_{variant}"

    event_types, rmse_base, rmse_var = align_rmse(baseline, variant)

    for proxy in RAIN_PROXY_INDICES.keys():
        idx = RAIN_PROXY_INDICES[proxy]
        plot_diagnostics_for_proxy(proxy, idx, event_types, rmse_base, rmse_var, out_dir)


if __name__ == "__main__":
    main()