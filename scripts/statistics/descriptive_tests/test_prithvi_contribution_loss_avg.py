import sys
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path

BASE_ROOT = Path("~/scratch/prithvi_hr_extreme/outputs").expanduser()

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.hrx_var_names import HRX_VAR_NAMES

# ─────────────────────────────────────────────────────────────
# STAGE 0: Load per-event, per-variable RMSE arrays
# Shape: (n_events, n_vars) for each model/loss combination
# Replace these with your actual file loading logic
# ─────────────────────────────────────────────────────────────

def load_rmse(npz_path: str, key: str = "rmse") -> np.ndarray:
    """Load RMSE array from NPZ. Shape: (n_events, n_vars)."""
    data = np.load(npz_path)
    return data[key]   # shape (n_events, n_vars)

# --- MSE loss family ---
rmse_plain_mse   = load_rmse(f"{BASE_ROOT}/unet_plain_mse/eval_test/rmse_per_variable_per_sample_test.npz")        # baseline matched to mse loss
rmse_prithvi_mse = load_rmse(f"{BASE_ROOT}/unet_prithvi_mse/eval_test/rmse_per_variable_per_sample_test.npz")

# --- Tail loss family ---
rmse_plain_tail   = load_rmse(f"{BASE_ROOT}/unet_plain_tail/eval_test/rmse_per_variable_per_sample_test.npz")
rmse_prithvi_tail = load_rmse(f"{BASE_ROOT}/unet_prithvi_tail/eval_test/rmse_per_variable_per_sample_test.npz")

# --- Extreme loss family ---
rmse_plain_exloss   = load_rmse(f"{BASE_ROOT}/unet_plain_exloss/eval_test/rmse_per_variable_per_sample_test.npz")
rmse_prithvi_exloss = load_rmse(f"{BASE_ROOT}/unet_prithvi_exloss/eval_test/rmse_per_variable_per_sample_test.npz")

# ─────────────────────────────────────────────────────────────
# STAGE 1: Compute matched paired differences per loss family
# d[i, j] = RMSE_Prithvi[i,j] - RMSE_baseline[i,j]
# Negative d means Prithvi is better for that sample/variable
# ─────────────────────────────────────────────────────────────

d_mse    = rmse_prithvi_mse    - rmse_plain_mse      # (n_events, n_vars)
d_tail   = rmse_prithvi_tail   - rmse_plain_tail      # (n_events, n_vars)
d_exloss = rmse_prithvi_exloss - rmse_plain_exloss    # (n_events, n_vars)

# ─────────────────────────────────────────────────────────────
# STAGE 2: Average across loss families → loss-robust effect
# d_bar[i, j] = mean of the three matched differences
# ─────────────────────────────────────────────────────────────

d_bar = (d_mse + d_tail + d_exloss) / 3.0   # (n_events, n_vars)

# ─────────────────────────────────────────────────────────────
# STAGE 3: Test H0: d_bar >= 0 per variable
# i.e., test whether the across-loss Prithvi effect is < 0
# ─────────────────────────────────────────────────────────────

n_vars = d_bar.shape[1]
assert len(HRX_VAR_NAMES) == n_vars

results = []

for j, var_name in enumerate(HRX_VAR_NAMES):
    d_j = d_bar[:, j]   # shape (n_events,)

    # Descriptive stats
    mean_d = float(np.mean(d_j))
    median_d = float(np.median(d_j))
    std_d = float(np.std(d_j, ddof=1))

    # One-sided paired t-test: H1: mean < 0 (Prithvi better)
    t_stat, p_ttest = stats.ttest_1samp(d_j, popmean=0, alternative='less')

    # One-sided Wilcoxon signed-rank: H1: median < 0
    try:
        w_stat, p_wilcox = stats.wilcoxon(d_j, alternative='less', zero_method='wilcox')
    except ValueError:
        # All differences are zero — no test possible
        w_stat, p_wilcox = np.nan, np.nan

    results.append({
        "variable":   var_name,
        "mean_d_bar": mean_d,
        "median_d_bar": median_d,
        "std_d_bar":  std_d,
        "t_stat":     float(t_stat),
        "p_ttest":    float(p_ttest),
        "w_stat":     float(w_stat) if not np.isnan(w_stat) else np.nan,
        "p_wilcox":   float(p_wilcox) if not np.isnan(p_wilcox) else np.nan,
    })

# ─────────────────────────────────────────────────────────────
# STAGE 4: Apply Benjamini-Hochberg FDR correction
# Corrects for 69 simultaneous tests across variables
# ─────────────────────────────────────────────────────────────

from statsmodels.stats.multitest import multipletests

p_wilcox_vals = np.array([r["p_wilcox"] for r in results])
valid_mask = ~np.isnan(p_wilcox_vals)

reject = np.full(n_vars, False)
p_adj  = np.full(n_vars, np.nan)

if valid_mask.sum() > 0:
    rej, p_corrected, _, _ = multipletests(
        p_wilcox_vals[valid_mask], alpha=0.05, method='fdr_bh'
    )
    reject[valid_mask] = rej
    p_adj[valid_mask]  = p_corrected

for j, r in enumerate(results):
    r["p_wilcox_adj_bh"] = float(p_adj[j]) if not np.isnan(p_adj[j]) else np.nan
    r["significant_bh"]  = bool(reject[j])

# ─────────────────────────────────────────────────────────────
# STAGE 5: Summary output
# ─────────────────────────────────────────────────────────────

df = pd.DataFrame(results)
df = df.sort_values("p_wilcox_adj_bh")

print(f"\n{'='*70}")
print("Prithvi effect averaged across 3 loss families — per-variable results")
print(f"{'='*70}")
print(df[[
    "variable", "mean_d_bar", "median_d_bar",
    "p_ttest", "p_wilcox", "p_wilcox_adj_bh", "significant_bh"
]].to_string(index=False, float_format=lambda x: f"{x:.4e}"))

n_sig = df["significant_bh"].sum()
print(f"\nVariables where Prithvi significantly helps (BH-corrected α=0.05): {n_sig}/{n_vars}")
print(f"Variables where mean d_bar < 0 (Prithvi better on average): {(df['mean_d_bar'] < 0).sum()}/{n_vars}")

# Save results
df.to_csv(BASE_ROOT / "prithvi_effect_across_losses.csv", index=False)
print("\nSaved: " + str(BASE_ROOT / "prithvi_effect_across_losses.csv"))