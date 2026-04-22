import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from pathlib import Path

path = Path("~/scratch/prithvi_hr_extreme/outputs/hypothesis_tests_primary").expanduser()

base = "unet_plain_mse"
pairs = [
    ("unet_prithvi_mse", f"{path}/primary_hypothesis_unet_plain_mse_vs_unet_prithvi_mse.csv"),
    ("unet_prithvi_tail", f"{path}/primary_hypothesis_unet_plain_mse_vs_unet_prithvi_tail.csv"),
    ("unet_prithvi_exloss", f"{path}/primary_hypothesis_unet_plain_mse_vs_unet_prithvi_exloss.csv"),
]

dfs = []
for name, fname in pairs:
    df = pd.read_csv(fname)
    df = df[["variable", "mean_diff", "p_w_less_holm", "significant_improvement"]].copy()
    df = df.rename(columns={
        "mean_diff": f"mean_diff_{name}",
        "p_w_less_holm": f"p_{name}",
        "significant_improvement": f"sig_{name}",
    })
    dfs.append(df)

merged = dfs[0]
for df in dfs[1:]:
    merged = merged.merge(df, on="variable", how="inner")

# Across-loss average effect per variable (negative = Prithvi better on average)
mean_cols = [c for c in merged.columns if c.startswith("mean_diff_")]
merged["mean_diff_across_loss"] = merged[mean_cols].mean(axis=1)

# Example: define 'across-loss significant' as:
# all three mean_diff < 0 and at least one BH-corrected p < 0.05
p_cols = [c for c in merged.columns if c.startswith("p_")]
merged["across_loss_sig"] = (
    (merged[mean_cols] < 0).all(axis=1)
    & (merged[p_cols] < 0.05).any(axis=1)
)

# Count summary (for annotation)
n_total = len(merged)
n_mean_lt_zero = (merged["mean_diff_across_loss"] < 0).sum()
n_sig = merged["across_loss_sig"].sum()

print(f"{n_mean_lt_zero}/{n_total} with mean_diff_across_loss < 0")
print(f"{n_sig}/{n_total} across-loss significant")

# Histogram
fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
ax.hist(merged["mean_diff_across_loss"], bins=20, color="#4C72B0", alpha=0.8)
ax.axvline(0, color="grey", linestyle="--", linewidth=0.8)

ax.set_xlabel("Across-loss mean RMSE difference (Prithvi – baseline)")
ax.set_ylabel("Number of variables")
ax.set_title("Across-loss Prithvi effect over 69 variables")

# Annotate counts
text = f"{n_mean_lt_zero}/{n_total} < 0\n{n_sig}/{n_total} significant"
ax.text(0.98, 0.95, text, transform=ax.transAxes,
        ha="right", va="top", fontsize=9,
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))

fig.tight_layout()
fig.savefig(f"{path}/fig3_across_loss_histogram.png", bbox_inches="tight")
plt.close(fig)