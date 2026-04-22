import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

path = Path("~/scratch/prithvi_hr_extreme/outputs/hypothesis_tests_primary").expanduser()
summary_path = path / "primary_hypothesis_summary_across_variants.csv"
summary = pd.read_csv(summary_path)

# Keep only Prithvi variants
keep = ["unet_prithvi_mse", "unet_prithvi_tail", "unet_prithvi_exloss"]
df = summary[summary["variant"].isin(keep)].copy()

order = ["unet_prithvi_mse", "unet_prithvi_tail", "unet_prithvi_exloss"]
df["variant"] = pd.Categorical(df["variant"], categories=order, ordered=True)
df = df.sort_values("variant")

label_map = {
    "unet_prithvi_mse": "Prithvi + MSE",
    "unet_prithvi_tail": "Prithvi + Tail",
    "unet_prithvi_exloss": "Prithvi + Exloss",
}
labels = [label_map[v] for v in df["variant"]]

pct_improved = df["pct_significant_improvement"]
mean_rel = df["mean_rel_improvement_pct"]

fig, ax1 = plt.subplots(figsize=(6, 4), dpi=150)
x = range(len(labels))
width = 0.35

# Left axis: % improved
ax1.bar([i - width/2 for i in x], pct_improved, width=width,
        color="#4C72B0", label="% variables improved")
ax1.set_ylabel("% variables significantly improved")
ax1.set_xticks(list(x))
ax1.set_xticklabels(labels, rotation=0)
ax1.set_ylim(0, max(pct_improved.max(), 60) * 1.2)

# Right axis: mean relative change (%)
ax2 = ax1.twinx()
ax2.bar([i + width/2 for i in x], mean_rel, width=width,
        color="#DD8452", label="Mean relative RMSE change")
ax2.set_ylabel("Mean relative RMSE change (%)")
ax2.axhline(0, color="grey", linewidth=0.8, linestyle="--")

# Combine legends
h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, loc="upper left", frameon=False)

ax1.set_title("Prithvi variants: % improved vs mean change")

fig.tight_layout()
fig.savefig(f"{path}/fig1_prithvi_variants_summary.png", bbox_inches="tight")
plt.close(fig)