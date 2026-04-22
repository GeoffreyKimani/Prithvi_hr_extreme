import matplotlib.pyplot as plt
from pathlib import Path

path = Path("~/scratch/prithvi_hr_extreme/outputs/hypothesis_tests_primary").expanduser()

proxies = ["hgt_500", "q_700", "q_850", "msl", "2t"]
improvement_pct = [11.15, 5.74, 6.38, 4.62, 4.38]

fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
bars = ax.bar(proxies, improvement_pct, color="#4C72B0")

ax.set_ylabel("Relative RMSE improvement (%)")
ax.set_xlabel("Rainfall proxy")
ax.set_title("Prithvi + Exloss: key rainfall proxies")

# Annotate bars
for b, val in zip(bars, improvement_pct):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.2,
            f"{val:.1f}%", ha="center", va="bottom", fontsize=8)

ax.axhline(0, color="grey", linewidth=0.8)
fig.tight_layout()
fig.savefig(f"{path}/fig2_rainfall_proxy_improvement.png", bbox_inches="tight")
plt.close(fig)