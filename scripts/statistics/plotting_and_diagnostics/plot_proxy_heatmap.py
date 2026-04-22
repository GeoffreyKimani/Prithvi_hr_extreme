"""
plot_proxy_heatmap_prithvi.py
==============================
Reads per-sample RMSE NPZ files for a baseline and a Prithvi-backbone experiment,
then produces two publication-ready heatmaps:

  1. rain_proxy_heatmap_all_events.png
     Rows  = 8 rainfall proxy variables (q_1000, q_850, q_700, u_850, v_850,
              hgt_500, msl, 2t)
     Cols  = every event type found in the aligned dataset
     Cells = mean RMSE delta  (baseline − Prithvi)
             Blue  → Prithvi better  |  Red  → Prithvi worse  |  White → no change

  2. rain_proxy_heatmap_rain_events.png
     Same layout but restricted to rain/precipitation-type events only.
     Useful for isolating where Prithvi's atmospheric prior helps most.

Both figures include:
  - Diverging RdBu colormap centred at zero
  - Cell annotations with Δ RMSE values
  - Asterisk (*) overlay where the per-proxy Wilcoxon test is significant at α=0.05
  - Colourbar labelled in physical units (same as your RMSE values)

Outputs
-------
All files written to:
  ~/scratch/prithvi_hr_extreme/outputs/{PRITHVI_EXP}/eval_test/figures/

Usage
-----
  python plot_proxy_heatmap_prithvi.py

  # or override at runtime:
  BASELINE_EXP=unet_plain_mse PRITHVI_EXP=unet_prithvi_backbone \\
      python plot_proxy_heatmap_prithvi.py

Dependencies
------------
  numpy, pandas, matplotlib, scipy
  (all standard in your CSF3 conda/venv environment)

Script order
------------
  This is Step 3 in the three-script pipeline.  See the README at the bottom
  of this docstring for the full recommended execution order.

  Step 1 — diagnostics_rain_proxies.py
    Per-proxy Q-Q plots, histograms, and Shapiro-Wilk normality check.
    Confirms whether the RMSE differences are normally distributed.

  Step 2 — compare_rain_proxies_prithvi.py
    Aligns baseline and Prithvi NPZs, runs Wilcoxon signed-rank tests
    per proxy (all events + rain events), and writes the CSV that this
    script reads.

  Step 3 — plot_proxy_heatmap_prithvi.py  ← YOU ARE HERE
    Reads Step 2's CSV, computes per-proxy-per-event-type mean deltas,
    and renders the heatmaps.

You CAN run this script standalone without Step 2's CSV if you do not need
the Wilcoxon significance asterisks — in that case set SKIP_WILCOXON = True
at the top of the configuration block below.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import wilcoxon

# ── Project path setup ───────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.hrx_var_names import HRX_VAR_NAMES

# ═══════════════════════════════════════════════════════════════════════════════
#  USER CONFIGURATION — edit these before running
# ═══════════════════════════════════════════════════════════════════════════════

BASE_ROOT    = Path("~/scratch/prithvi_hr_extreme/outputs").expanduser()
BASELINE_EXP = "unet_plain_mse"
PRITHVI_EXP  = "unet_prithvi_mse"

# Set True to skip Wilcoxon overlays (faster; useful before Step 2 has been run)
SKIP_WILCOXON = False

# Significance level for asterisk overlay
ALPHA = 0.05

# Keywords that identify rain-type events (case-insensitive substring match)
RAIN_EVENT_KEYWORDS: list[str] = [
    "rain", "flood", "precip", "convective",
    "thunder", "tropical", "hurricane", "typhoon",
]

# Rainfall proxy definitions — must match those in compare_rain_proxies_prithvi.py
RAIN_PROXY_DEFINITIONS: dict[str, str] = {
    "2t":      "2t",
    "msl":     "msl",
    "q_1000":  "q 100000",
    "q_850":   "q 85000",
    "q_700":   "q 70000",
    "u_850":   "u 85000",
    "v_850":   "v 85000",
    "hgt_500": "hgtn 50000",
}

# Display labels for each proxy — used as y-axis tick labels
PROXY_LABELS: dict[str, str] = {
    "2t":      "T 2m",
    "msl":     "MSLP",
    "q_1000":  "q 1000 hPa",
    "q_850":   "q 850 hPa",
    "q_700":   "q 700 hPa",
    "u_850":   "u 850 hPa",
    "v_850":   "v 850 hPa",
    "hgt_500": "Z 500 hPa",
}

# Ordered list of proxy names — controls row order in heatmap (top → bottom)
PROXY_ORDER: list[str] = [
    "q_1000", "q_850", "q_700",   # moisture column (grouped)
    "u_850", "v_850",              # low-level dynamics
    "hgt_500",                     # upper-air synoptic support
    "msl",                         # surface pressure
    "2t",                          # thermodynamic capacity
]

# ═══════════════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_per_sample_rmse(exp_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load rmse_per_variable_per_sample_test.npz for an experiment.

    Returns
    -------
    sample_indices : (N,) int
    event_types    : (N,) str
    rmse           : (N, C) float   — C = number of output channels
    """
    path = (
        BASE_ROOT / exp_name / "eval_test"
        / "rmse_per_variable_per_sample_test.npz"
    )
    if not path.exists():
        raise FileNotFoundError(
            f"NPZ not found: {path}\n"
            f"Has '{exp_name}' been evaluated? Check the eval_test/ directory."
        )
    data = np.load(path, allow_pickle=True)
    return data["sample_indices"], data["event_types"], data["rmse"]


def align_experiments(
    baseline_exp: str,
    prithvi_exp: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Inner-join baseline and Prithvi samples on sample_idx.
    Returns aligned (event_types, rmse_base, rmse_prithvi).
    """
    b_idx, b_et, b_rmse = load_per_sample_rmse(baseline_exp)
    p_idx, _,   p_rmse  = load_per_sample_rmse(prithvi_exp)

    df_b = pd.DataFrame({"sid": b_idx, "ib": np.arange(len(b_idx))})
    df_p = pd.DataFrame({"sid": p_idx, "ip": np.arange(len(p_idx))})
    m    = df_b.merge(df_p, on="sid", how="inner").sort_values("sid")

    return b_et[m["ib"].to_numpy()], b_rmse[m["ib"].to_numpy()], p_rmse[m["ip"].to_numpy()]


def resolve_proxy_indices() -> dict[str, int]:
    """Map proxy names → integer column indices in HRX_VAR_NAMES."""
    out, missing = {}, []
    for name, vstr in RAIN_PROXY_DEFINITIONS.items():
        try:
            out[name] = HRX_VAR_NAMES.index(vstr)
        except ValueError:
            missing.append((name, vstr))
    if missing:
        raise ValueError(
            "Variable strings not found in HRX_VAR_NAMES:\n"
            + "\n".join(f"  {n!r} → {v!r}" for n, v in missing)
            + "\nAdjust RAIN_PROXY_DEFINITIONS or check configs/hrx_var_names.py."
        )
    return out


# ═══════════════════════════════════════════════════════════════════════════════
#  HEATMAP DATA BUILDER
# ═══════════════════════════════════════════════════════════════════════════════

def build_heatmap_arrays(
    event_types: np.ndarray,
    rmse_base: np.ndarray,
    rmse_prithvi: np.ndarray,
    proxy_indices: dict[str, int],
    event_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    """
    Compute mean RMSE delta and Wilcoxon p-value matrices.

    Parameters
    ----------
    event_mask : optional boolean mask restricting which samples to include.
                 If None, all samples are used.

    Returns
    -------
    delta_matrix : (n_proxies, n_event_types) float   — mean (baseline − Prithvi)
    pval_matrix  : (n_proxies, n_event_types) float   — Wilcoxon p-value
    event_labels : list of str
    counts_matrix: (n_proxies, n_event_types) int     — n samples per cell
    """
    if event_mask is not None:
        et     = event_types[event_mask]
        rb     = rmse_base[event_mask]
        rp     = rmse_prithvi[event_mask]
    else:
        et, rb, rp = event_types, rmse_base, rmse_prithvi

    unique_events = sorted(set(str(e) for e in et))
    n_proxies     = len(PROXY_ORDER)
    n_events      = len(unique_events)
    event_to_col  = {ev: i for i, ev in enumerate(unique_events)}

    delta_mat  = np.full((n_proxies, n_events), np.nan)
    pval_mat   = np.ones((n_proxies, n_events))     # default p=1 (not significant)
    counts_mat = np.zeros((n_proxies, n_events), dtype=int)

    for row_i, proxy_name in enumerate(PROXY_ORDER):
        var_idx = proxy_indices[proxy_name]
        for ev in unique_events:
            col_i = event_to_col[ev]
            mask  = np.array([str(e) == ev for e in et])
            if mask.sum() < 2:
                continue
            diffs = rb[mask, var_idx] - rp[mask, var_idx]
            delta_mat[row_i, col_i]  = float(diffs.mean())
            counts_mat[row_i, col_i] = int(mask.sum())
            if not SKIP_WILCOXON:
                try:
                    _, p = wilcoxon(diffs, alternative="greater", zero_method="wilcox")
                    pval_mat[row_i, col_i] = p
                except ValueError:
                    pass  # all diffs zero; leave p=1

    return delta_mat, pval_mat, unique_events, counts_mat


# ═══════════════════════════════════════════════════════════════════════════════
#  PLOTTING
# ═══════════════════════════════════════════════════════════════════════════════

def plot_heatmap(
    delta_mat: np.ndarray,
    pval_mat: np.ndarray,
    event_labels: list[str],
    counts_mat: np.ndarray,
    title: str,
    out_path: Path,
) -> None:
    """
    Render and save one diverging heatmap.

    Colour scale is symmetric around zero, clipped at the 95th percentile of
    |delta| to prevent extreme outliers from washing out the palette.

    Asterisks (*) mark cells where the Wilcoxon p-value < ALPHA.
    """
    n_proxies = len(PROXY_ORDER)
    n_events  = len(event_labels)

    # Symmetric colour limit — robust to outliers
    abs_vals = np.abs(delta_mat[~np.isnan(delta_mat)])
    vmax     = float(np.percentile(abs_vals, 95)) if len(abs_vals) else 1.0
    vmax     = max(vmax, 1e-6)   # guard against all-zero data

    cmap = plt.cm.RdBu    # Red = Prithvi worse, Blue = Prithvi better
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    # Figure sizing: scale width with number of event types
    fig_w = max(10, 1.4 * n_events)
    fig_h = max(5,  0.9 * n_proxies + 2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(delta_mat, cmap=cmap, norm=norm, aspect="auto")

    # ── Colourbar ──
    cbar = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label("Mean RMSE delta\n(baseline − Prithvi)", fontsize=11)
    cbar.ax.tick_params(labelsize=9)

    # ── Axis labels ──
    ax.set_xticks(range(n_events))
    ax.set_xticklabels(event_labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(n_proxies))
    ax.set_yticklabels([PROXY_LABELS[p] for p in PROXY_ORDER], fontsize=10)
    ax.set_xlabel("Event Type", fontsize=11, labelpad=8)
    ax.set_ylabel("Rainfall Proxy Variable", fontsize=11, labelpad=8)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=14)

    # ── Cell annotations ──
    for ri in range(n_proxies):
        for ci in range(n_events):
            val = delta_mat[ri, ci]
            if np.isnan(val):
                ax.text(ci, ri, "—", ha="center", va="center",
                        fontsize=7, color="#aaaaaa")
                continue

            # Choose text colour for legibility against the cell fill
            cell_colour = cmap(norm(val))
            luminance   = 0.299*cell_colour[0] + 0.587*cell_colour[1] + 0.114*cell_colour[2]
            txt_colour  = "white" if luminance < 0.45 else "black"

            sig_marker = "*" if (not SKIP_WILCOXON and pval_mat[ri, ci] < ALPHA) else ""
            label      = f"{val:+.3f}{sig_marker}"

            ax.text(ci, ri, label, ha="center", va="center",
                    fontsize=7.5, color=txt_colour, fontweight="bold" if sig_marker else "normal")

    # ── Grid lines between cells ──
    ax.set_xticks(np.arange(n_events + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(n_proxies + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linewidth=0.8)
    ax.tick_params(which="minor", length=0)

    # ── Legend for asterisk ──
    if not SKIP_WILCOXON:
        ax.text(
            1.01, 0.01,
            f"* p < {ALPHA}\n  (Wilcoxon,\n  H₁: baseline > Prithvi)",
            transform=ax.transAxes,
            fontsize=8, va="bottom", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="#cccccc", alpha=0.85),
        )

    plt.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print(f"Experiments\n  baseline : {BASELINE_EXP}\n  prithvi  : {PRITHVI_EXP}\n")

    event_types, rmse_base, rmse_prithvi = align_experiments(BASELINE_EXP, PRITHVI_EXP)
    proxy_indices = resolve_proxy_indices()

    out_dir = BASE_ROOT / PRITHVI_EXP / "eval_test" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Figure 1: all events ──────────────────────────────────────────────────
    print("Building heatmap — ALL events …")
    delta_all, pval_all, ev_labels, counts_all = build_heatmap_arrays(
        event_types, rmse_base, rmse_prithvi, proxy_indices, event_mask=None
    )
    plot_heatmap(
        delta_all, pval_all, ev_labels, counts_all,
        title=(
            f"Rainfall Proxy RMSE Delta — All Events\n"
            f"{BASELINE_EXP}  vs.  {PRITHVI_EXP}\n"
            f"Blue = Prithvi better   |   Red = Prithvi worse   |   * p < {ALPHA}"
        ),
        out_path=out_dir / "rain_proxy_heatmap_all_events.png",
    )

    # ── Figure 2: rain events only ────────────────────────────────────────────
    et_lower  = np.array([str(e).lower() for e in event_types])
    rain_mask = np.zeros(len(et_lower), dtype=bool)
    for kw in RAIN_EVENT_KEYWORDS:
        rain_mask |= np.char.find(et_lower, kw.lower()) >= 0

    n_rain = int(rain_mask.sum())
    print(f"Building heatmap — RAIN events only ({n_rain} / {len(event_types)} samples) …")

    if n_rain < 2:
        print("  WARNING: fewer than 2 rain-type samples found — skipping rain-only heatmap.")
        print(f"  Check RAIN_EVENT_KEYWORDS against your event labels: {set(str(e) for e in event_types)}")
    else:
        delta_rain, pval_rain, ev_labels_rain, counts_rain = build_heatmap_arrays(
            event_types, rmse_base, rmse_prithvi, proxy_indices, event_mask=rain_mask
        )
        plot_heatmap(
            delta_rain, pval_rain, ev_labels_rain, counts_rain,
            title=(
                f"Rainfall Proxy RMSE Delta — Rain-Type Events Only  (n={n_rain})\n"
                f"{BASELINE_EXP}  vs.  {PRITHVI_EXP}\n"
                f"Blue = Prithvi better   |   Red = Prithvi worse   |   * p < {ALPHA}"
            ),
            out_path=out_dir / "rain_proxy_heatmap_rain_events.png",
        )

    print("\nDone. All figures written to:", out_dir)


if __name__ == "__main__":
    main()