"""
compare_rain_proxies_prithvi.py
================================
Compares per-sample RMSE between a baseline model and the Prithvi-backbone
variant, specifically for rainfall-proxy variables.

Outputs
-------
1. compare_rain_proxies_prithvi.csv
   Columns: proxy_name | var_index | n_samples | baseline_mean | prithvi_mean
            | delta_mean | pct_improvement | wilcoxon_stat | wilcoxon_p
            | significant (bool, α=0.05) | n_rain_events | wilcoxon_stat_rain
            | wilcoxon_p_rain | significant_rain (bool, α=0.05)
   - All-events rows are computed over every aligned sample.
   - Rain-event rows reuse the same aligned data filtered to rain-type events.

2. rain_proxy_delta_summary.txt
   Human-readable console-style summary of findings.

Usage
-----
    python compare_rain_proxies_prithvi.py

    # or override experiment names at runtime
    BASELINE_EXP=unet_plain_mse PRITHVI_EXP=unet_prithvi_backbone \
        python compare_rain_proxies_prithvi.py
"""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

# ── Project path setup ───────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.hrx_var_names import HRX_VAR_NAMES

# ── Experiment configuration ─────────────────────────────────────────────────
BASE_ROOT = Path("~/scratch/prithvi_hr_extreme/outputs").expanduser()

BASELINE_EXP = "unet_plain_exloss"
PRITHVI_EXP  = "unet_prithvi_exloss"

# Keywords used to identify rain/precipitation events in the event_types array.
# Case-insensitive substring match — adjust to match your actual event labels.
RAIN_EVENT_KEYWORDS: list[str] = [
    "rain",
    "flood",
    "convective",
    "thunder",
    "waterspout",
    "hail",
]

# ── Rainfall proxy variable definitions ──────────────────────────────────────
# Each entry maps a short human-readable name to the exact string in HRX_VAR_NAMES.
# Justification for each proxy is documented in summarize_event_type_rmse.py.
RAIN_PROXY_DEFINITIONS: dict[str, str] = {
    "2t":      "2t",          # 2-m temperature (Clausius-Clapeyron moisture capacity)
    "msl":     "msl",         # Mean sea-level pressure (cyclone driver)
    "q_1000":  "q 100000",    # Specific humidity — boundary layer (moisture fuel)
    "q_850":   "q 85000",     # Specific humidity — low-level jet transport
    "q_700":   "q 70000",     # Specific humidity — mid-level moisture depth
    "u_850":   "u 85000",     # Zonal wind 850 hPa (convergence)
    "v_850":   "v 85000",     # Meridional wind 850 hPa (moisture flux)
    "hgt_500": "hgtn 50000",   # Geopotential height 500 hPa (synoptic forcing troughs)
}

# Significance level
ALPHA = 0.05


# ── Data loading ─────────────────────────────────────────────────────────────

def load_per_sample_rmse(exp_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load per-sample RMSE from an experiment's eval_test directory.

    Returns
    -------
    sample_indices : (N,) int array
    event_types    : (N,) object array of strings
    rmse           : (N, C) float array  — C = number of output variables
    """
    path = BASE_ROOT / exp_name / "eval_test" / "rmse_per_variable_per_sample_test.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"NPZ not found: {path}\n"
            f"Check that experiment '{exp_name}' has been evaluated and the file "
            f"'rmse_per_variable_per_sample_test.npz' exists under eval_test/."
        )
    data = np.load(path, allow_pickle=True)
    return data["sample_indices"], data["event_types"], data["rmse"]


def align_experiments(
    baseline_exp: str,
    prithvi_exp: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Align baseline and Prithvi RMSE arrays by shared sample indices.

    Returns
    -------
    event_types_aligned  : (N_shared,) array
    rmse_base_aligned    : (N_shared, C) array
    rmse_prithvi_aligned : (N_shared, C) array
    """
    base_idx, base_et, base_rmse       = load_per_sample_rmse(baseline_exp)
    prith_idx, _prith_et, prith_rmse   = load_per_sample_rmse(prithvi_exp)

    df_base  = pd.DataFrame({"sample_idx": base_idx,  "i_base":  np.arange(len(base_idx))})
    df_prith = pd.DataFrame({"sample_idx": prith_idx, "i_prith": np.arange(len(prith_idx))})

    merged = (
        df_base.merge(df_prith, on="sample_idx", how="inner")
        .sort_values("sample_idx")
        .reset_index(drop=True)
    )

    bi = merged["i_base"].to_numpy()
    pi = merged["i_prith"].to_numpy()

    return base_et[bi], base_rmse[bi], prith_rmse[pi]


# ── Rain-event filtering ─────────────────────────────────────────────────────

def rain_event_mask(event_types: np.ndarray) -> np.ndarray:
    """
    Boolean mask selecting samples whose event_type contains any
    RAIN_EVENT_KEYWORDS (case-insensitive).
    """
    et_lower = np.array([str(et).lower() for et in event_types])
    mask = np.zeros(len(et_lower), dtype=bool)
    for kw in RAIN_EVENT_KEYWORDS:
        mask |= np.char.find(et_lower, kw.lower()) >= 0
    return mask


# ── Statistical test ─────────────────────────────────────────────────────────

def run_wilcoxon(
    rmse_base: np.ndarray,
    rmse_prithvi: np.ndarray,
    var_idx: int,
) -> dict:
    """
    Wilcoxon signed-rank test on per-sample RMSE differences for one variable.

    H₁ (alternative='greater'): RMSE_baseline > RMSE_prithvi
    i.e., Prithvi produces lower RMSE (improvement).

    Returns a dict with: n, baseline_mean, prithvi_mean, delta_mean,
    pct_improvement, wilcoxon_stat, wilcoxon_p, significant.
    """
    base_vec  = rmse_base[:, var_idx]
    prith_vec = rmse_prithvi[:, var_idx]
    diffs     = base_vec - prith_vec  # positive = Prithvi better

    n = len(diffs)
    baseline_mean = float(base_vec.mean())
    prithvi_mean  = float(prith_vec.mean())
    delta_mean    = float(diffs.mean())
    pct_improvement = (delta_mean / baseline_mean * 100) if baseline_mean != 0 else float("nan")

    try:
        stat, p = wilcoxon(diffs, alternative="greater", zero_method="wilcox")
    except ValueError:
        # All diffs are zero — no variation
        stat, p = float("nan"), float("nan")

    return {
        "n":               n,
        "baseline_mean":   round(baseline_mean, 6),
        "prithvi_mean":    round(prithvi_mean, 6),
        "delta_mean":      round(delta_mean, 6),
        "pct_improvement": round(pct_improvement, 3),
        "wilcoxon_stat":   round(float(stat), 3) if not np.isnan(stat) else "nan",
        "wilcoxon_p":      f"{p:.4g}" if not np.isnan(p) else "nan",
        "significant":     bool(p < ALPHA) if not np.isnan(p) else False,
    }


# ── Resolve proxy indices ─────────────────────────────────────────────────────

def resolve_proxy_indices() -> dict[str, int]:
    """
    Resolve each RAIN_PROXY_DEFINITIONS entry to an integer index into HRX_VAR_NAMES.
    Raises a clear error if any variable name is not found.
    """
    indices: dict[str, int] = {}
    missing = []
    for proxy_name, var_string in RAIN_PROXY_DEFINITIONS.items():
        try:
            indices[proxy_name] = HRX_VAR_NAMES.index(var_string)
        except ValueError:
            missing.append((proxy_name, var_string))

    if missing:
        lines = "\n".join(f"  {n!r} → looking for {v!r}" for n, v in missing)
        raise ValueError(
            f"The following proxy variable strings were NOT found in HRX_VAR_NAMES:\n{lines}\n"
            f"Check configs/hrx_var_names.py and adjust RAIN_PROXY_DEFINITIONS accordingly."
        )
    return indices


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"Loading data for experiments:\n  baseline → {BASELINE_EXP}\n  prithvi  → {PRITHVI_EXP}\n")

    event_types, rmse_base, rmse_prithvi = align_experiments(BASELINE_EXP, PRITHVI_EXP)
    proxy_indices = resolve_proxy_indices()

    rain_mask = rain_event_mask(event_types)
    n_rain = int(rain_mask.sum())
    n_total = len(event_types)
    print(f"Aligned samples : {n_total}")
    print(f"Rain-type events: {n_rain} ({n_rain / n_total * 100:.1f}%)")
    print(f"Keywords matched: {RAIN_EVENT_KEYWORDS}\n")

    rows: list[dict] = []

    for proxy_name, var_idx in proxy_indices.items():
        # ── All events ──
        all_stats = run_wilcoxon(rmse_base, rmse_prithvi, var_idx)

        # ── Rain events only ──
        if n_rain > 1:
            rain_stats = run_wilcoxon(
                rmse_base[rain_mask], rmse_prithvi[rain_mask], var_idx
            )
        else:
            rain_stats = {k: "nan" for k in all_stats}
            rain_stats["n"] = n_rain

        row = {
            "proxy_name":         proxy_name,
            "var_index":          var_idx,
            "var_string":         RAIN_PROXY_DEFINITIONS[proxy_name],
            # ── All events ──
            "n_samples":          all_stats["n"],
            "baseline_mean":      all_stats["baseline_mean"],
            "prithvi_mean":       all_stats["prithvi_mean"],
            "delta_mean":         all_stats["delta_mean"],
            "pct_improvement":    all_stats["pct_improvement"],
            "wilcoxon_stat":      all_stats["wilcoxon_stat"],
            "wilcoxon_p":         all_stats["wilcoxon_p"],
            "significant":        all_stats["significant"],
            # ── Rain events only ──
            "n_rain_events":      rain_stats["n"],
            "baseline_mean_rain": rain_stats.get("baseline_mean", "nan"),
            "prithvi_mean_rain":  rain_stats.get("prithvi_mean", "nan"),
            "delta_mean_rain":    rain_stats.get("delta_mean", "nan"),
            "pct_improvement_rain": rain_stats.get("pct_improvement", "nan"),
            "wilcoxon_stat_rain": rain_stats.get("wilcoxon_stat", "nan"),
            "wilcoxon_p_rain":    rain_stats.get("wilcoxon_p", "nan"),
            "significant_rain":   rain_stats.get("significant", False),
        }
        rows.append(row)

        sig_all  = "✓" if row["significant"]      else "✗"
        sig_rain = "✓" if row["significant_rain"] else "✗"
        print(
            f"  {proxy_name:<10}  Δ={row['delta_mean']:+.4f}  ({row['pct_improvement']:+.2f}%)  "
            f"p_all={row['wilcoxon_p']:<8} {sig_all}   "
            f"p_rain={row['wilcoxon_p_rain']:<8} {sig_rain}"
        )

    # ── Write CSV ──
    out_dir = BASE_ROOT / PRITHVI_EXP / "eval_test"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "compare_rain_proxies_prithvi.csv"

    fieldnames = list(rows[0].keys())
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nCSV written → {csv_path}")

    # ── Write text summary ──
    summary_path = out_dir / "rain_proxy_delta_summary.txt"
    with summary_path.open("w") as f:
        f.write(f"Rainfall Proxy RMSE Comparison: {BASELINE_EXP} vs. {PRITHVI_EXP}\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Total aligned samples : {n_total}\n")
        f.write(f"Rain-type events      : {n_rain}  ({n_rain / n_total * 100:.1f}%)\n")
        f.write(f"Significance level    : α = {ALPHA}\n\n")
        f.write(f"{'Proxy':<10}  {'Δ RMSE (all)':>14}  {'%imp':>7}  {'p (all)':>10}  "
                f"{'Sig?':>5}  {'p (rain)':>10}  {'Sig?':>5}\n")
        f.write("-" * 70 + "\n")
        for r in rows:
            sig_all  = "YES" if r["significant"]      else "no"
            sig_rain = "YES" if r["significant_rain"] else "no"
            f.write(
                f"{r['proxy_name']:<10}  {r['delta_mean']:>+14.6f}  "
                f"{r['pct_improvement']:>+7.2f}%  {str(r['wilcoxon_p']):>10}  "
                f"{sig_all:>5}  {str(r['wilcoxon_p_rain']):>10}  {sig_rain:>5}\n"
            )
        f.write("\n")
        sig_proxies_all  = [r["proxy_name"] for r in rows if r["significant"]]
        sig_proxies_rain = [r["proxy_name"] for r in rows if r["significant_rain"]]
        f.write(f"Significant across ALL events  ({len(sig_proxies_all)}/{len(rows)}): "
                f"{', '.join(sig_proxies_all) or 'none'}\n")
        f.write(f"Significant in RAIN events only ({len(sig_proxies_rain)}/{len(rows)}): "
                f"{', '.join(sig_proxies_rain) or 'none'}\n")
    print(f"Summary written → {summary_path}")


if __name__ == "__main__":
    main()