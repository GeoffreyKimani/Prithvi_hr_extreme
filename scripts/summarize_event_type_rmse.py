from pathlib import Path
import sys
import numpy as np
import yaml

SCRIPT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_ROOT.parent))

from configs.hrx_var_names import HRX_VAR_NAMES

# Define rainfall-proxy variables from HR-Extreme variable list.
# From Table 1 and dataset description, the closest proxies are:
#   - 2t: Temperature 2m above ground (not rainfall itself but impacts rain)
#   - q at low levels (specific humidity)
#   - We assume you have a "pr" or "precip" channel; if not, we'll focus on low-level q.
#
# Here we'll treat:
#   - q_1000mb, q_925mb, q_850mb as rainfall-related humidity proxies.
#   You need to map these to their channel indices in your 69-channel ordering.

RAIN_PROXY_INDICES = {
    "2t": HRX_VAR_NAMES.index("2t"),
    "q 100000": HRX_VAR_NAMES.index("q 100000"),
    "q 92500":  HRX_VAR_NAMES.index("q 92500"),
    "q 85000":  HRX_VAR_NAMES.index("q 85000")
}

ALL_VAR_NAMES = None  # optional: list of 69 variable names if you have them


def load_per_event_results(npz_path: str | Path):
    npz = np.load(npz_path, allow_pickle=True)
    event_types = npz["event_types"].tolist()
    rmse = npz["rmse"]  # [N_event_types, C]
    return event_types, rmse


def summarize_rainfall_focus(event_types, rmse, out_csv: str | Path):
    """
    Build a small table: for each event type, report:
      - mean RMSE over all 69 variables
      - mean RMSE over rainfall proxies
    """
    import csv

    out_csv = Path(out_csv)
    rows = []

    # indices for rainfall proxies
    rain_idx = list(RAIN_PROXY_INDICES.values())

    for i, et in enumerate(event_types):
        rmse_vec = rmse[i]  # (C,)

        mean_rmse_all = float(rmse_vec.mean())
        mean_rmse_rain = float(rmse_vec[rain_idx].mean()) if rain_idx else float("nan")

        rows.append({
            "event_type": et,
            "mean_rmse_all_vars": mean_rmse_all,
            "mean_rmse_rain_proxies": mean_rmse_rain,
        })

    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["event_type", "mean_rmse_all_vars", "mean_rmse_rain_proxies"])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"Wrote event-type summary to {out_csv}")


def main():
    exp_name = "unet_prithvi_exloss"  # <-- set this to your experiment name
    ckpt_root = Path(f"~/scratch/prithvi_hr_extreme/outputs/{exp_name}").expanduser()
    npz_path = ckpt_root / "eval_test" / "rmse_per_variable_per_event_test.npz"

    event_types, rmse = load_per_event_results(npz_path)

    out_csv = ckpt_root / "eval_test" / "event_type_rmse_summary_rain.csv"
    summarize_rainfall_focus(event_types, rmse, out_csv)


if __name__ == "__main__":
    main()