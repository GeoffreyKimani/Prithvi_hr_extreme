from pathlib import Path
import sys
import numpy as np
import csv

SCRIPT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_ROOT.parent))

from configs.hrx_var_names import HRX_VAR_NAMES


def main():
    # Adjust if your outputs_root is elsewhere
    exp_name = "unet_prithvi_exloss"  # <-- set this to your experiment name
    ckpt_root = Path(f"~/scratch/prithvi_hr_extreme/outputs/{exp_name}").expanduser()
    eval_dir = ckpt_root / "eval_test"
    npz_path = eval_dir / "rmse_per_variable_test.npy"  # or .npz depending on your code

    # If you used np.save in eval: it's .npy; if np.savez_compressed: it's .npz.
    if npz_path.suffix == ".npy":
        rmse = np.load(npz_path)  # shape (69,)
    else:
        data = np.load(npz_path, allow_pickle=True)
        # assume key is "rmse" if you used savez; adjust if different
        rmse = data["rmse"] if "rmse" in data.files else data[data.files[0]]

    assert rmse.shape[0] == len(HRX_VAR_NAMES), (
        f"Expected {len(HRX_VAR_NAMES)} variables, got {rmse.shape[0]}"
    )

    out_csv = eval_dir / "rmse_per_variable_test_table.csv"

    with out_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["variable", "rmse_hr_extreme"])
        for name, val in zip(HRX_VAR_NAMES, rmse):
            writer.writerow([name, float(val)])

    print(f"Wrote per-variable RMSE table to {out_csv}")


if __name__ == "__main__":
    main()