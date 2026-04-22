from pathlib import Path
import argparse
import numpy as np
import pandas as pd

BASE_ROOT = Path("~/scratch/prithvi_hr_extreme/outputs").expanduser()

def load_eventtype_var_rmse(exp_name: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load event_type × variable RMSE for one experiment.

    Returns
    -------
    event_types : np.ndarray of shape (N_types,)
    rmse        : np.ndarray of shape (N_types, N_vars)
    """
    path = BASE_ROOT / exp_name / "eval_test" / "rmse_per_variable_per_event_test.npz"
    data = np.load(path, allow_pickle=True)
    event_types = data["event_types"]          # (N_types,)
    rmse = data["rmse"]                        # (N_types, N_vars)
    return event_types, rmse


def paired_rmse_diff_eventtype_var(
    baseline: str,
    variant: str,
    var_names: list[str] | None = None,
) -> pd.DataFrame:
    """
    Compute paired RMSE differences (variant - baseline) for each event_type and variable.

    Returns
    -------
    DataFrame with columns:
      event_type, variable, rmse_<baseline>, rmse_<variant>, delta_rmse, rel_improvement_pct
    """
    ev_base, rmse_base = load_eventtype_var_rmse(baseline)
    ev_var, rmse_var = load_eventtype_var_rmse(variant)

    # Sanity check: same event_types and shape
    assert np.array_equal(ev_base, ev_var), "Event types differ between experiments"
    assert rmse_base.shape == rmse_var.shape, "RMSE shapes differ between experiments"

    n_types, n_vars = rmse_base.shape

    # Variable names: either passed in or simple indices
    if var_names is None:
        var_names = [f"var_{j}" for j in range(n_vars)]
    else:
        assert len(var_names) == n_vars, "var_names length must match number of variables"

    records = []
    for i in range(n_types):
        et = ev_base[i]
        for j in range(n_vars):
            vname = var_names[j]
            b = rmse_base[i, j]
            v = rmse_var[i, j]
            delta = v - b
            rel_improv = 100.0 * (b - v) / b if b != 0.0 else np.nan
            records.append(
                {
                    "event_type": et,
                    "variable": vname,
                    f"rmse_{baseline}": b,
                    f"rmse_{variant}": v,
                    "delta_rmse": delta,
                    "rel_improvement_pct": rel_improv,
                }
            )

    df = pd.DataFrame.from_records(records)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", default="unet_plain")
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["unet_prithvi_mse", "unet_prithvi_tail", "unet_prithvi_exloss"],
    )
    args = parser.parse_args()

    out_dir = BASE_ROOT / "comparisons"
    out_dir.mkdir(parents=True, exist_ok=True)

    for variant in args.variants:
        df_diffs = paired_rmse_diff_eventtype_var(args.baseline, variant)
        out_path = out_dir / f"rmse_diffs_{args.baseline}_vs_{variant}.csv"
        df_diffs.to_csv(out_path, index=False)
        print(f"Saved RMSE differences to {out_path}")


if __name__ == "__main__":
    main()