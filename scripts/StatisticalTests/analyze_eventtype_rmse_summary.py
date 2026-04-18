from pathlib import Path
import argparse
import pandas as pd

BASE_ROOT = Path("~/scratch/prithvi_hr_extreme/outputs").expanduser()


def load_variable_rmse(exp_name: str) -> pd.DataFrame:
    """Load per-variable RMSE summary for one experiment."""
    path = BASE_ROOT / exp_name / "eval_test" / "rmse_per_variable_test_table.csv"
    df = pd.read_csv(path)
    df = df.rename(columns={"rmse_hr_extreme": f"rmse_{exp_name}"})
    return df  # columns: variable, rmse_<exp>


def load_eventtype_rmse(exp_name: str) -> pd.DataFrame:
    """Load per-event-type RMSE summary for one experiment (rain proxies)."""
    path = BASE_ROOT / exp_name / "eval_test" / "event_type_rmse_summary_rain.csv"
    df = pd.read_csv(path)
    df = df.rename(
        columns={
            "mean_rmse_all_vars": f"mean_all_vars_{exp_name}",
            "mean_rmse_rain_proxies": f"mean_rain_rmse_{exp_name}",
        }
    )
    return df  # columns: event_type, mean_all_vars_<exp>, mean_rain_rmse_<exp>


def compare_experiments(baseline: str, variant: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build joined summary tables baseline vs variant."""
    # Per-variable
    df_base_var = load_variable_rmse(baseline)
    df_var_var = load_variable_rmse(variant)
    df_var = df_base_var.merge(df_var_var, on="variable", how="inner")

    # Add absolute and relative improvements (baseline - variant)
    rmse_base_col = f"rmse_{baseline}"
    rmse_var_col = f"rmse_{variant}"
    df_var["delta_rmse"] = df_var[rmse_var_col] - df_var[rmse_base_col]
    df_var["rel_improvement_pct"] = 100.0 * (
        df_var[rmse_base_col] - df_var[rmse_var_col]
    ) / df_var[rmse_base_col]

    # Per-event-type (rain)
    df_base_evt = load_eventtype_rmse(baseline)
    df_var_evt = load_eventtype_rmse(variant)
    df_evt = df_base_evt.merge(df_var_evt, on="event_type", how="inner")

    base_all = f"mean_all_vars_{baseline}"
    var_all = f"mean_all_vars_{variant}"
    base_rain = f"mean_rain_rmse_{baseline}"
    var_rain = f"mean_rain_rmse_{variant}"

    df_evt["delta_all_vars"] = df_evt[var_all] - df_evt[base_all]
    df_evt["rel_improvement_all_pct"] = 100.0 * (
        df_evt[base_all] - df_evt[var_all]
    ) / df_evt[base_all]

    df_evt["delta_rain"] = df_evt[var_rain] - df_evt[base_rain]
    df_evt["rel_improvement_rain_pct"] = 100.0 * (
        df_evt[base_rain] - df_evt[var_rain]
    ) / df_evt[base_rain]

    return df_var, df_evt


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
        df_var, df_evt = compare_experiments(args.baseline, variant)

        df_var.to_csv(
            out_dir / f"compare_variables_{args.baseline}_vs_{variant}.csv",
            index=False,
        )
        df_evt.to_csv(
            out_dir / f"compare_eventtypes_{args.baseline}_vs_{variant}.csv",
            index=False,
        )

        print(f"Saved RMSE comparison for {args.baseline} vs {variant} in {out_dir}")


if __name__ == "__main__":
    main()