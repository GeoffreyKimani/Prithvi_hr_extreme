from pathlib import Path
import argparse
import pandas as pd

BASE_ROOT = Path("~/scratch/prithvi_hr_extreme/outputs").expanduser()


def summarize_variant_result(csv_path: Path, alpha: float = 0.05) -> dict:
    """
    Summarize one primary-hypothesis result CSV.

    Expected columns:
      - variant
      - variable
      - mean_diff
      - p_w_less_holm
    """
    df = pd.read_csv(csv_path)

    required_cols = {"variant", "variable", "mean_diff", "p_w_less_holm"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {missing}")

    variant = df["variant"].iloc[0]
    n_variables = len(df)

    sig_improve = ((df["p_w_less_holm"] < alpha) & (df["mean_diff"] < 0)).sum()
    sig_worse = ((df["p_w_less_holm"] < alpha) & (df["mean_diff"] > 0)).sum()
    no_change = n_variables - sig_improve - sig_worse

    pct_improve = 100.0 * sig_improve / n_variables
    pct_worse = 100.0 * sig_worse / n_variables
    pct_no_change = 100.0 * no_change / n_variables

    mean_rel_improvement = df["rel_improvement_pct"].mean()
    median_rel_improvement = df["rel_improvement_pct"].median()

    top_improved = (
        df.sort_values("mean_diff", ascending=True)
        .head(5)["variable"]
        .tolist()
    )
    top_worsened = (
        df.sort_values("mean_diff", ascending=False)
        .head(5)["variable"]
        .tolist()
    )

    return {
        "variant": variant,
        "n_variables": n_variables,
        "alpha": alpha,
        "n_significant_improvement": int(sig_improve),
        "n_significant_worsening": int(sig_worse),
        "n_no_significant_change": int(no_change),
        "pct_significant_improvement": pct_improve,
        "pct_significant_worsening": pct_worse,
        "pct_no_significant_change": pct_no_change,
        "mean_rel_improvement_pct": mean_rel_improvement,
        "median_rel_improvement_pct": median_rel_improvement,
        "top_5_improved_variables": ", ".join(top_improved),
        "top_5_worsened_variables": ", ".join(top_worsened),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=str,
        default=str(BASE_ROOT / "hypothesis_tests_primary"),
        help="Directory containing primary_hypothesis_*.csv files",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance threshold applied to Holm-adjusted Wilcoxon p-values",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir).expanduser()
    out_dir = input_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(input_dir.glob("primary_hypothesis_*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No primary_hypothesis_*.csv files found in {input_dir}")

    summaries = []
    for csv_path in csv_files:
        summaries.append(summarize_variant_result(csv_path, alpha=args.alpha))

    df_summary = pd.DataFrame(summaries)
    out_path = out_dir / "primary_hypothesis_summary_across_variants.csv"
    df_summary.to_csv(out_path, index=False)

    print(f"Saved summary table to {out_path}")
    print(df_summary[[
        "variant",
        "n_significant_improvement",
        "n_significant_worsening",
        "n_no_significant_change",
        "pct_significant_improvement",
        "pct_significant_worsening",
        "pct_no_significant_change",
    ]])


if __name__ == "__main__":
    main()