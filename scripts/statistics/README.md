This folder contains scripts for:
- computing descriptive RMSE statistics for HR-Extreme experiments,
- running hypothesis tests on Prithvi vs plain U-Net variants,
- generating plots and diagnostics used in the paper and slides.

## Folder structure

- `descriptive_tests/`
- `hypothesis_testing/`
- `plotting_and_diagnostics/`
- top-level helpers: `npz_reader.py`, `analyze_eventtype_rmse_summary.py`, `analyze_eventtype_var_rmse_diffs.py`

### `test_primary_hypothesis_per_sample.py`
**Question:**  
Does adding Prithvi features reduce per-sample RMSE for each HR-Extreme variable, relative to a plain U-Net baseline, under a given loss?

**Inputs:**

- Per-sample RMSE NPZs for each experiment:
  `~/scratch/prithvi_hr_extreme/outputs/<exp_name>/eval_test/rmse_per_variable_per_sample_test.npz`
  containing:
  - `sample_indices` (N,)
  - `event_types` (N,)
  - `rmse` (N, C) per-variable RMSE
  - `mean_rmse_all_vars` (N,) overall RMSE per sample

**Outputs:**

- For each `variant` vs `baseline` pair, a CSV:
  `~/scratch/prithvi_hr_extreme/outputs/hypothesis_tests_primary/primary_hypothesis_<baseline>_vs_<variant>.csv`

  Each row = variable, with:
  - `mean_diff`, `median_diff`, bootstrap CI for the mean diff
  - one-sided paired t-test and Wilcoxon statistics (variant < baseline)
  - `rel_improvement_pct` (mean percentage RMSE change)
  - Holm–Bonferroni adjusted Wilcoxon p-value `p_w_less_holm`
  - `significant_improvement` flag

**Example usage:**

```bash
cd DescriptiveStatistics/HypothesisTesting

python test_primary_hypothesis_per_sample.py \
  --baseline unet_plain_mse \
  --variants unet_prithvi_mse unet_prithvi_tail unet_prithvi_exloss \
  --n-bootstrap 10000
```

This reproduces the per-variable results used in the slides (Prithvi vs plain under matched loss families).

## Descriptive tests (`descriptive_tests/`)
These scripts compute descriptive summaries without formal hypothesis tests.

- `test_eventtype_rmse_improvement.py` – per-event RMSE comparisons (baseline vs variant) at event-type resolution.
- `test_prithvi_contribution_loss_avg.py` – across-loss average RMSE difference test for Prithvi vs plain.

## Plotting and diagnostics (`plotting_and_diagnostics/`)
These scripts generate the figures used in the paper:
- `plot_prithvi_variants.py` – bar / dual-axis plot for "% variables improved vs mean relative RMSE change" across Prithvi variants.
- `plot_loss_average.py` – distributions of across-loss average RMSE differences (e.g. histogram / KDE).
- `plot_rainfall_proxies.py`, `compare_rain_proxies.py` – proxy-level plots for rainfall-linked variables.
- `plot_proxy_heatmap.py` – heatmaps over variables × losses or variables × event types.
- `distributions.py` – Q–Q plots and distributions used for normality assessment.
- `diagnostics_rain_proxies.py` – additional diagnostics for rainfall proxies.

## Helpers
- `npz_reader.py` – utility functions to inspect NPZ files (per-sample / per-event RMSE) for debugging.
- `analyze_eventtype_rmse_summary.py` – analysis of per-event-type RMSE summaries.
- `analyze_eventtype_var_rmse_diffs.py` – analysis of variable-wise RMSE differences across event types.