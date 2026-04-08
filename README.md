# HR-Extreme + Prithvi-WxC

This repository contains a research codebase for experimenting with Prithvi-WxC foundation model features on the HR-Extreme extreme weather dataset. It implements a U-Net baseline on HR-Extreme tiles and a hybrid model that fuses precomputed Prithvi feature maps at the U-Net bottleneck.

## Features

- HR-Extreme event-tile forecasting (3× input steps → 1× target step, 69 variables).
- Baseline temporal-fusion U-Net model (`unet_plain`).
- Hybrid “HR U-Net + Prithvi feature fusion” model (`unet_prithvi_*`).
- Offline Prithvi feature extraction pipeline from MERRA-2.
- Tail-aware loss functions (tail-weighted MSE and Exloss-style loss).
- Evaluation in physical units: per-variable RMSE and per-event-type RMSE.

## Repository structure

```text
configs/       # Model + training configs (except machine-specific paths)
datasets/      # HR-Extreme and HR-Extreme+Prithvi dataset classes
models/        # HREncoder, HRHead, HRUNet, HRPrithviModel
training/      # train/eval scripts and loss/metric helpers
scripts/       # Data prep, index building, feature extraction, analysis
index_files/   # Split/index CSVs (paths and metadata)
stats/         # Derived stats (e.g., HR-Extreme quantiles, if present)
```

Key files:

- `configs/hrx_prithvi_backbone.yaml`: experiment + model hyperparameters.
- `configs/prithvi_validation_config.yaml`: Prithvi-WxC validation config (paths resolved at runtime).

- `datasets/hr_extreme_dataset.py`: baseline HR-Extreme dataset.
- `datasets/hr_extreme_prithvi_dataset.py`: paired HR-Extreme + Prithvi dataset.

- `models/hr_encoder.py`, `models/hr_head.py`, `models/hr_unet_model.py`: U-Net baseline.
- `models/hr_prithvi_model.py`: hybrid HR U-Net + Prithvi feature fusion model.

- `scripts/PrithviFeatureExtractor/prithvi_feature_extractor.py`: Prithvi feature extraction.
- `scripts/DataLayerArtifacts/*`: HR-Extreme/Prithvi indexing and geo-alignment helpers.

- `training/train_hr_prithvi.py`: main training entrypoint.
- `training/eval_hr_prithvi.py`: evaluation entrypoint.

## Installation

1. Clone this repository:

```bash
git clone git@github.com:GeoffreyKimani/Prithvi_hr_extreme.git
cd prithvi_hr_extreme
```

2. Create and activate a conda environment:

```bash
conda env create -f environment.yml
conda activate hrx-prithvi
```

3. Install the Prithvi-WxC codebase separately and set `PRITHVI_ROOT` in your paths config (see below).

## Configuration

There are two kinds of configs:

- **Portable configs** (tracked in Git):
  - `configs/hrx_prithvi_backbone.yaml`
  - `configs/prithvi_validation_config.yaml`

- **Machine-specific paths** (not tracked):
  - `configs/paths.yaml` – local paths to HR-Extreme tiles, Prithvi feature NPZs, and the Prithvi-WxC repo.

At minimum, set:

- `hr_extreme`: directory with HR-Extreme `.npz` tiles.
- `prithvi_feats_root`: directory with `~/scratch/prithvi_features/{split}/`.
- `prithvi_repo_root`: path to your cloned Prithvi-WxC repo.

## Data preparation

### 1. HR-Extreme tiles

This repo assumes HR-Extreme `.npz` files were generated using the official HR-Extreme makedata scripts for 2020 (train/val/test splits). Place them under the `hr_extreme` path configured in `configs/paths.yaml`, and ensure that `index_files/hrx_prithvi_{train,val,test}.csv` point to the correct `.npz` locations.

If needed, use:

```bash
python scripts/build_splits_hrx_prithvi.py
```

to regenerate the `hrx_prithvi_{split}.csv` index files.

### 2. MERRA-2 for Prithvi

Download and preprocess the MERRA-2 inputs required for the HR-Extreme event dates:

```bash
python scripts/DataLayerArtifacts/download_merra2_for_hr_extreme.py \
  --input-data-dir /path/to/Prithvi-WxC/data/merra-2 \
  --download-dir   /path/to/Prithvi-WxC/data/merra-2_raw
```

### 3. Prithvi feature extraction

Build Prithvi-aligned indices and extract features:

```bash
# Build event indices for Prithvi
python scripts/DataLayerArtifacts/prithvi_hrrr_index_extractor.py --split train
python scripts/DataLayerArtifacts/prithvi_hrrr_index_extractor.py --split test

# Attach lat/lon bounds to the indices
python scripts/DataLayerArtifacts/hrrr_indices_to_lat_lon.py

# Extract Prithvi features for HR-Extreme events
python scripts/PrithviFeatureExtractor/prithvi_feature_extractor.py
```

This will create `~/scratch/prithvi_features/{train,test}/` NPZs that are referenced by `HRExtremeWithPrithviDataset`.

## Running experiments

### Baseline U-Net (HR‑Extreme only)

```bash
python training/train_hr_prithvi.py \
  --config configs/hrx_prithvi_backbone.yaml \
  training.experiment_name=unet_plain
```

### Hybrid U-Net + Prithvi (MSE loss)

```bash
python training/train_hr_prithvi.py \
  --config configs/hrx_prithvi_backbone.yaml \
  training.experiment_name=unet_prithvi_mse
```

### Hybrid U-Net + Prithvi (tail-weighted loss)

```bash
python training/train_hr_prithvi.py \
  --config configs/hrx_prithvi_backbone.yaml \
  training.experiment_name=unet_prithvi_tail
```

### Hybrid U-Net + Prithvi (Exloss-style loss)

```bash
python training/train_hr_prithvi.py \
  --config configs/hrx_prithvi_backbone.yaml \
  training.experiment_name=unet_prithvi_exloss
```

Checkpoints and evaluation outputs are written under:

```text
outputs/<experiment_name>/
  best_model.pt
  last_model.pt
  eval_test/
    rmse_per_variable_test.npy
    rmse_per_variable_test_table.csv
    rmse_per_variable_per_event_test.npz
    event_type_rmse_summary_rain.csv
```

### Test-time evaluation

To evaluate a trained model on the test split:

```bash
python training/eval_hr_prithvi.py \
  --config configs/hrx_prithvi_backbone.yaml \
  training.experiment_name=unet_prithvi_exloss
```