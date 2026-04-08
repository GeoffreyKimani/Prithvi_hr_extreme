"""
Prithvi Feature Extractor Module
This module extracts weather features from MERRA2 data using the Prithvi-WxC model
and processes them for HR-Extreme extreme weather events.
Key Functionality:
- Loads and initializes the pretrained Prithvi-WxC model for inference
- Builds MERRA2 datasets centered on specific datetime events
- Converts MERRA2 samples to Prithvi-compatible batches with proper padding
- Runs Prithvi inference on weather data
- Crops and interpolates model outputs to 320x320 HRRR tiles
- Saves extracted features as compressed NumPy files with metadata
Dependencies:
- PyTorch for model inference
- OmegaConf for configuration management
- Prithvi-WxC model and dataloaders
- MERRA2 climate data and validation utilities
Workflow:
1. Load validation configuration and pretrained Prithvi model
2. Read HR-Extreme event index (dates, locations)
3. For each event: build MERRA2 dataset, run Prithvi, crop/resize output
4. Save features with associated metadata (datetime, lat/lon bounds, hazard info)
- NPZ files containing feature tensors [P, 320, 320] and metadata dictionaries
- Organized into train/ and test/ subdirectories
"""
import sys 
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from omegaconf import OmegaConf
from types import SimpleNamespace

# --- Paths ---
HOME = Path.home()
PROJ_ROOT = HOME / "scratch" / "prithvi_hr_extreme"
PRITHVI_ROOT = HOME / "scratch" / "Prithvi-WxC"      # root of Prithvi repo
TRAIN_INDEX_PATH = PROJ_ROOT / "index_files" / "prithvi_index_train_with_latlon.csv"
TEST_INDEX_PATH = PROJ_ROOT / "index_files" / "prithvi_index_test_with_latlon.csv"
VAL_CFG_PATH = PROJ_ROOT / "configs" / "prithvi_validation_config.yaml"
PRITHVI_FEAT_DIR = HOME / "scratch" / "prithvi_features" # changed this
PRITHVI_FEAT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda")  # keep CPU for smoke test first

# Make Prithvi code importable
sys.path.insert(0, str(PRITHVI_ROOT))

from PrithviWxC.dataloaders.merra2 import Merra2Dataset, preproc        # from merra2.py
from PrithviWxC.model import PrithviWxC          # from model.py
from validation.validate_prithvi_wxc import (
    assemble_input_scalers,
    assemble_static_input_scalers,
    assemble_output_scalers,
)

# Make local script modules importable (e.g. DataLayerArtifacts)
SCRIPT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_ROOT.parent))

from DataLayerArtifacts.lat_lon_to_merra2_indices import latlon_to_merra_indices, to_minus180_180_scalar

def _dict_to_namespace(d):
    """Recursively convert a dictionary to a SimpleNamespace."""
    if isinstance(d, dict):
        return SimpleNamespace(**{k: _dict_to_namespace(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [_dict_to_namespace(item) for item in d]
    else:
        return d
    
def ns_get(ns, key, default=None):
    return getattr(ns, key, default)

def load_validation_config():
    """Load Prithvi validation config and resolve paths under PRITHVI_ROOT."""
    cfg = OmegaConf.load(str(VAL_CFG_PATH))  # returns a DictConfig

    # Fix any relative paths in cfg to be absolute based on PRITHVI_ROOT
    for key in ["data_path_surface", "data_path_vertical", "climatology_path_surface", "climatology_path_vertical"]:
        if key in cfg["data"] and cfg["data"][key] is not None:
            cfg["data"][key] = str(PRITHVI_ROOT / cfg["data"][key])

     # Similarly fix relative paths passed to the model
    for key in ["input_scalers_surface_path", "input_scalers_vertical_path", "output_scalers_surface_path", "output_scalers_vertical_path"]:
        if key in cfg["model"] and cfg["model"][key] is not None:
            cfg["model"][key] = str(PRITHVI_ROOT / cfg["model"][key])

    # Convert the config to a plain container for easier access in the rest of the code (optional)
    cfg_plain = OmegaConf.to_container(cfg, resolve=True)

    # Wrap in a SimpleNamespace for attribute access (optional)
    cfg_ns = _dict_to_namespace(cfg_plain)
    return cfg_ns

def load_prithvi_model(cfg):
    """
        Build PrithviWxC model for inference with HR-Extreme static layout.

        - Base config: validation/config.yaml.
        - Override:
            in_channels_static = 10
            positional_encoding = "absolute"
        - Expand static input scalers from 4 -> 10 channels by tiling.

    Returns: PrithviWxC model with pretrained weights, in eval mode.
    """
    
    # Load the config file
    config = cfg 

    # 1) Build scalers exactly like validation
    input_mu, input_sigma = assemble_input_scalers(config)
    static_input_mu, static_input_sigma = assemble_static_input_scalers(config)
    output_var = assemble_output_scalers(config)

    # 2) Construct model with same arguments as get_model
    mcfg = config.model
    dcfg = config.data

    in_channels = len(dcfg.surface_vars) + len(dcfg.levels) * len(dcfg.vertical_vars)
    n_lats_px = dcfg.input_size_lat + sum(dcfg.padding.lat)
    n_lons_px = dcfg.input_size_lon + sum(dcfg.padding.lon)

    # in_channels_static = mcfg.num_static_channels
    in_channels_static = 10  # for now just use 10 static channels (lat, lon, and 8 static vars) since the pretrained model was trained with that
    # Expand 4-channel static scalers to 10 by tiling
    if static_input_mu.shape[0] != in_channels_static:
        reps = (in_channels_static + static_input_mu.shape[0] - 1) // static_input_mu.shape[0]
        static_input_mu = static_input_mu.repeat(reps)[:in_channels_static]
        static_input_sigma = static_input_sigma.repeat(reps)[:in_channels_static]

    # print("static_input_mu shape:", static_input_mu.shape)
    # print("static_input_sigma shape:", static_input_sigma.shape)
    # print("in_channels_static:", in_channels_static)


    model = PrithviWxC(
        in_channels=in_channels,
        input_size_time=2,  # same as validation
        in_channels_static=in_channels_static,
        input_scalers_mu=input_mu,
        input_scalers_sigma=input_sigma,
        input_scalers_epsilon=0.0,
        static_input_scalers_mu=static_input_mu,
        static_input_scalers_sigma=static_input_sigma,
        static_input_scalers_epsilon=0.0,
        output_scalers=torch.sqrt(output_var),
        n_lats_px=n_lats_px,
        n_lons_px=n_lons_px,
        patch_size_px=tuple(mcfg.token_size),
        mask_unit_size_px=config.mask_unit_size,
        mask_ratio_inputs=config.mask_ratio_inputs,
        mask_ratio_targets=0.0,
        embed_dim=mcfg.embed_dim,
        n_blocks_encoder=mcfg.n_blocks_encoder,
        n_blocks_decoder=mcfg.n_blocks_decoder,
        mlp_multiplier=mcfg.mlp_multiplier,
        n_heads=mcfg.n_heads,
        dropout=mcfg.dropout_rate,
        drop_path=mcfg.drop_path,
        parameter_dropout=mcfg.parameter_dropout,
        residual=mcfg.residual,
        masking_mode=mcfg.masking_mode,
        encoder_shifting=mcfg.encoder_shift,
        decoder_shifting=mcfg.decoder_shift,
        # positional_encoding=ns_get(mcfg, "positional_encoding", "absolute"),
        positional_encoding="absolute",  # force absolute
        checkpoint_encoder=[int(i) for i in mcfg.checkpoint_encoder],
        checkpoint_decoder=[int(i) for i in mcfg.checkpoint_decoder],
    )

    # 3) Load pretrained weights
    weights_path = PRITHVI_ROOT / "data" / "weights" / "prithvi.wxc.2300m.v1.pt"
    state = torch.load(weights_path, map_location="cpu")
    if "state_dict" in state:
        state = state["state_dict"]
    if "model_state" in state:
        state = state["model_state"]

    # Filter out keys whose shapes don't match the current model
    model_state = model.state_dict()
    filtered_state = {}
    for k, v in state.items():
        if k in model_state and model_state[k].shape == v.shape:
            filtered_state[k] = v
        # else: silently drop

    missing, unexpected = model.load_state_dict(filtered_state, strict=False)
    print(f"Loaded {len(filtered_state)} params; "
        f"missing={len(missing)}, unexpected={len(unexpected)}")

    model.to(DEVICE)
    model.eval()
    return model

def build_merra_dataset_for_row(cfg, row):
    """
        Given a HR-Extreme row with a 'datetime' string, build a small Merra2Dataset
        centered around that time (±24h window, input_time=-3h, lead_time=0h). [137]
    """
    data_cfg = cfg.data
    mcfg = cfg.model

    data_cfg.lead_time = 0
    data_cfg.input_time = -3


    # 1. Build a tiny time_range around this row's datetime
    dt = datetime.fromisoformat(row["datetime"])
    
    # Create a time range window of 3hrs around the datetime (1.5hrs before and after)
    start_dt = dt - timedelta(hours=24)
    end_dt = dt + timedelta(hours=24)
    time_range = (start_dt.strftime("%Y-%m-%dT%H:%M:%S"), end_dt.strftime("%Y-%m-%dT%H:%M:%S"))

    # 2. Read lead_times / input_times from config (or define small lists)
    lead_times = [data_cfg.lead_time]    # Merra2Dataset expects list
    input_times = [data_cfg.input_time]       

    # 3. Construct Merra2Dataset with explicit args
    ds = Merra2Dataset(
        time_range=time_range,
        lead_times=lead_times,
        input_times=input_times,
        data_path_surface=data_cfg.data_path_surface,
        data_path_vertical=data_cfg.data_path_vertical,
        climatology_path_surface=ns_get(data_cfg, "climatology_path_surface", None),
        climatology_path_vertical=ns_get(data_cfg, "climatology_path_vertical", None),
        surface_vars=data_cfg.surface_vars,
        static_surface_vars=data_cfg.static_surface_vars,
        vertical_vars=data_cfg.vertical_vars,
        levels=data_cfg.levels,
        roll_longitudes=ns_get(data_cfg, "roll_longitudes_train", False),
        positional_encoding=ns_get(mcfg, "positional_encoding", "absolute"),
    )

    return ds, dt

def merra_sample_to_prithvi_batch(sample: dict, cfg) -> dict[str, torch.Tensor]:
    """
    Convert a Merra2Dataset sample dict into a Prithvi-ready batch dict with
    correct padding and shapes, on DEVICE. Uses the official 'preproc'.
    """
    # batch of size 1
    batch_list = [sample]

    padding = {
        "lat": tuple(cfg.data.padding.lat),
        "lon": tuple(cfg.data.padding.lon),
        "level": tuple(cfg.data.padding.level),
    }

    batch = preproc(batch_list, padding)  # dict with x, y, static, climate, etc.

    # move tensors to DEVICE
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(DEVICE)

    return batch

def run_prithvi_on_dataset_sample(model, sample, cfg):
    """Run Prithvi on one Merra2 sample, return [P, H, W]."""
    batch = merra_sample_to_prithvi_batch(sample, cfg)
    
    # print("static shape:", batch["static"].shape)
    # print("model.in_channels_static:", model.in_channels_static)
    # print("model.positional_encoding:", model.positional_encoding)

    with torch.no_grad():
        out = model(batch)   # [1, P, H, W]
    return out.squeeze(0).cpu()    # [P, H, W]

def crop_and_interpolate_to_320(feats: torch.Tensor, row) -> torch.Tensor:
    """
    Crop Prithvi features [P, H, W] to the HRRR tile given by row's lat/lon bounds,
    then resize to [P, 320, 320].

    feats: [P, H, W] (output of run_prithvi_on_dataset_sample, CPU tensor)
    row:  pandas Series with lat_min, lat_max, lon_min, lon_max
    """
    assert feats.ndim == 3, f"Expected [P,H,W], got {feats.shape}"
    P, H, W = feats.shape

    lat_min = float(row["lat_min"])
    lat_max = float(row["lat_max"])
    lon_min = to_minus180_180_scalar(float(row["lon_min"]))
    lon_max = to_minus180_180_scalar(float(row["lon_max"]))

    j0, j1, i0, i1 = latlon_to_merra_indices(lat_min, lat_max, lon_min, lon_max, H, W)

    # debugs
    # print("feats shape:", feats.shape)
    # print("lat/lon:", lat_min, lat_max, lon_min, lon_max)
    # print("indices:", "j0,j1 =", j0, j1, "i0,i1 =", i0, i1)

    # crop in lat (H) and lon (W)
    feats_crop = feats[:, j0:j1, i0:i1]  # [P, Hc, Wc]
    print("crop shape:", feats_crop.shape)

    if feats_crop.shape[-1] == 0 or feats_crop.shape[-2] == 0:
        raise RuntimeError(
            f"Empty crop: feats={tuple(feats.shape)}, "
            f"lat=({lat_min},{lat_max}), lon=({lon_min},{lon_max}), "
            f"j=({j0},{j1}), i=({i0},{i1})"
        )

    # resize cropped patch to 320x320
    feats_crop = feats_crop.unsqueeze(0)  # [1,P,Hc,Wc]
    feats_320 = F.interpolate(
        feats_crop, size=(320, 320), mode="bilinear", align_corners=False
    )
    return feats_320.squeeze(0)  # [P,320,320]

def save_prithvi_features(row, feats_320, dt, out_dir=PRITHVI_FEAT_DIR):
    """
    Save Prithvi features for one HR-Extreme event.

    Args:
        row: Pandas Series for the event (must contain 'npz_filename', 'datetime',
             and lat/lon and hazard info).
        feats_320: torch.Tensor or np.ndarray of shape [P, 320, 320].
        dt: datetime corresponding to the Merra2 sample time.
        out_dir: base directory to store outputs.

    Output:
        <out_dir>/<stem>_prithvi.npz, where <stem> is row['npz_filename'] without .npz.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(row["npz_filename"]).stem
    out_path = out_dir / f"{stem}_prithvi.npz"

    if isinstance(feats_320, torch.Tensor):
        feats_np = feats_320.numpy()
    else:
        feats_np = np.asarray(feats_320)

    meta = {
        "datetime": row["datetime"],
        "event_type": row.get("event_type", ""),
        "npz_dir": row.get("npz_dir", ""),
        "minX": row.get("minX", None),
        "minY": row.get("minY", None),
        "maxX": row.get("maxX", None),
        "maxY": row.get("maxY", None),
        "lat_min": row.get("lat_min", None),
        "lat_max": row.get("lat_max", None),
        "lon_min": row.get("lon_min", None),
        "lon_max": row.get("lon_max", None),
        "merra_time_used": dt.isoformat(),
        "channels": int(feats_np.shape[0]),
        "height": int(feats_np.shape[1]),
        "width": int(feats_np.shape[2]),
    }

    np.savez_compressed(out_path, feats=feats_np, **meta)
    print(f"Saved Prithvi features to {out_path}")
    return out_path


def main():
    # choose which split to process in this run
    split = "train"  # or "test"
    if split == "train":
        df = pd.read_csv(TRAIN_INDEX_PATH)

        # parse datetimes
        df["date"] = pd.to_datetime(df["datetime"])

        # create a mask for the train set for running small extractions
        train_mask = (df["date"] >= "2020-01-01") & (df["date"] < "2020-02-01") # done jan, feb, mar, april, may, june change back to july
        train_df = df[train_mask].reset_index(drop=True)
        
        work_df = train_df
        out_dir = PRITHVI_FEAT_DIR / "train"
        print(f"Train rows: {len(train_df)}")

    else:
        df = pd.read_csv(TEST_INDEX_PATH)

        # parse datetimes
        df["date"] = pd.to_datetime(df["datetime"])

        # create a mask for the test set for running small extractions
        test_mask = (df["date"] >= "2020-11-01") & (df["date"] <= "2020-12-31") # done July, Aug, Sept, Oct, Nov, Dec
        test_df = df[test_mask].reset_index(drop=True)

        work_df = test_df
        out_dir = PRITHVI_FEAT_DIR / "test"
        print(f"Test rows: {len(test_df)}")


    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_validation_config()
    model = load_prithvi_model(cfg)

    for i, row in work_df.iterrows():
        stem = Path(row["npz_filename"]).stem
        out_path = out_dir / f"{stem}_prithvi.npz"

        if out_path.exists():
            print(f"\n[{split}] Row {i}: {row['npz_filename']} already has {out_path.name}, skipping.")
            continue

        print(f"\n[{split}] Row {i}: {row['npz_filename']}, time={row['datetime']}")

        ds, dt = build_merra_dataset_for_row(cfg, row)

        # 1) Get Merra2 sample
        sample = ds[0]

        # 2) (optional) print shapes only for first few rows
        if i < 3:
            batch = merra_sample_to_prithvi_batch(sample, cfg)
            print("batch['x'] shape:", batch["x"].shape)
            print("batch['y'] shape:", batch["y"].shape)
            print("batch['static'] shape:", batch["static"].shape)

        # 3) Run Prithvi
        prithvi_out = run_prithvi_on_dataset_sample(model, sample, cfg)  # [P, H, W]
        if i < 3:
            print("Raw Prithvi output shape:", prithvi_out.shape)

        # 4) Crop to HRRR tile and interpolate to 320×320
        feats_320 = crop_and_interpolate_to_320(prithvi_out, row)
        if i < 3:
            print("Cropped+interpolated features shape:", feats_320.shape)

        # 5) Save features with metadata into split-specific dir
        save_prithvi_features(row, feats_320, dt, out_dir=out_dir)



if __name__ == "__main__":
    main()