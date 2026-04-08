#!/usr/bin/env python
"""
Inspect spatial tiles (lat_min/max, lon_min/max) for:
  - longitude convention (0–360 vs -180–180),
  - whether any tiles wrap across the dateline.

Intended for HR-Extreme / similar index files before using them with Prithvi-WxC.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Inspect lat/lon tiles for longitude convention and dateline crossings."
    )
    p.add_argument(
        "--index-path",
        type=str,
        required=True,
        help="Path to index CSV (must contain lat_min, lat_max, lon_min, lon_max).",
    )
    p.add_argument(
        "--out-wrap-csv",
        type=str,
        default=None,
        help="Optional path to write CSV of tiles that appear to wrap across the dateline.",
    )
    return p.parse_args()


def to_minus180_180(lon: np.ndarray) -> np.ndarray:
    """Map [0,360) to [-180,180); leave already-negative values alone."""
    lon = lon.copy()
    mask = lon >= 180.0
    lon[mask] = lon[mask] - 360.0
    return lon


def main() -> None:
    args = parse_args()
    idx_path = Path(args.index_path)

    if not idx_path.exists():
        raise FileNotFoundError(idx_path)

    df = pd.read_csv(idx_path)

    required = ["lat_min", "lat_max", "lon_min", "lon_max"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {idx_path}: {missing}")

    lat_min = df["lat_min"].to_numpy()
    lat_max = df["lat_max"].to_numpy()
    lon_min = df["lon_min"].to_numpy()
    lon_max = df["lon_max"].to_numpy()

    n = len(df)
    print(f"Loaded {n} tiles from {idx_path}")

    # Basic sanity: lat_min < lat_max
    bad_lat = lat_min >= lat_max
    print(f"Tiles with lat_min >= lat_max: {bad_lat.sum()}")

    # Detect longitude convention
    has_ge_180 = (lon_min >= 180.0) | (lon_max >= 180.0)
    print(f"Tiles with any longitude >= 180 (0–360 style): {has_ge_180.sum()}")

    # In raw (index) convention, detect wrap tiles (lon_min > lon_max) for 0–360 grids
    wrap_raw = (lon_min > lon_max) & has_ge_180
    print(f"Tiles that appear to wrap (lon_min > lon_max in 0–360): {wrap_raw.sum()}")

    # Optionally dump wrap tiles
    if args.out_wrap_csv and wrap_raw.any():
        # Convert to -180–180 for use with MERRA/Prithvi
        lon_min_conv = to_minus180_180(lon_min)
        lon_max_conv = to_minus180_180(lon_max)

        # After conversion, check if any intervals still look inverted
        bad_lon_after = lon_min_conv >= lon_max_conv
        print(
            "Tiles with lon_min_conv >= lon_max_conv after 0–360→-180–180 conversion:",
            bad_lon_after.sum(),
        )

        # Write out the wrap tiles with converted longitudes for potential use in cropping MERRA/Prithvi features
        out_path = Path(args.out_wrap_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df_wrap = df.loc[wrap_raw].copy()
        df_wrap["lon_min_conv"] = lon_min_conv[wrap_raw]
        df_wrap["lon_max_conv"] = lon_max_conv[wrap_raw]
        df_wrap.to_csv(out_path, index=False)
        print(f"Wrote {wrap_raw.sum()} wrap tiles to {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()