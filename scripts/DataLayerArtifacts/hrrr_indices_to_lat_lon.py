import numpy as np
import pandas as pd
from pathlib import Path

HOME = Path.home()
hr_extreme_root = HOME / "scratch" / "HR-Extreme"
latlon_path = hr_extreme_root / "index_files" / "latlon_grid_hrrr.npy"

# load HRRR lat/lon grid
latlon = np.load(latlon_path, allow_pickle=True)
lat = latlon[..., 0]   # (1059, 1799)
lon = latlon[..., 1]   # (1059, 1799)

proj_root = HOME / "scratch" / "prithvi_hr_extreme"
index_dir = proj_root / "index_files"

for name in ["prithvi_index_train.csv", "prithvi_index_test.csv"]:
    df = pd.read_csv(index_dir / name)

    lat_min_list, lat_max_list = [], []
    lon_min_list, lon_max_list = [], []

    for _, r in df.iterrows():
        y0, y1 = int(r["minY"]), int(r["maxY"])
        x0, x1 = int(r["minX"]), int(r["maxX"])

        lat_patch = lat[y0:y1, x0:x1]
        lon_patch = lon[y0:y1, x0:x1]

        lat_min_list.append(float(lat_patch.min()))
        lat_max_list.append(float(lat_patch.max()))
        lon_min_list.append(float(lon_patch.min()))
        lon_max_list.append(float(lon_patch.max()))

    df["lat_min"] = lat_min_list
    df["lat_max"] = lat_max_list
    df["lon_min"] = lon_min_list
    df["lon_max"] = lon_max_list

    out_path = index_dir / name.replace(".csv", "_with_latlon.csv")
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows to {out_path}")