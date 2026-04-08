import argparse
import pandas as pd
from pathlib import Path

HOME = Path.home()
DATA_ROOT = HOME / "scratch" / "hr_extreme_data" # Update this path to where the HR-Extreme .npz files are stored (too lazy to expand ~ in code)
HR_EXTREME_INDEX_ROOT = HOME / "scratch" / "HR-Extreme" / "index_files" # similarly, update this to where the HR-Extreme CSV files are stored (they contain metadata about the events and bounding boxes)

TRAIN_DIR = DATA_ROOT / "20200101_20200630"
TEST_DIR  = DATA_ROOT / "20200701_20201231"

TRAIN_CSV = HR_EXTREME_INDEX_ROOT / "data_20200101_20200630_info.csv"
TEST_CSV  = HR_EXTREME_INDEX_ROOT / "data_20200701_20201231_info.csv"

parser = argparse.ArgumentParser(description="Build Prithvi index for HR-Extreme.")
parser.add_argument(
    "--split",
    choices=["train", "test"],
    default="train",
    help="Which split to build index for.",
)
args = parser.parse_args()
data_dir = None

if args.split == "train":
    df = pd.read_csv(TRAIN_CSV)
    print(f"Read {len(df)} rows from {TRAIN_CSV}")
    data_dir = TRAIN_DIR
    out_name = "prithvi_index_train.csv"
elif args.split == "test":
    df = pd.read_csv(TEST_CSV)
    print(f"Read {len(df)} rows from {TEST_CSV}")
    data_dir = TEST_DIR
    out_name = "prithvi_index_test.csv"

# Map filenames in the chosen split
files = {p.name: p for p in data_dir.glob("*.npz")}
# print the first 10 files to verify
# print(f"First 10 files in {data_dir}, type{type(files)}, len files: {len(files)}")
# print(f"First 10 file names: {list(files.keys())[:2]}\n\n")

# Save index inside the *current project* under index_files/
# (run this script from the prithvi_hr_extreme repo root)
prithvi_index_dir = Path("index_files")
prithvi_index_dir.mkdir(parents=True, exist_ok=True)

rows = []
for _, r in df.iterrows():
    dt = pd.to_datetime(r["begin_time"])
    date_str = dt.strftime("%Y%m%d%H")

    bbox = r["bounding_box"]  # "853_757_903_828"
    minX, minY, maxX, maxY = map(int, bbox.split("_"))
    event_type = r["type"]

    prefix = f"{date_str}_{event_type}_{bbox}"
    matches = [name for name in files if name.startswith(prefix)]
    # print(f"Matches for {prefix}: {matches}")

    for fname in matches:
        rows.append(
            {
                "npz_dir": str(data_dir),
                "npz_filename": fname,
                "event_type": event_type,
                "datetime": dt.isoformat(),
                "minX": minX,
                "minY": minY,
                "maxX": maxX,
                "maxY": maxY,
            }
        )

out = pd.DataFrame(rows)
out_path = prithvi_index_dir / out_name
out.to_csv(out_path, index=False)
print(f"Wrote {len(out)} rows to {out_path}")