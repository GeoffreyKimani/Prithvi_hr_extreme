#!/usr/bin/env python
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# root = Path("/net/scratch/j22132gk/HR-Extreme/20200101_20200630")
# bad = []

# for p in root.glob("*.npz"):
#     with np.load(p) as arr:
#         x = arr["inputs"]
#         y = arr["targets"]
#     if x.shape[1] == 0 or y.shape[1] == 0:
#         bad.append(p)

# print("Bad files:", len(bad))

# Uncomment to delete them once you're sure!!! 
# Comment back once done, to avoid accidentally deleting more files 
# for p in bad:
#     p.unlink()


def check_prithvi_split(split_dir: Path):
    bad = []

    for f in sorted(split_dir.glob("*_prithvi.npz")):
        try:
            with np.load(f) as pz:
                # Force reading all arrays to catch EOF/truncation
                for k in pz.files:
                    _ = pz[k]
                # Optionally check for the expected key/shape
                if "feats" not in pz.files:
                    bad.append((f, "missing_feats_key"))
        except Exception as e:  # catches EOFError + anything else
            bad.append((f, f"{type(e).__name__}: {e!s}"))

    return bad


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=str,
        default=str(Path("~/scratch/prithvi_features").expanduser()),
        help="Root directory for Prithvi features (contains train/val/test).",
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="If set, delete bad npz files after detection.",
    )
    parser.add_argument(
        "--report",
        type=str,
        default="prithvi_bad_files.csv",
        help="Path to CSV report of bad files.",
    )
    args = parser.parse_args()

    root = Path(args.root)
    splits = ["train", "val", "test"]

    all_bad = []

    for split in splits:
        d = root / split
        if not d.exists():
            continue
        print(f"\nChecking Prithvi split: {split} ({d})")
        bad = check_prithvi_split(d)
        print(f"  Found {len(bad)} bad files in {split}")
        for f, msg in bad:
            print(f"  BAD: {f} -> {msg}")
            all_bad.append({"split": split, "path": str(f), "error": msg})

            if args.delete:
                try:
                    f.unlink()
                    print(f"    Deleted {f}")
                except Exception as e:
                    print(f"    Failed to delete {f}: {e}")

    if all_bad:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(all_bad).to_csv(report_path, index=False)
        print(f"\nWrote report with {len(all_bad)} entries to {report_path}")
    else:
        print("\nNo bad Prithvi feature files found.")


if __name__ == "__main__":
    main()