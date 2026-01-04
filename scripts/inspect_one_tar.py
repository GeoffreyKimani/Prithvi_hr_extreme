# scripts/inspect_one_tar.py
import tarfile
from pathlib import Path
import numpy as np

tar_path = str(Path("~/scratch/hr_extreme_data/202007_202012/0010.tar").expanduser())  # adjust to an existing tar

with tarfile.open(tar_path, "r") as tar:
    for member in tar:
        print("Member name:", member.name)
        if not member.name.endswith(".npz"):
            continue

        f = tar.extractfile(member)
        if f is None:
            continue

        try:
            data = np.load(f)
        except Exception as e:
            print("  Failed to load as npz:", e)
            continue

        print("  Loaded as npz.")
        print("  Keys:", list(data.keys()))
        for k in data.keys():
            arr = data[k]
            print(f"    {k}: shape={arr.shape}, dtype={arr.dtype}")
        break  # inspect only the first .npz file