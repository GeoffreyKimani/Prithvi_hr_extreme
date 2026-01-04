import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path


class HRExtremeDataset(Dataset):
    """
    HR-Extreme dataset wrapper for extracted NPZ files.

    Each .npz file is expected to contain:
      - x:    (T_in, 69, 320, 320)
      - y:    (69, 320, 320)
      - mask: (320, 320)
    """

    def __init__(self, data_dir: str, split: str = "train"):
        self.data_dir = Path(data_dir)
        self.split = split

        self.files = sorted(self.data_dir.glob("*.npz"))
        if not self.files:
            raise RuntimeError(f"No .npz files found in {self.data_dir}")
        
        # Sanity check: ensure every path is readable
        missing = [p for p in self.files if not p.is_file()]
        if missing:
            raise RuntimeError(f"{len(missing)} dataset files are missing, e.g. {missing[0]}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        print(f"Loading from: {path}")
        arr = np.load(path)

        x = arr["x"]      # (T_in, 69, 320, 320)
        y = arr["y"]      # (69, 320, 320)
        mask = arr["mask"]  # (320, 320)

        x = torch.from_numpy(x).float()     # (T_in, 69, 320, 320)
        y = torch.from_numpy(y).float()     # (69, 320, 320)
        mask = torch.from_numpy(mask).float()  # (320, 320)

        return x, y, mask
