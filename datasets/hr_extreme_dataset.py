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

    def __init__(self, data_dir, stats_path=None, normalize=True, split="train"):
        self.data_dir = Path(data_dir).expanduser()
        self.stats_path = stats_path
        self.normalize = normalize
        self.split = split

        if not self.data_dir.is_dir():
            raise RuntimeError(f"Data directory does not exist: {self.data_dir}")

        self.files = sorted(self.data_dir.glob("*.npz"))
        if not self.files:
            raise RuntimeError(f"No .npz files found in {self.data_dir}")

        # Sanity check: make sure all files resolve
        missing = [p for p in self.files if not p.is_file()]
        if missing:
            raise RuntimeError(
                f"{len(missing)} dataset files are missing or broken, "
                f"example: {missing[0]}"
            )
        
        if stats_path is not None and self.normalize:
            stats = np.load(stats_path)
            # HF file uses keys "means" and "stds" for the 69 channels
            mean = stats["means"].astype("float32")   # shape (69,)
            std = stats["stds"].astype("float32")     # shape (69,)

            # Guard against zeros in std to avoid division by zero
            std[std == 0] = 1.0

            # Store as (1, C, 1, 1) for easy broadcasting
            self.mean_x = torch.from_numpy(mean)[None, :, None, None]
            self.std_x = torch.from_numpy(std)[None, :, None, None]
        else:
            self.mean_x = None
            self.std_x = None

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        with np.load(path) as arr:
            x = arr["x"]      # (T_in, 69, 320, 320)
            y = arr["y"]      # (69, 320, 320)
            mask = arr["mask"]  # (320, 320)

        x = torch.from_numpy(x).float()     # (T_in, 69, 320, 320)
        y = torch.from_numpy(y).float()     # (69, 320, 320)
        mask = torch.from_numpy(mask).float()  # (320, 320)

        if self.normalize and self.mean_x is not None and self.std_x is not None:
            x = (x - self.mean_x) / self.std_x
            
            y_4d = y.unsqueeze(0)  # (1,69,H,W)
            y_4d = (y_4d - self.mean_x) / self.std_x
            y = y_4d.squeeze(0)

        return x, y, mask
