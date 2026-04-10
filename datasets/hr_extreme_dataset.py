import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import pandas as pd


class HRExtremeDataset(Dataset):
    """
    HR-Extreme tiles without Prithvi features, backed by an index CSV.

    Each row in the index CSV must contain at least:
      - hrx_path: absolute or relative path to an .npz file

    Each .npz HR-Extreme file is expected to contain keys:
      - 'inputs': (1, T_in, 69, 320, 320)  input sequence of HRRR variables
      - 'targets': (1, T_out, 69, 320, 320) forecast targets (usually T_out = 1)
      - 'masks': (1, 320, 320) spatial mask of valid event area

    This dataset:
      - Loads the NPZ referenced by 'hrx_path' in the CSV.
      - Slices to shapes:
          x: (T_in, 69, 320, 320)
          y: (69, 320, 320)
          mask: (320, 320)
      - Optionally applies channel-wise normalization using the same
        means/stds used for HR-Extreme experiments.
    """

    def __init__(self, index_csv: str | Path, stats_path: str | Path | None = None, normalize: bool = True):
        self.index_csv = Path(index_csv)
        if not self.index_csv.is_file():
            raise RuntimeError(f"Index CSV does not exist: {self.index_csv}")

        self.df = pd.read_csv(self.index_csv)
        if len(self.df) == 0:
            raise RuntimeError(f"No rows in index file: {self.index_csv}")

        self.normalize = normalize

        if stats_path is not None and normalize:
            stats = np.load(str(Path(stats_path).expanduser()))
            mean = stats["means"].astype("float32")  # (69,)
            std = stats["stds"].astype("float32")    # (69,)
            std[std == 0] = 1.0

            self.mean_x = torch.from_numpy(mean)[None, :, None, None]  # (1,C,1,1)
            self.std_x = torch.from_numpy(std)[None, :, None, None]
        else:
            self.mean_x = None
            self.std_x = None

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        # retry a few times if we hit a bad file
        for _ in range(3):
            row = self.df.iloc[idx]
            event_type = row.get("event_type", "")
            path = Path(row["hrx_path"]).expanduser()

            if not path.is_file():
                # bad path -> move on
                idx = (idx + 1) % len(self.df)
                continue

            with np.load(path) as arr:
                x = arr["inputs"]    # (1, T_in, C, H, W)
                y = arr["targets"]   # (1, T_out, C, H, W)
                mask = arr["masks"]  # (1, H, W)

            if x.shape[1] == 0 or y.shape[1] == 0:
                idx = (idx + 1) % len(self.df)
                continue

            # Strip the leading singleton batch dimension and the target time dimension 
            x = x[0]        # (T_in, C, H, W)
            y = y[0, 0]     # (C, H, W)
            mask = mask[0]  # (H, W)

            x = torch.from_numpy(x).float()
            y = torch.from_numpy(y).float()
            mask = torch.from_numpy(mask).float()

            if self.normalize and self.mean_x is not None and self.std_x is not None:
                x = (x - self.mean_x) / self.std_x              # (T_in,C,H,W)
                y_4d = y.unsqueeze(0)                           # (1,C,H,W)
                y_4d = (y_4d - self.mean_x) / self.std_x
                y = y_4d.squeeze(0)

            return x, y, mask, event_type

        raise RuntimeError(f"Too many malformed samples around index {idx}")