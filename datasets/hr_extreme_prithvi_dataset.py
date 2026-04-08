from pathlib import Path
from typing import Callable, Optional, Tuple, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class HRExtremePrithviDataset(Dataset):
    """
    Dataset for HR-Extreme where inputs are 160-channel Prithvi features.

    Each *_prithvi.npz is expected to contain at least:
      - feats: (C=160, H=320, W=320)

    Targets (y, mask) can either:
      - live in the same npz (keys 'y', 'mask'), or
      - be loaded from the original HR-Extreme npz via load_target_fn.
    """

    def __init__(
        self,
        prithvi_dir,
        stats_path: Optional[str] = None,
        normalize: bool = True,
        load_target_fn: Optional[Callable[[Path], Tuple[torch.Tensor, Optional[torch.Tensor]]]] = None,
        required_suffix: str = "_prithvi.npz",
    ) -> None:
        self.prithvi_dir = Path(prithvi_dir).expanduser()
        self.normalize = normalize
        self.load_target_fn = load_target_fn
        self.required_suffix = required_suffix

        if not self.prithvi_dir.is_dir():
            raise RuntimeError(f"Prithvi dir does not exist: {self.prithvi_dir}")

        self.files: List[Path] = sorted(
            f for f in self.prithvi_dir.glob(f"*{self.required_suffix}") if f.is_file()
        )
        if not self.files:
            raise RuntimeError(f"No {self.required_suffix} files found in {self.prithvi_dir}")

        # Normalization stats for Prithvi feats
        self.mean_x = None
        self.std_x = None
        if stats_path is not None and normalize:
            stats = np.load(str(Path(stats_path).expanduser()))
            # assume keys "means" and "stds" like your HRRR stats, but length 160
            mean = stats["means"].astype("float32")
            std = stats["stds"].astype("float32")
            std[std == 0] = 1.0
            # [C] -> [C, 1, 1]
            if mean.ndim == 1:
                mean = mean[:, None, None]
            if std.ndim == 1:
                std = std[:, None, None]
            self.mean_x = torch.from_numpy(mean)  # [C,1,1]
            self.std_x = torch.from_numpy(std)    # [C,1,1]

    def __len__(self) -> int:
        return len(self.files)

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.mean_x is None or self.std_x is None:
            return x
        # x: [C,H,W], mean_x/std_x: [C,1,1]
        return (x - self.mean_x) / (self.std_x + 1e-6)

    def __getitem__(self, idx: int):
        npz_path = self.files[idx]
        data = np.load(npz_path, allow_pickle=True)

        if "feats" not in data:
            raise KeyError(f"'feats' not found in {npz_path}")
        feats = data["feats"]  # (160,320,320)
        x = torch.from_numpy(feats).float()

        if self.normalize:
            x = self._normalize(x)

        # Targets / mask
        if "y" in data:
            y = torch.from_numpy(data["y"]).float()
            mask = torch.from_numpy(data["mask"]).float() if "mask" in data else None
        else:
            if self.load_target_fn is None:
                raise RuntimeError(
                    f"No 'y' in {npz_path} and no load_target_fn provided."
                )
            y, mask = self.load_target_fn(npz_path)

        return x, y, mask



class HRExtremeWithPrithviDataset(Dataset):
    """
    Paired dataset: HRR-Extreme + precomputed Prithvi features.

    Each row in index_csv must contain:
      - hrx_path: HR-Extreme npz path
      - prithvi_path: Prithvi feature npz path
    """

    def __init__(self, index_csv: str | Path, stats_path: str | Path):
        self.index_csv = Path(index_csv)
        self.df = pd.read_csv(self.index_csv)

        if not len(self.df):
            raise RuntimeError(f"No rows in index file: {self.index_csv}")

        self.stats_path = Path(stats_path)
        stats = np.load(self.stats_path)
        mean = stats["means"].astype("float32")  # (69,)
        std = stats["stds"].astype("float32")
        std[std == 0] = 1.0
        self.mean_x = torch.from_numpy(mean)[None, :, None, None]  # (1,C,1,1)
        self.std_x = torch.from_numpy(std)[None, :, None, None]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # loop in case we hit a bad file; at most a few retries
        start_idx = idx
        for _ in range(3):
            row = self.df.iloc[idx]
            event_type = row.get("event_type", "")

            hrx_path = Path(row["hrx_path"])
            prithvi_path = Path(row["prithvi_path"])

            with np.load(hrx_path) as arr:
                x = arr["inputs"]   # expect (1, T_in, 69, 320, 320)
                y = arr["targets"]  # expect (1, T_out, 69, 320, 320)
                mask = arr["masks"] # expect (1, 320, 320)

            # sanity check time dims
            if x.shape[1] == 0 or y.shape[1] == 0:
                # bad sample, move to next index
                # print(f"[WARNING] Empty time dimension in {hrx_path}, skipping.")
                idx = (idx + 1) % len(self.df)
                continue

            x = x[0]       # (T_in, 69, 320, 320)
            y = y[0, 0]    # (69, 320, 320)
            mask = mask[0] # (320, 320)

            # load Prithvi features, but guard against corrupted npz
            try:
                with np.load(prithvi_path) as pz:
                    feats = pz["feats"]  # (160, 320, 320)
            except Exception as e:
                # corrupted or empty file, skip this sample
                print(f"{idx+1} [WARNING] Failed to load {prithvi_path}: {e}, skipping.")
                idx = (idx + 1) % len(self.df)
                continue

            x = torch.from_numpy(x).float()
            y = torch.from_numpy(y).float()
            mask = torch.from_numpy(mask).float()
            feats_prithvi = torch.from_numpy(feats).float()

            # normalize HRRR input & target using existing stats
            x = (x - self.mean_x) / self.std_x
            y_4d = y.unsqueeze(0)
            y_4d = (y_4d - self.mean_x) / self.std_x
            y = y_4d.squeeze(0)

            return x, feats_prithvi, y, mask, event_type

        # if we somehow hit 3 bad files in a row, raise
        raise RuntimeError(f"Too many malformed samples around index {start_idx}")
