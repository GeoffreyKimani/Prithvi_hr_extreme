from pathlib import Path

import yaml
import numpy as np
import pandas as pd
import torch

# Adjust to your project paths
PROJ_ROOT = Path("~/scratch/prithvi_hr_extreme").expanduser()
IDX_CSV = PROJ_ROOT / "index_files" / "hrx_prithvi_train.csv"   # full train index
OUT_PATH = PROJ_ROOT / "stats" / "hrx_train_quantiles_90.npz"

N_SAMPLES = 10000       # total pixels per channel to sample (approx)
QUANTILE = 0.90         # 90th percentile
RANDOM_SEED = 42


paths_cfg = yaml.safe_load(open("configs/paths.yaml"))
hrx_paths = paths_cfg["data"]["hr_extreme"]
STATS_PATH = hrx_paths["stats_path"]
STATS_PATH = Path(STATS_PATH).expanduser()

stats = np.load(STATS_PATH)
mean = stats["means"].astype("float32")  # (C,)
std  = stats["stds"].astype("float32")
std[std == 0] = 1.0
mean_t = torch.from_numpy(mean)[None, :, None, None]  # (1,C,1,1)
std_t  = torch.from_numpy(std)[None, :, None, None]


def main():
    np.random.seed(RANDOM_SEED)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(IDX_CSV)
    n_rows = len(df)
    print(f"Train index rows: {n_rows}")

    # Read one file to get channel count
    example_path = Path(df.iloc[0]["hrx_path"])
    with np.load(example_path) as arr:
        x_ex = arr["inputs"]  # (1, T_in, C, H, W)
        C = x_ex.shape[2]
        H, W = x_ex.shape[-2:]

    print(f"Detected C={C}, H={H}, W={W}")

    # Target total samples per channel → samples per file (roughly)
    # Each file has T_in * H * W pixels per channel; we downsample heavily.
    samples_per_channel = [[] for _ in range(C)]

    # Shuffle row order to avoid bias
    idxs = np.arange(n_rows)
    np.random.shuffle(idxs)

    # Simple loop over files, sampling a small subset of pixels from each
    for row_idx in idxs:
        row = df.iloc[row_idx]
        hrx_path = Path(row["hrx_path"])

        try:
            with np.load(hrx_path) as arr:
                x = arr["inputs"]  # (1, T_in, C, H, W)
        except Exception as e:
            print(f"[WARN] Failed to load {hrx_path}: {e}")
            continue

        # collapse time dimension so we treat all timesteps equally
        # x: (1, T_in, C, H, W) -> (T_in, C, H, W)
        x = x[0]  # (T_in, C, H, W)
        x_t = torch.from_numpy(x).float()  # (T_in,C,H,W)

        # normalize channel-wise to match training
        x_norm = (x_t - mean_t) / std_t     # broadcast over T,H,W
        x = x_norm.numpy()

        T = x.shape[0]

        # How many pixels could we sample here?
        total_pixels = T * H * W

        # To keep cost small, sample at most, say, 1000 pixels per file
        n_pick = min(1000, total_pixels)
        if n_pick <= 0:
            continue

        # flatten to (T*H*W, C)
        x_flat = x.transpose(1, 0, 2, 3).reshape(C, -1).T  # (T*H*W, C)

        pick_idxs = np.random.choice(total_pixels, size=n_pick, replace=False)
        picks = x_flat[pick_idxs]  # (n_pick, C)

        # append samples to each channel until we reach N_SAMPLES
        for c in range(C):
            if len(samples_per_channel[c]) < N_SAMPLES:
                samples_per_channel[c].append(picks[:, c])

        # Check if all channels have enough samples
        if all(len(s) >= N_SAMPLES for s in samples_per_channel):
            break

    # Stack and trim samples then compute quantiles
    q_high = np.zeros(C, dtype=np.float32)
    for c in range(C):
        if len(samples_per_channel[c]) == 0:
            print(f"[WARN] No samples collected for channel {c}, setting q_high to 0")
            q_high[c] = 0.0
            continue

        samples_c = np.concatenate(samples_per_channel[c], axis=0)
        if samples_c.shape[0] > N_SAMPLES:
            samples_c = samples_c[:N_SAMPLES]

        q_high[c] = np.quantile(samples_c, QUANTILE)
        print(f"Channel {c}: q_{int(QUANTILE*100)} = {q_high[c]:.4f}")

    np.savez_compressed(OUT_PATH, q_high=q_high, quantile=QUANTILE)
    print(f"Saved quantiles to {OUT_PATH}")


if __name__ == "__main__":
    main()
