import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.hr_encoder import HREncoder
from models.prithvi_backbone import PrithviBackbone
from models.hr_head import HRHead
from models.hr_prithvi_model import HRPrithviModel
from datasets.hr_extreme_dataset import HRExtremeDataset


def masked_mse(y_hat, y, mask=None):
    """
    Masked MSE loss.

    Args:
        y_hat: (B, C, H, W)
        y:     (B, C, H, W)
        mask:  (B, 1, H, W) or (B, H, W) or None

    Returns:
        scalar loss
    """
    diff = y_hat - y
    if mask is not None:
        # Broadcast mask to match channels if needed
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)  # (B, 1, H, W)
        diff = diff * mask
        denom = mask.sum().clamp_min(1.0)
    else:
        denom = diff.numel()

    mse = (diff ** 2).sum() / denom
    return mse


def main():
    # ---- Paths (adapt to your setup) ----
    data_dir = str(Path("~/scratch/hr_extreme_npz/tiny_train_real").expanduser())   # directory of .npz files with x,y
    config_path = str(Path("~/scratch/Prithvi-WxC/data/config.yaml").expanduser())
    weights_path = str(Path("~/scratch/Prithvi-WxC/data/weights/prithvi.wxc.2300m.v1.pt").expanduser())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Dataset & DataLoader ----
    dataset = HRExtremeDataset(data_dir=data_dir, split="train")
    loader = DataLoader(
        dataset,
        batch_size=1, #2,
        shuffle=True,
        num_workers=0, #4,
        pin_memory=True,
    )

    # ---- Model components ----
    hr_encoder = HREncoder(
        in_channels=69,
        time_steps=2, #3, todo revisit
        c_hidden=128,
        c_hidden2=256,
        c_back_in=256,
        h_in=320,
        w_in=320,
        h_p=80,
        w_p=80,
    ).to(device)

    # For now, we can pass dummy scalers and let backbone load weights with its own shapes.
    # If that still causes issues, you can temporarily set load_weights=False and skip backbone use.
    prithvi_backbone = PrithviBackbone(
        config_path=config_path,
        weights_path=weights_path,
        in_mu=None,
        in_sig=None,
        static_mu=None,
        static_sig=None,
        output_sig=None,
        device=device,
        load_weights=False,   # start with False to avoid state_dict shape issues
    )

    hr_head = HRHead(
        c_back_in=256,
        c_head_hidden=256,
        h_p=80,
        w_p=80,
        h_out=320,
        w_out=320,
        out_channels=69,
    ).to(device)

    model = HRPrithviModel(hr_encoder, prithvi_backbone, hr_head).to(device)

    # ---- Optimizer (encoder + head only; backbone is frozen) ----
    params = list(hr_encoder.parameters()) + list(hr_head.parameters())
    optimizer = torch.optim.Adam(params, lr=1e-4, weight_decay=1e-5)

    # ---- Training loop (very simple) ----
    model.train()
    num_epochs = 1  # start with 1 just to check everything runs

    for epoch in range(num_epochs):
        for step, (x, y, mask) in enumerate(loader):
            # x: (B, T_in, 69, 320, 320)
            # y: (B, 69, 320, 320)
            # mask: (B, 320, 320)

            x = x.to(device)       # (B, T_in, 69, 320, 320)
            y = y.to(device)
            mask = mask.to(device)

            y_hat = model(x)       # model expects (B, T, 69, 320, 320)

            loss = masked_mse(y_hat, y, mask=mask)
            loss.backward()
            optimizer.step()

            if step % 10 == 0:
                print(f"Epoch {epoch} Step {step} Loss {loss.item():.4f}")

            # For a first GPU test, you can break early
            if step == 20:
                break

        print(f"Epoch {epoch} completed.")

    print("Training dry-run finished.")


if __name__ == "__main__":
    main()