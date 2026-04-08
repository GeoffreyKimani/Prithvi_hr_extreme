import yaml
from pathlib import Path

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.hr_encoder import HREncoder
from models.prithvi_backbone import PrithviBackbone
from models.hr_head import HRHead
from models.hr_unet_model import HRUNet
from models.hr_prithvi_model import HRPrithviModel
from datasets.hr_extreme_dataset import HRExtremeDataset
from datasets.hr_extreme_prithvi_dataset import HRExtremeWithPrithviDataset
from training.losses import masked_mse, tail_weighted_mse_all, evaluate, evaluate_rmse_physical, evaluate_rmse_per_variable_phys, exloss_simplified


def save_checkpoint(path, model, optimizer, scheduler, epoch, best_val_loss=None):
    state = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "best_val_loss": best_val_loss,
    }
    torch.save(state, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    if scheduler is not None and "scheduler_state" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state"])
    start_epoch = ckpt.get("epoch", 0) + 1
    best_val = ckpt.get("best_val_loss", None)
    return start_epoch, best_val


def main():
    # --- Load paths and config ---
    paths_cfg = yaml.safe_load(open("configs/paths.yaml"))
    hrx_paths = paths_cfg["data"]["hr_extreme"]

    exp_cfg = yaml.safe_load(open("configs/hrx_prithvi_backbone.yaml"))
    train_cfg = exp_cfg["training"]
    exp_name = train_cfg.get("experiment_name", "unet_prithvi_mse")

    num_epochs = train_cfg.get("num_epochs", 1)
    max_steps = train_cfg.get("max_steps_per_epoch", None)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    stats_path = str(Path(hrx_paths["stats_path"]).expanduser())

    # --- Output / checkpoints separated by experiment ---
    base_ckpt_root = Path(hrx_paths["outputs_root"]).expanduser()
    ckpt_root = base_ckpt_root / exp_name
    ckpt_root.mkdir(parents=True, exist_ok=True)

    best_ckpt_path = ckpt_root / "best_model.pt"
    last_ckpt_path = ckpt_root / "last_model.pt"

    # --- Datasets and loaders ---
    idx_root = Path("index_files")
    train_idx_csv = idx_root / "hrx_prithvi_train.csv"
    val_idx_csv   = idx_root / "hrx_prithvi_val.csv"
    
    if exp_name == "unet_plain":
        # HR-Extreme only (no Prithvi features)
        train_dataset = HRExtremeDataset(
            index_csv=train_idx_csv,
            stats_path=stats_path,
            normalize=True
        )
        val_dataset = HRExtremeDataset(
            index_csv=val_idx_csv,
            stats_path=stats_path,
            normalize=True
        )
    else:
        train_dataset = HRExtremeWithPrithviDataset(
            index_csv=train_idx_csv,
            stats_path=stats_path
        )
        val_dataset = HRExtremeWithPrithviDataset(
            index_csv=val_idx_csv,
            stats_path=stats_path
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=train_cfg["num_workers"],
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=train_cfg["num_workers"],
        pin_memory=True,
    )

    # --- Model ---
    hr_encoder = HREncoder(exp_cfg).to(device)
    hr_head    = HRHead(exp_cfg).to(device)

    if exp_name == "unet_plain":
        # plain HR-only U-Net
        model = HRUNet(hr_encoder, hr_head).to(device)
    else:
        # U-Net + Prithvi fusion
        prithvi_backbone = None  # for now; features are precomputed
        model = HRPrithviModel(hr_encoder, prithvi_backbone, hr_head).to(device)

    # --- Optimizer & scheduler ---
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-5
    )

    # --- Load quantiles only if needed ---
    if "tail" in exp_name or "exloss" in exp_name:
        quantiles_path = Path(
            hrx_paths.get(
                "quantiles_path",
                "~/scratch/prithvi_hr_extreme/stats/hrx_train_quantiles_90.npz",
            )
        ).expanduser()
        q_data = np.load(quantiles_path)
        q_high_np = q_data["q_high"].astype("float32")
        q_high = torch.from_numpy(q_high_np)  # (C,)
    else:
        q_high = None

    # --- Resume from last checkpoint (per experiment) ---
    start_epoch = 0
    best_val_loss = None

    print(f"Experiment: {exp_name}")
    if last_ckpt_path.is_file():
        print(f"Resuming from checkpoint: {last_ckpt_path}")
        start_epoch, best_val_loss = load_checkpoint(
            last_ckpt_path, model, optimizer, scheduler, device=device
        )
    else:
        print("No existing checkpoint, starting from scratch.")

    alpha_tail = train_cfg.get("alpha_tail", 2.0)
    beta_exloss = train_cfg.get("beta_exloss", 1.0)

    for epoch in range(start_epoch, num_epochs):
        model.train()
        epoch_loss_sum = 0.0
        n_batches = 0

        for step, batch in enumerate(train_loader):
            if exp_name == "unet_plain":
                # Dataset returns (x_hr, y, mask)
                x_hr, y, mask = batch
            else:
                # Dataset returns (x_hr, feats_prithvi, y, mask, event_type)
                x_hr, feats_prithvi, y, mask, _event_type = batch
                feats_prithvi = feats_prithvi.to(device)

            x_hr = x_hr.to(device)
            y    = y.to(device)
            mask = mask.to(device)

            optimizer.zero_grad()

            if exp_name == "unet_plain":
                y_hat = model(x_hr)  # ignore feats_prithvi
            else:
                y_hat = model(x_hr, feats_prithvi=feats_prithvi)

            # Select loss
            if exp_name in ("unet_plain", "unet_prithvi_mse"):
                loss = masked_mse(y_hat, y, mask=mask)
            elif exp_name == "unet_prithvi_tail":
                loss = tail_weighted_mse_all(y_hat, y, mask, q_high, alpha=alpha_tail)
            elif exp_name == "unet_prithvi_exloss":
                loss = exloss_simplified(y_hat, y, mask, q_high, beta=beta_exloss)
            else:
                raise ValueError(f"Unknown experiment_name: {exp_name}")

            loss.backward()
            optimizer.step()

            epoch_loss_sum += loss.item()
            n_batches += 1

            if max_steps is not None and step + 1 >= max_steps:
                break

        scheduler.step()

        train_loss_epoch = epoch_loss_sum / max(1, n_batches)
        val_loss_epoch = evaluate(model, val_loader, device)  # keeps your existing RMSE eval

        print(f"Epoch {epoch} Train {train_loss_epoch:.4f}  Val {val_loss_epoch:.4f}")

        # save checkpoints
        if best_val_loss is None or val_loss_epoch < best_val_loss:
            best_val_loss = val_loss_epoch
            save_checkpoint(best_ckpt_path, model, optimizer, scheduler, epoch, best_val_loss)

        save_checkpoint(last_ckpt_path, model, optimizer, scheduler, epoch, best_val_loss)
    print("Training finished.")

    # ---- Final evaluation in physical units ----
    mu_y = val_dataset.mean_x.to(device)
    std_y = val_dataset.std_x.to(device)

    rmse_phys = evaluate_rmse_physical(model, val_loader, device, mu_y, std_y)
    rmse_vars_extreme = evaluate_rmse_per_variable_phys(model, val_loader, device, mu_y, std_y)
    rmse_extreme_mean  = rmse_vars_extreme.mean()
    print(f"Epoch {epoch} \nVal RMSE (phys units): {rmse_phys:.3f} \nPaper RMSE: {rmse_extreme_mean}")

if __name__ == "__main__":
    main()