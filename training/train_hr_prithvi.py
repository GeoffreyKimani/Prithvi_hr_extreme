import yaml
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.hr_encoder import HREncoder
from models.prithvi_backbone import PrithviBackbone
from models.hr_head import HRHead
from models.hr_prithvi_model import HRPrithviModel
from datasets.hr_extreme_dataset import HRExtremeDataset


# def masked_mse(y_hat, y, mask=None):
#     """
#     Masked MSE loss.

#     Args:
#         y_hat: (B, C, H, W)
#         y:     (B, C, H, W)
#         mask:  (B, 1, H, W) or (B, H, W) or None

#     Returns:
#         scalar loss
#     """
#     diff = y_hat - y
#     if mask is not None:
#         # Broadcast mask to match channels if needed
#         if mask.dim() == 3:
#             mask = mask.unsqueeze(1)  # (B, 1, H, W)
#         diff = diff * mask
#         denom = mask.sum().clamp_min(1.0)
#     else:
#         denom = diff.numel()

#     mse = (diff ** 2).sum() / denom
#     return mse

def masked_mse(y_hat, y, mask=None):
    """
    y_hat, y: (B, C, H, W)
    mask: (B, H, W) or (B, 1, H, W) or None
    """
    diff2 = (y_hat - y) ** 2  # (B,C,H,W)

    if mask is not None:
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)          # (B,1,H,W)
        # mask is 0/1; broadcast over channels
        diff2 = diff2 * mask                 # (B,C,H,W)
        denom = mask.sum() * y.shape[1]      # (#valid pixels) * C
    else:
        denom = diff2.numel()

    if denom == 0:
        return diff2.mean()  # edge case

    return diff2.sum() / denom


def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for x, y, mask in loader:
            x = x.to(device)
            y = y.to(device)
            mask = mask.to(device)

            y_hat = model(x)
            loss = masked_mse(y_hat, y, mask=mask)

            total_loss += loss.item()
            n_batches += 1

    return total_loss / max(n_batches, 1)


def evaluate_rmse_physical(model, loader, device, mu, std):
    """
    Evaluate RMSE in physical units by un-normalizing predictions and targets.

    Args:
        model: the HRPrithviModel to evaluate
        loader: DataLoader for the validation set
        device: torch device
        std_y: (C,) array of standard deviations for each channel, used to un-normalize predictions and targets
    """
    model.eval()
    squared_error_sum = 0.0
    n_pixels = 0

    with torch.no_grad():
        for x, y, mask in loader:
            x = x.to(device)
            y = y.to(device)    # normalized targets
            mask = mask.to(device)

            y_hat = model(x)  # normalized predictions

            # Un-normalize predictions and targets
            y_hat_phys = y_hat * std + mu
            y_phys = y * std + mu

            # Compute masked squared error in physical units
            se = (y_hat_phys - y_phys) ** 2  # (B,C,H,W)
            
            if mask is not None:
                if mask.dim() == 3:
                    mask = mask.unsqueeze(1)  # (B,1,H,W)
                se = se * mask
                n_pixels += mask.sum().item() * y.shape[1]  # valid pixels * channels
            else:
                n_pixels += se.numel()

            squared_error_sum += se.sum().item()

    rmse = (squared_error_sum / max(n_pixels, 1)) ** 0.5
    return rmse


def main():
    # Load paths
    paths_cfg = yaml.safe_load(open("configs/paths.yaml"))
    data_paths = paths_cfg["data"]["hr_extreme"]
    model_paths = paths_cfg["model"]

    train_dir = str(Path(data_paths["train"]).expanduser())  # start with tiny_train
    val_dir = str(Path(data_paths["val"]).expanduser())      # start with tiny_val
    stats_path = str(Path(data_paths["stats_path"]).expanduser())
    config_path = str(Path(model_paths["prithvi_config"]).expanduser())
    weights_path = str(Path(model_paths["prithvi_weights_base"]).expanduser())

    # Load model/training hyperparams
    exp_cfg = yaml.safe_load(open("configs/hrx_prithvi_backbone.yaml"))
    enc_cfg = exp_cfg["encoder"]
    train_cfg = exp_cfg["training"]
    num_epochs = train_cfg.get("num_epochs", 1)
    max_steps = train_cfg.get("max_steps_per_epoch", None)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Dataset & DataLoader ----
    train_dataset = HRExtremeDataset(
        data_dir=train_dir,
        stats_path=stats_path,
        normalize=True,
        split="train",
    )
    val_dataset = HRExtremeDataset(
        data_dir=val_dir,
        stats_path=stats_path,
        normalize=True,
        split="val",
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

    # ---- Model components ----
    hr_encoder = HREncoder(
        in_channels=exp_cfg["hr_extreme"]["in_channels"],
        time_steps=exp_cfg["hr_extreme"]["time_steps"],
        c_hidden=enc_cfg["c_hidden"],
        c_hidden2=enc_cfg["c_hidden2"],
        c_back_in=enc_cfg["c_back_in"],
        h_in=exp_cfg["hr_extreme"]["height"],
        w_in=exp_cfg["hr_extreme"]["width"],
        h_p=enc_cfg["h_p"],
        w_p=enc_cfg["w_p"],
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
    for epoch in range(num_epochs):
        model.train()
        epoch_loss_sum = 0.0
        n_batches = 0

        for step, (x, y, mask) in enumerate(train_loader):
            x = x.to(device)       # (B, T_in, 69, 320, 320)
            y = y.to(device)        # (B, 69, 320, 320)
            mask = mask.to(device)

            optimizer.zero_grad()
            y_hat = model(x)

            loss = masked_mse(y_hat, y, mask=mask)
            
            # if epoch == 0 and step == 0:
            #     with torch.no_grad():
            #         raw = ((y_hat - y) ** 2).mean().item()
            #     print(f"DEBUG step {step} Loss {loss.item():.3f} raw_MSE {raw:.3f}")
            #     print("DEBUG pred (mean/std):", y_hat.mean().item(), y_hat.std().item())
            #     print("DEBUG y (mean/std):", y.mean().item(), y.std().item())
            #     print("DEBUG raw loss:", ((y_hat - y) ** 2).mean().item())

            # if epoch == 0 and step % 20 == 0:
            #     print("DEBUG step", step, "mask sum:", mask.sum(dim=(-1,-2)).mean().item())

            loss.backward()

            # grad = next(model.parameters()).grad
            # if epoch == 0 and step == 0:
            #     print("grad mean:", grad.abs().mean().item())

            optimizer.step()

            if max_steps is not None and step >= max_steps:
                break

            epoch_loss_sum += loss.item()
            n_batches += 1

            if step % 50 == 0:
                print(f"Epoch {epoch} Step {step} Loss {loss.item():.4f}")

        train_loss_epoch = epoch_loss_sum / n_batches
        val_loss_epoch = evaluate(model, val_loader, device)
        print(f"Epoch {epoch} Train {train_loss_epoch:.4f}  Val {val_loss_epoch:.4f}\n")        
    print("Training dry-run finished.")

    # ---- Final evaluation in physical units ----
    mu_y = val_dataset.mean_x.to(device)
    std_y = val_dataset.std_x.to(device)

    rmse_phys = evaluate_rmse_physical(model, val_loader, device, mu_y, std_y)
    print(f"Epoch {epoch} Val RMSE (phys units): {rmse_phys:.3f}")

if __name__ == "__main__":
    main()