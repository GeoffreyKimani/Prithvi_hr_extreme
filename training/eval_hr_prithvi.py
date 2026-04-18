import yaml
from pathlib import Path
from collections import defaultdict

import torch
from torch.utils.data import DataLoader
import numpy as np

from datasets.hr_extreme_dataset import HRExtremeDataset
from datasets.hr_extreme_prithvi_dataset import HRExtremeWithPrithviDataset
from models.hr_encoder import HREncoder
from models.hr_unet_model import HRUNet
from models.hr_head import HRHead
from models.hr_prithvi_model import HRPrithviModel

from training.train_hr_prithvi import (
    masked_mse,
    evaluate_rmse_physical,
    evaluate_rmse_per_variable_phys,
)

from training.utils import unpack_batch

def build_model(exp_cfg, model_paths, device, exp_name: str):
    hr_encoder = HREncoder(exp_cfg).to(device)
    hr_head    = HRHead(exp_cfg).to(device)

    use_prithvi = "prithvi" in exp_name

    if not use_prithvi:
        model = HRUNet(hr_encoder, hr_head).to(device)
    else:
        prithvi_backbone = None
        model = HRPrithviModel(hr_encoder, prithvi_backbone, hr_head).to(device)

    return model


def load_checkpoint(model, ckpt_path, device):
    ckpt_path = Path(ckpt_path).expanduser()
    state = torch.load(ckpt_path, map_location=device)
    if "model_state" in state:
        state = state["model_state"]
    model.load_state_dict(state, strict=True)
    return model


def evaluate_test(model, loader, device, mu_y, std_y):
    model.eval()
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in loader:
            x_hr, feats_prithvi, y, mask, _event_type = unpack_batch(batch, device)

            x_hr = x_hr.to(device)
            y = y.to(device)
            mask = mask.to(device)

            if feats_prithvi is None:
                y_hat = model(x_hr)
            else:
                feats_prithvi = feats_prithvi.to(device)
                y_hat = model(x_hr, feats_prithvi=feats_prithvi)

            loss = masked_mse(y_hat, y, mask=mask)
            total_loss += loss.item()
            n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)

    # RMSE metrics in physical units
    rmse_phys = evaluate_rmse_physical(model, loader, device, mu_y, std_y)
    rmse_per_var = evaluate_rmse_per_variable_phys(model, loader, device, mu_y, std_y)
    rmse_extreme_mean = rmse_per_var.mean()

    return avg_loss, rmse_phys, rmse_extreme_mean, rmse_per_var


def evaluate_rmse_per_variable_per_event(model, loader, device, mu, std, out_path):
    """
    Compute per-variable RMSE in physical units, grouped by event_type.
    Works with both datasets (with and without Prithvi features).

    Saves a npz file with:
      - event_types: list of unique event types (strings)
      - rmse: array of shape [N_event_types, C] with per-variable RMSE
    """
    model.eval()
    C = mu.shape[1]  # 69

    sq_err_sum = defaultdict(lambda: torch.zeros(C, device=device))
    count = defaultdict(lambda: torch.zeros(C, device=device))

    with torch.no_grad():
        for batch in loader:
            x_hr, feats_prithvi, y, mask, event_type = unpack_batch(batch, device)

            if feats_prithvi is None:
                y_hat = model(x_hr)
            else:
                y_hat = model(x_hr, feats_prithvi=feats_prithvi)

            # de-normalize to physical units
            m = mu.to(device)   # (1,C,1,1)
            s = std.to(device)
            y_hat_phys = y_hat * s + m
            y_phys     = y * s + m

            if mask.dim() == 3:
                mask4 = mask.unsqueeze(1)  # (B,1,H,W)
            else:
                mask4 = mask

            diff2 = (y_hat_phys - y_phys) ** 2   # (B,C,H,W)
            diff2 = diff2 * mask4

            B = y.shape[0]
            # if event_type is None:
            #     event_type = ["all"] * B

            for b in range(B):
                et = event_type[b]
                # sum over spatial dims
                se_sum_b = diff2[b].sum(dim=(1, 2))       # (C,)
                cnt_b    = mask4[b].sum(dim=(1, 2))       # (1,H,W) -> (1,) per channel

                sq_err_sum[et] += se_sum_b
                count[et]      += cnt_b

    # assemble results
    event_types = sorted(sq_err_sum.keys())
    rmse_list = []
    for et in event_types:
        se_sum = sq_err_sum[et]
        cnt    = torch.clamp(count[et], min=1.0)
        mse    = se_sum / cnt
        rmse   = torch.sqrt(mse)   # (C,)
        rmse_list.append(rmse.cpu().numpy())

    rmse_arr = np.stack(rmse_list, axis=0)  # [N_event_types, C]

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        event_types=np.array(event_types, dtype=object),
        rmse=rmse_arr,
    )
    print(f"Saved per-variable RMSE per event type to {out_path}")


def evaluate_rmse_per_variable_per_sample(model, loader, device, mu, std, out_path):
    """
    Compute per-sample per-variable RMSE in physical units.

    Saves a npz file with:
    - sample_indices: [N]
    - event_types: [N]
    - rmse: [N, C]
    - mean_rmse_all_vars: [N]
    """
    model.eval()
    records_rmse = []
    records_event_type = []
    records_sample_idx = []

    sample_idx = 0

    with torch.no_grad():
        for batch in loader:
            x_hr, feats_prithvi, y, mask, event_type = unpack_batch(batch, device)

            if feats_prithvi is None:
                y_hat = model(x_hr)
            else:
                y_hat = model(x_hr, feats_prithvi=feats_prithvi)

            m = mu.to(device)
            s = std.to(device)
            y_hat_phys = y_hat * s + m
            y_phys = y * s + m

            if mask.dim() == 3:
                mask4 = mask.unsqueeze(1)
            else:
                mask4 = mask

            diff2 = (y_hat_phys - y_phys) ** 2
            diff2 = diff2 * mask4

            B = y.shape[0]

            for b in range(B):
                se_sum_b = diff2[b].sum(dim=(1, 2))              # (C,)
                cnt_b = mask4[b].sum(dim=(1, 2)).clamp(min=1.0)  # (1,) broadcastable to (C,)
                mse_b = se_sum_b / cnt_b
                rmse_b = torch.sqrt(mse_b)                       # (C,)

                records_rmse.append(rmse_b.cpu().numpy())
                records_event_type.append(event_type[b])
                records_sample_idx.append(sample_idx)
                sample_idx += 1

    rmse_arr = np.stack(records_rmse, axis=0)  # [N, C]
    mean_rmse_all = rmse_arr.mean(axis=1)      # [N]

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        sample_indices=np.array(records_sample_idx),
        event_types=np.array(records_event_type, dtype=object),
        rmse=rmse_arr,
        mean_rmse_all_vars=mean_rmse_all,
    )

    print(f"Saved per-sample per-variable RMSE to {out_path}")


def main():
    # Load paths
    paths_cfg = yaml.safe_load(open("configs/paths.yaml"))
    hrx_paths = paths_cfg["data"]["hr_extreme"]
    model_paths = paths_cfg["model"]

    # Test index CSV
    idx_root = Path("index_files")
    test_idx_csv = idx_root / "hrx_prithvi_test.csv"

    # Stats for HRRR normalization
    stats_path = str(Path(hrx_paths["stats_path"]).expanduser())

    # Load experiment config
    exp_cfg = yaml.safe_load(open("configs/hrx_prithvi_backbone.yaml"))
    eval_cfg = exp_cfg.get("evaluation", {})
    train_cfg = exp_cfg.get("training", {})
    exp_name = train_cfg.get("experiment_name", "unet_plain_mse")

    use_prithvi = "prithvi" in exp_name

    if exp_name.endswith("_mse"):
        loss_name = "mse"
    elif exp_name.endswith("_tail"):
        loss_name = "tail"
    elif exp_name.endswith("_exloss"):
        loss_name = "exloss"
    else:
        raise ValueError(f"Could not infer loss type from experiment_name: {exp_name}")

    batch_size = eval_cfg.get("batch_size", 8)
    num_workers = eval_cfg.get("num_workers", 4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset & DataLoader
    use_prithvi = "prithvi" in exp_name

    if not use_prithvi:
        test_dataset = HRExtremeDataset(
            index_csv=test_idx_csv,
            stats_path=stats_path,
            normalize=True,
        )
    else:
        test_dataset = HRExtremeWithPrithviDataset(
            index_csv=test_idx_csv,
            stats_path=stats_path,
        )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Build model and load checkpoint
    model = build_model(exp_cfg, model_paths, device, exp_name)

    # Decide checkpoint path: you can store it in config or pass via CLI; here assume config
    ckpt_root_base = Path(hrx_paths["outputs_root"]).expanduser()
    ckpt_root = ckpt_root_base / exp_name
    ckpt_name = eval_cfg.get("checkpoint_name", "best_model.pt")
    ckpt_path = ckpt_root / ckpt_name

    model = load_checkpoint(model, ckpt_path, device)

    # Prepare normalization stats for y
    mu_y = test_dataset.mean_x.to(device)
    std_y = test_dataset.std_x.to(device)

    avg_loss, rmse_phys, rmse_extreme_mean, rmse_per_var = evaluate_test(
        model, test_loader, device, mu_y, std_y
    )

    print(f"Evaluation results on {exp_name} checkpoint {ckpt_name}:")
    print(f"\tTest masked MSE (normalized): {avg_loss:.4f}")
    print(f"\tTest RMSE (physical units): {rmse_phys:.3f}")
    print(f"\tTest RMSE mean over variables (HR-Extreme style): {rmse_extreme_mean:.3f}")

    # Optionally save per-variable RMSE to a .npy or .csv for later analysis
    out_dir = ckpt_root / "eval_test"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "rmse_per_variable_test.npy", rmse_per_var)

    # Per-event-type RMSE
    # if exp_name != "unet_plain":
    per_event_out = out_dir / "rmse_per_variable_per_event_test.npz"
    evaluate_rmse_per_variable_per_event(
        model, test_loader, device, mu_y, std_y, per_event_out
    )

    # Per-sample RMSE (for paired statistical testing)
    per_sample_out = out_dir / "rmse_per_variable_per_sample_test.npz"
    evaluate_rmse_per_variable_per_sample(
        model, test_loader, device, mu_y, std_y, per_sample_out
    )


if __name__ == "__main__":
    main()