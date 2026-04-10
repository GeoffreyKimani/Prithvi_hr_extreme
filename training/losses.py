import torch
import torch.nn as nn

from training.utils import unpack_batch


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


def tail_weighted_mse_all(y_hat, y, mask, q_high: torch.Tensor, alpha: float = 2.0):
    """
    Tail-weighted MSE over all channels.

    y_hat, y: (B, C, H, W) normalized
    mask: (B, H, W) or (B,1,H,W) or None
    q_high: (C,) high-quantile thresholds in *normalized space* or same space as y
    alpha: extra weight for pixels where y > q_high[c]
    """
    diff2 = (y_hat - y) ** 2  # (B,C,H,W)

    if mask is not None:
        if mask.dim() == 3:
            mask4 = mask.unsqueeze(1)  # (B,1,H,W)
        else:
            mask4 = mask
    else:
        mask4 = torch.ones_like(diff2[:, :1])

    # broadcast quantiles to (1,C,1,1)
    q = q_high.to(y.device)[None, :, None, None]  # (1,C,1,1)

    # indicator of extremes based on target y
    extreme = (y > q).float()  # (B,C,H,W)
    weights = 1.0 + alpha * extreme

    weighted_diff2 = diff2 * mask4 * weights
    denom = (mask4 * weights).sum()

    if denom <= 0:
        return diff2.mean()

    return weighted_diff2.sum() / denom


def exloss_simplified(y_hat, y, mask, q_high, beta=1.0, eps=1e-6):
    """
    Simplified Exloss-style asymmetric loss.

    - Base: masked MSE.
    - Extra weight when we *underestimate* high-tail targets (y >= q_high).
    - q_high: (C,) tensor in *normalized* units, matching y's channels.
    - beta: strength of the extra penalty (beta=0 => pure MSE).
    """
    # y_hat, y: (B, C, H, W); mask: (B, H, W) or (B,1,H,W)
    if mask.dim() == 3:
        mask4 = mask.unsqueeze(1)  # (B,1,H,W)
    else:
        mask4 = mask

    B, C, H, W = y.shape

    # Broadcast q_high to (1,C,1,1)
    qh = q_high.to(y.device).view(1, C, 1, 1)

    # Base squared error
    se = (y_hat - y) ** 2  # (B,C,H,W)

    # Condition: target in high tail, but prediction lower than target (underestimate)
    high_target = y >= qh
    under = y_hat < y
    extreme_under = high_target & under  # (B,C,H,W)

    # Relative error (for scaling)
    rel_err = torch.abs(y_hat - y) / torch.clamp(torch.abs(y), min=eps)

    # Scaling factor: 1 + beta * rel_err for underestimation of high tail, else 1
    scale = torch.where(extreme_under, 1.0 + beta * rel_err, torch.ones_like(se))

    loss = se * scale * mask4
    denom = torch.clamp(mask4.sum(), min=1.0)
    return loss.sum() / denom


# Evaluation metrics section
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in loader:
            x_hr, feats_prithvi, y, mask, _event_type = unpack_batch(batch, device)

            if feats_prithvi is None:
                y_hat = model(x_hr)
            else:
                y_hat = model(x_hr, feats_prithvi=feats_prithvi)

            loss = masked_mse(y_hat, y, mask=mask)

            total_loss += loss.item()
            n_batches += 1

    return total_loss / max(n_batches, 1)


def evaluate_rmse_per_variable_phys(model, loader, device, mu, std):
    model.eval()
    C = mu.shape[1]  # 69

    sq_err_sum = torch.zeros(C, device=device)
    count = torch.zeros(C, device=device)

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

            # de-normalize to physical units
            m = mu.to(device)       # (1,C,1,1)
            s = std.to(device)
            y_hat_phys = y_hat * s + m
            y_phys     = y * s + m

            if mask.dim() == 3:
                mask4 = mask.unsqueeze(1)            # (B,1,H,W)
            else:
                mask4 = mask

            diff2 = (y_hat_phys - y_phys) ** 2       # (B,C,H,W)
            diff2 = diff2 * mask4                    # apply mask

            # sum over batch and spatial dims
            sq_err_sum += diff2.sum(dim=(0, 2, 3))
            # count valid pixels per channel
            count += (mask4.sum(dim=(0, 2, 3)))

    # avoid division by zero
    count = torch.clamp(count, min=1.0)
    mse = sq_err_sum / count                         # (C,)
    rmse = torch.sqrt(mse)                           # (C,)

    return rmse.cpu().numpy()  # per-variable RMSE in physical units


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