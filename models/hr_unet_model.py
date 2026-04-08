import torch
import torch.nn as nn


class HRUNet(nn.Module):
    """
    HR-only U-Net style forecaster for HR-Extreme tiles.

    - Uses the same HR encoder and decoder (HRHead) as the Prithvi model.
    - Does NOT use any Prithvi features or backbone.
    - Input:  x_hr  of shape (B, T_in, C, H, W) in normalized units.
    - Output: y_hat of shape (B, C, H, W) in normalized units.
    """

    def __init__(self, hr_encoder: nn.Module, hr_head: nn.Module):
        super().__init__()
        self.hr_encoder = hr_encoder
        self.hr_head = hr_head

    def forward(self, x_hr: torch.Tensor) -> torch.Tensor:
        # encoder is expected to accept (B, T_in, C, H, W)
        bottleneck, skips = self.hr_encoder(x_hr)   # bottleneck: (B, Cb, Hb, Wb)
        y_hat = self.hr_head(bottleneck, skips)     # (B, C, H, W)
        return y_hat
