import torch
import torch.nn as nn
import torch.nn.functional as F


class HRHead(nn.Module):
    """
    Simple decoder/head that maps backbone features back to
    HR-Extreme resolution and channels.

    Input:  (B, C_back_in, H_p, W_p)
    Output: (B, 69, 320, 320)
    """

    def __init__(
        self,
        c_back_in: int = 256,
        c_head_hidden: int = 256,
        h_p: int = 80,
        w_p: int = 80,
        h_out: int = 320,
        w_out: int = 320,
        out_channels: int = 69,
    ):
        super().__init__()
        self.h_p = h_p
        self.w_p = w_p
        self.h_out = h_out
        self.w_out = w_out

        # Simple conv stack at backbone resolution
        self.conv = nn.Sequential(
            nn.Conv2d(c_back_in, c_head_hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(c_head_hidden),
            nn.GELU(),
            nn.Conv2d(c_head_hidden, c_head_hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(c_head_hidden),
            nn.GELU(),
        )

        # Final 1x1 to 69 channels
        self.to_out = nn.Conv2d(c_head_hidden, out_channels, kernel_size=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (B, C_back_in, H_p, W_p)
        x = self.conv(z)
        x = self.to_out(x)  # (B, 69, H_p, W_p)

        # Upsample from (H_p, W_p) -> (320, 320)
        x = F.interpolate(
            x,
            size=(self.h_out, self.w_out),
            mode="bilinear",
            align_corners=False,
        )
        return x