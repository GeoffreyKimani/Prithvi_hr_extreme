import torch
import torch.nn as nn

from .hr_encoder import HREncoder
from .prithvi_backbone import PrithviBackbone
from .hr_head import HRHead


class HRPrithviModel(nn.Module):
    """
    Full model:
      HR-Extreme encoder  -> Prithvi backbone (frozen, placeholder) -> HR head.

    This is the network you will train on HR-Extreme:
      x_hr (B, T, 69, 320, 320) -> y_hat (B, 69, 320, 320)
    """

    def __init__(
        self,
        hr_encoder: HREncoder,
        prithvi_backbone: PrithviBackbone,
        hr_head: HRHead,
    ):
        super().__init__()
        self.hr_encoder = hr_encoder
        self.prithvi_backbone = prithvi_backbone
        self.hr_head = hr_head

    def forward(self, x_hr: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_hr: HR-Extreme input, shape (B, T, 69, 320, 320).

        Returns:
            y_hat: Predicted HR-Extreme field at t+1, shape (B, 69, 320, 320).
        """
        z_in = self.hr_encoder(x_hr)                       # (B, C_back_in, H_p, W_p)
        z_feat = self.prithvi_backbone.forward_from_features(z_in)
        y_hat = self.hr_head(z_feat)                       # (B, 69, 320, 320)
        return y_hat
