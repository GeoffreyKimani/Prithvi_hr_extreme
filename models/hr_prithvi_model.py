import torch
import torch.nn as nn

from .hr_encoder import HREncoder
from .prithvi_backbone import PrithviBackbone
from .hr_head import HRHead


class HRPrithviModel(nn.Module):
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

        # --- Fusion parameters ---
        # These must match your HR encoder bottleneck shape.
        # Example: if bottleneck is 1/16 of 320 -> 20x20.
        self.prithvi_in_channels = 160
        self.prithvi_bottleneck_channels = 256   # C_p (tunable)
        self.fused_channels = 256               # C_fused (tunable)

        # Downsample Prithvi features from 320x320 to H_b x W_b (here assume 20x20)
        # Using 4 convs with stride 2 (320 -> 160 -> 80 -> 40 -> 20).
        self.prithvi_adapter = nn.Sequential(
            nn.Conv2d(self.prithvi_in_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, self.prithvi_bottleneck_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

        # After concat(bottleneck, prithvi_bottleneck) along channels:
        #   [B, C_enc + C_p, H_b, W_b] -> [B, fused_channels, H_b, W_b]
        # We don't know C_enc here; will infer dynamically in forward.
        self.fusion_conv = nn.Conv2d(
            in_channels=256 + self.prithvi_bottleneck_channels,  # 512
            out_channels=self.fused_channels,                    # 256
            kernel_size=3,
            padding=1,
        )
        self.fusion_relu = nn.ReLU(inplace=True)

    def forward(
    self,
    x_hr: torch.Tensor,
    feats_prithvi: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        x_hr: (B, T, 69, 320, 320)
        feats_prithvi: optional (B, 160, 320, 320) precomputed Prithvi features.
        """
        bottleneck, skips = self.hr_encoder(x_hr)  # (B, C_enc, H_b, W_b)

        # If no Prithvi features provided, behave like plain HRRR UNet
        if feats_prithvi is None:
            z_feat = self.prithvi_backbone.forward_from_features(bottleneck)
            return self.hr_head(z_feat, skips)

        # --- Prithvi adapter path ---
        # feats_prithvi: [B,160,320,320] -> [B,C_p,H_b,W_b]
        prithvi_bottleneck = self.prithvi_adapter(feats_prithvi)

        # --- Fusion at bottleneck ---
        # Concatenate along channels
        fused = torch.cat([bottleneck, prithvi_bottleneck], dim=1)  # [B, C_enc + C_p, H_b, W_b]

        # Ensure fusion_conv has correct in_channels (once)
        if self.fusion_conv.in_channels != fused.shape[1]:
            # Recreate fusion_conv with proper input channels, keep out_channels=fused_channels
            self.fusion_conv = nn.Conv2d(
                in_channels=fused.shape[1],
                out_channels=self.fused_channels,
                kernel_size=3,
                padding=1,
            ).to(fused.device)

        fused = self.fusion_conv(fused)
        fused = self.fusion_relu(fused)

        # Decode fused bottleneck
        y_hat = self.hr_head(fused, skips)
        return y_hat
