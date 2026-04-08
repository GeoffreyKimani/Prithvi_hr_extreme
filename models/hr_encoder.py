import torch
import torch.nn as nn
import torch.nn.functional as F


class HREncoder(nn.Module):
    """
    HR-Extreme encoder / adapter.

    Goal:
    - Take a short HR-Extreme context window x of shape (B, T, C_hr, H_hr, W_hr),
      typically T=3, C_hr=69, H_hr=W_hr=320.
    - Fuse the temporal dimension.
    - Encode and modestly downsample in space.
    - Project to a feature map z_in that will be fed into the Prithvi backbone.

    Output:
    - z_in: (B, C_back_in, H_p, W_p), e.g. (B, 256, 80, 80).
    """

    def __init__(self, cfg):
        """
        The args below are specified from the config file to keep code clean
        Args:
            in_channels: Number of HR-Extreme physical variables (69).
            time_steps: Length of temporal context window (e.g. 3: t-2, t-1, t).
            c_hidden: Channel width after temporal fusion.
            c_hidden2: Channel width after first downsampling block.
            c_back_in: Channel width expected by the downstream backbone adapter.
            h_in, w_in: Input spatial size (320 x 320).
            h_p, w_p: Output spatial size after downsampling (e.g. 80 x 80).
        """
        super().__init__()

        c1 = cfg['encoder']['c_down1']
        c2 = cfg['encoder']['c_down2']
        c3 = cfg['encoder']['c_down3']
        c4 = cfg['encoder']['c_down4']
        cb = cfg['encoder']['c_bottleneck']

        in_channels = cfg["hr_extreme"]["in_channels"]      # 69
        time_steps  = cfg["hr_extreme"]["time_steps"]       # 2

        # ---- Temporal fusion via 3D convolution ----
        # x: (B,T,C,H,W) -> (B,C,T,H,W) -> Conv3d over time -> (B,c1,1,H,W)
        self.temporal_conv = nn.Conv3d(
            in_channels=in_channels,
            out_channels=c1,
            kernel_size=(time_steps, 1, 1),
            stride=(1, 1, 1),
            padding=(0, 0, 0),
            bias=True,
        )

        # Normalization + nonlinearity after temporal fusion
        self.temporal_norm = nn.BatchNorm3d(c1)
        self.temporal_act = nn.GELU()

        # ---- Spatial encoding / downsampling blocks ----
        # After temporal fusion we treat data as (B, c1, H, W) and downsample 4 times.
        
        # level 1: (H,W) 320x320 -> 160x160
        self.down1 = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1, c1, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool1 = nn.MaxPool2d(2)

        # level 2: (H,W) 160x160 -> 80x80
        self.down2 = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c2, c2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool2 = nn.MaxPool2d(2)

        # level 3: 80x80 -> 40x40
        self.down3 = nn.Sequential(
            nn.Conv2d(c2, c3, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c3, c3, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool3 = nn.MaxPool2d(2)

        # level 4: 40x40 -> 20x20
        self.down4 = nn.Sequential(
            nn.Conv2d(c3, c4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c4, c4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool4 = nn.MaxPool2d(2)

        # bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(c4, cb, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(cb, cb, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.out_channels = cb

    def forward(self, x):
        # x: (B, T, C_hr, H, W)
        B, T, C, H, W = x.shape

        # (B, C_hr, T, H, W)
        x = x.permute(0, 2, 1, 3, 4)

        # temporal fusion: (B, 69, T, H, W) -> (B, c1, 1, H, W)
        x = self.temporal_conv(x)
        x = self.temporal_norm(x)
        x = self.temporal_act(x)
        x = x.squeeze(2)          # (B, c1, H, W)

        # spatial encoder with skips
        x1 = self.down1(x)                     # 320x320
        x2 = self.down2(self.pool1(x1))        # 160x160 -> 80x80
        x3 = self.down3(self.pool2(x2))        # 80x80 -> 40x40
        x4 = self.down4(self.pool3(x3))        # 40x40 -> 20x20
        bottleneck = self.bottleneck(self.pool4(x4))

        return bottleneck, (x1, x2, x3, x4)