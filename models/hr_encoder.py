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

    def __init__(
        self,
        in_channels: int = 69,
        time_steps: int = 3,
        c_hidden: int = 128,
        c_hidden2: int = 256,
        c_back_in: int = 256,
        h_in: int = 320,
        w_in: int = 320,
        h_p: int = 80,
        w_p: int = 80,
    ):
        """
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

        assert time_steps >= 1, "time_steps must be >= 1"
        self.in_channels = in_channels
        self.time_steps = time_steps
        self.h_in = h_in
        self.w_in = w_in
        self.h_p = h_p
        self.w_p = w_p

        # ---- Temporal fusion via 3D convolution ----
        # We treat the input as (B, C_hr, T, H, W) and use a kernel that spans
        # the entire temporal dimension to collapse T -> 1.
        #
        # Kernel size (time_steps, 1, 1) means:
        # - Combine information across the T input frames
        # - Do not mix spatial neighbors yet
        self.temporal_conv = nn.Conv3d(
            in_channels=in_channels,
            out_channels=c_hidden,
            kernel_size=(time_steps, 1, 1),
            stride=(1, 1, 1),
            padding=(0, 0, 0),
            bias=True,
        )

        # Normalization + nonlinearity after temporal fusion
        self.temporal_norm = nn.BatchNorm3d(c_hidden)
        self.temporal_act = nn.GELU()

        # ---- Spatial encoding / downsampling blocks ----
        # After temporal fusion, we squeeze the time dimension and treat the data
        # as a 2D feature map (B, C, H, W). We then downsample spatially twice
        # using stride-2 convolutions to go from H_in,W_in -> H_p,W_p.
        #
        # Block 1: (H,W) 320x320 -> 160x160
        self.spatial_block1 = nn.Sequential(
            nn.Conv2d(c_hidden, c_hidden2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(c_hidden2),
            nn.GELU(),
        )

        # Block 2: (H,W) 160x160 -> 80x80
        self.spatial_block2 = nn.Sequential(
            nn.Conv2d(c_hidden2, c_hidden2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(c_hidden2),
            nn.GELU(),
        )

        # Optional sanity check: confirm that 320 -> 80 given two stride-2 blocks.
        # This also allows you to swap different input sizes without silent mismatch.
        with torch.no_grad():
            dummy = torch.zeros(1, c_hidden, h_in, w_in)
            dummy = self.spatial_block1(dummy)
            dummy = self.spatial_block2(dummy)
            h_out, w_out = dummy.shape[-2], dummy.shape[-1]
            assert h_out == h_p and w_out == w_p, (
                f"Downsampling produced ({h_out}, {w_out}) "
                f"but expected ({h_p}, {w_p}). "
                "Adjust h_p, w_p or the spatial blocks."
            )

        # ---- Projection to backbone feature channels ----
        # Final 1x1 conv maps to C_back_in, which will be the channel dimension
        # used by the Prithvi backbone adapter/tokenizer.
        self.to_backbone = nn.Conv2d(
            c_hidden2,
            c_back_in,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: HR-Extreme input tensor of shape (B, T, C_hr, H_hr, W_hr).
               - B: batch size
               - T: time_steps (e.g., 3)
               - C_hr: in_channels (69)
               - H_hr, W_hr: input spatial size (e.g., 320 x 320)

        Returns:
            z_in: Tensor of shape (B, C_back_in, H_p, W_p),
                  e.g. (B, 256, 80, 80), ready for the backbone.
        """
        # Sanity check shapes
        assert x.dim() == 5, f"Expected input of shape (B, T, C, H, W), got {x.shape}"
        b, t, c, h, w = x.shape
        assert t == self.time_steps, f"Expected time_steps={self.time_steps}, got {t}"
        assert c == self.in_channels, f"Expected in_channels={self.in_channels}, got {c}"
        assert h == self.h_in and w == self.w_in, (
            f"Expected spatial size ({self.h_in}, {self.w_in}), got ({h}, {w})"
        )

        # Rearrange to (B, C_hr, T, H, W) for Conv3d
        x_3d = x.permute(0, 2, 1, 3, 4).contiguous()

        # Temporal fusion: (B, C_hr, T, H, W) -> (B, c_hidden, 1, H, W)
        x_3d = self.temporal_conv(x_3d)
        x_3d = self.temporal_norm(x_3d)
        x_3d = self.temporal_act(x_3d)

        # Remove the singleton time dimension: (B, c_hidden, 1, H, W) -> (B, c_hidden, H, W)
        x_2d = x_3d.squeeze(2)

        # Spatial encoding / downsampling:
        # (B, c_hidden, H, W) -> (B, c_hidden2, H/2, W/2) -> (B, c_hidden2, H/4, W/4)
        x_2d = self.spatial_block1(x_2d)
        x_2d = self.spatial_block2(x_2d)

        # Project to backbone input channels: (B, c_hidden2, H_p, W_p) -> (B, c_back_in, H_p, W_p)
        z_in = self.to_backbone(x_2d)

        return z_in