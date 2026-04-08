import torch
import torch.nn as nn
import torch.nn.functional as F


class HRHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        c1 = cfg["encoder"]["c_down1"]
        c2 = cfg["encoder"]["c_down2"]
        c3 = cfg["encoder"]["c_down3"]
        c4 = cfg["encoder"]["c_down4"]
        cb = cfg["encoder"]["c_bottleneck"]
        c_up3 = cfg["head"]["c_up3"]
        c_up2 = cfg["head"]["c_up2"]
        c_up1 = cfg["head"]["c_up1"]
        c_out = cfg["hr_extreme"]["in_channels"]  # 69

        # up from bottleneck 20x20 -> 40x40
        self.up4 = nn.ConvTranspose2d(cb, c4, kernel_size=2, stride=2)
        self.dec4 = nn.Sequential(
            nn.Conv2d(c4 + c4, c_up3, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_up3, c_up3, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # up from bottleneck 40x40 -> 80x80
        self.up3 = nn.ConvTranspose2d(c4, c3, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(c3 + c3, c_up2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_up2, c_up2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # 80x80 -> 160x160
        self.up2 = nn.ConvTranspose2d(c_up2, c2, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(c2 + c2, c_up1, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_up1, c_up1, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # 160x160 -> 320x320
        self.up1 = nn.ConvTranspose2d(c_up1, c1, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(c1 + c1, c1, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1, c1, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.out_conv = nn.Conv2d(c1, c_out, kernel_size=1)

    def forward(self, bottleneck, skips):
        x1, x2, x3, x4 = skips

        x = self.up4(bottleneck)
        x = torch.cat([x, x4], dim=1)
        x = self.dec4(x)

        x = self.up3(x)
        x = torch.cat([x, x3], dim=1)
        x = self.dec3(x)

        x = self.up2(x)
        x = torch.cat([x, x2], dim=1)
        x = self.dec2(x)

        x = self.up1(x)
        x = torch.cat([x, x1], dim=1)
        x = self.dec1(x)

        y_hat = self.out_conv(x)
        return y_hat