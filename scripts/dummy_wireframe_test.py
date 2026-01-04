import torch
from pathlib import Path

from models.hr_encoder import HREncoder
from models.prithvi_backbone import PrithviBackbone
from models.hr_head import HRHead

# If you use a config loader, you can pull paths from configs/paths.yaml instead.
CONFIG_PATH = str(Path("~/scratch/Prithvi-WxC/data/config.yaml").expanduser())
WEIGHTS_PATH = str(Path("~/scratch/Prithvi-WxC/data/weights/prithvi.wxc.2300m.v1.pt").expanduser())


# For this dummy test we don't actually use Prithvi's scalers; we can pass simple placeholders.
# Later you'll replace these with real scalers as in Prithvi_inference.py.
def dummy_scalers(c_in=160):
    mu = torch.zeros(c_in)
    sig = torch.ones(c_in)
    return mu, sig


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Instantiate HR-Extreme encoder
    hr_encoder = HREncoder(
        in_channels=69,
        time_steps=3,
        c_hidden=128,
        c_hidden2=256,
        c_back_in=256,
        h_in=320,
        w_in=320,
        h_p=80,
        w_p=80,
    ).to(device)

    # 2. Instantiate Prithvi backbone (with placeholder scalers)
    in_mu, in_sig = dummy_scalers()
    static_mu, static_sig = dummy_scalers(c_in=4)
    output_sig = torch.ones(160)

    prithvi_backbone = PrithviBackbone(
        config_path=CONFIG_PATH,
        weights_path=WEIGHTS_PATH,
        in_mu=in_mu,
        in_sig=in_sig,
        static_mu=static_mu,
        static_sig=static_sig,
        output_sig=output_sig,
        device=device,
    )

    # 3. Instantiate HR head
    hr_head = HRHead(
        c_back_in=256,
        c_head_hidden=256,
        h_p=80,
        w_p=80,
        h_out=320,
        w_out=320,
        out_channels=69,
    ).to(device)

    # 4. Create dummy HR-Extreme input: (B, T, C, H, W) = (2, 3, 69, 320, 320)
    x_hr = torch.randn(2, 3, 69, 320, 320, device=device)

    # 5. Forward pass through encoder -> backbone -> head
    with torch.no_grad():
        z_in = hr_encoder(x_hr)  # (2, 256, 80, 80)
        print("z_in shape:", z_in.shape)

        z_feat = prithvi_backbone.forward_from_features(z_in)  # currently identical
        print("z_feat shape:", z_feat.shape)

        y_hat = hr_head(z_feat)  # (2, 69, 320, 320)
        print("y_hat shape:", y_hat.shape)

    # Sanity checks
    assert y_hat.shape == (2, 69, 320, 320)
    print("Composite model test passed.")


if __name__ == "__main__":
    main()