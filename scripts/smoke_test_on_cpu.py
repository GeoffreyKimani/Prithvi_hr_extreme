import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from torch.utils.data import DataLoader
import yaml, torch
from pathlib import Path
from datasets.hr_extreme_dataset import HRExtremeDataset
from models.hr_prithvi_model import HRPrithviModel
from models.hr_encoder import HREncoder
from models.prithvi_backbone import PrithviBackbone
from models.hr_head import HRHead

# // Dataloader checker: make sure we can load the dataset and get a batch without errors.
# Load the config file
paths_cfg = yaml.safe_load(Path("configs/paths.yaml").expanduser().open())
exp_cfg = yaml.safe_load(Path("configs/hrx_prithvi_backbone.yaml").expanduser().open())

train_dir = str(Path(paths_cfg["data"]["hr_extreme"]["tiny_train"]).expanduser())
stats_path = str(Path(paths_cfg["data"]["hr_extreme"]["stats_path"]).expanduser())

ds = HRExtremeDataset(train_dir, stats_path=stats_path, normalize=True)
print(f"\n\n\tDataset size: {len(ds)} samples")

x, y, mask = ds[0]
print(f"\tx shape: {x.shape},y shape: {y.shape}, mask shape: {mask.shape}")

# // One full forward pass through the model to check for errors.
device = torch.device("cpu")
loader = DataLoader(ds, batch_size=2, shuffle=False, num_workers=0)

hr_encoder = HREncoder(exp_cfg).to(device)
config_path = str(Path(paths_cfg["model"]["prithvi_config"]).expanduser())
prithvi_backbone = PrithviBackbone(
    config_path=config_path,
    weights_path=None,           # or load_weights=False to avoid heavy load
    load_weights=False,
    device=device,
)
hr_head = HRHead(exp_cfg).to(device)

model = HRPrithviModel(hr_encoder, prithvi_backbone, hr_head).to(device)

x, y, mask = next(iter(loader))   # one mini-batch
x = x.to(device)                  # (B, T, C, H, W)

with torch.no_grad():
    y_hat = model(x)

print("\ty_hat", y_hat.shape)       # expect (B, 69, 320, 320)

# // Mini train step on cpu to ensure it works without errors.
from training.train_hr_prithvi import masked_mse

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

model.train()
x, y, mask = next(iter(loader))
x, y, mask = x.to(device), y.to(device), mask.to(device)

optimizer.zero_grad()
y_hat = model(x)
loss = masked_mse(y_hat, y, mask=mask)
loss.backward()
optimizer.step()

print("\tloss", loss.item())
