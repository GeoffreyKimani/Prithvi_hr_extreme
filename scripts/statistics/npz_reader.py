import numpy as np
from pathlib import Path

exp_name = "unet_prithvi_tail"  # <-- set this to your experiment name
path = Path("~/scratch/prithvi_hr_extreme/outputs") / exp_name / "eval_test" / "rmse_per_variable_per_event_test.npz"
# path = Path("~/scratch/prithvi_hr_extreme/outputs") / exp_name / "eval_test" / "rmse_per_variable_test.npy"
path = path.expanduser()

data = np.load(path, allow_pickle=True)
print("Loaded data from:", path)
print("\nArray names:", data.files)  # list of array names

rmse = data["rmse"]          # shape: (N_events, N_variables)
event_types = data["event_types"]      # shape: (N_events,)

print("RMSE shape:", rmse.shape)  # (N_events, N_variables)
print("Event types shape:", event_types.shape)  # (N_events,)

# print("Sample indices items:", data["sample_indices"][:5])  # (N_events,)
# print("Event types:", event_types[:5])  # (N_events,)
# print("Mean RMSE:", data["mean_rmse_all_vars"][:5])  # (N_events,)
# print("RMSE:", rmse[:5])        # (N_events, N_vars)
# print("Event types:", event_types[:5])   # first few event type labels