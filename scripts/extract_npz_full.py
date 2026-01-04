import tarfile
from pathlib import Path
import numpy as np

SRC_DIR = Path("~/scratch/hr_extreme_data/202007_202012").expanduser()
OUT_DIR = Path("~/scratch/hr_extreme_npz/all").expanduser()
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    count = 0
    for tar_path in sorted(SRC_DIR.glob("*.tar")):
        print(f"Processing {tar_path} ...")
        with tarfile.open(tar_path, "r") as tar:
            for member in tar:
                if not member.name.endswith(".npz"):
                    continue
                f = tar.extractfile(member)
                if f is None:
                    continue

                try:
                    data = np.load(f)
                except Exception as e:
                    print(f"  Skipping {member.name}: np.load failed ({e})")
                    continue

                if not all(k in data for k in ["inputs", "targets", "masks"]):
                    print(f"  Skipping {member.name}: keys={list(data.keys())}")
                    continue

                try:
                    inputs = data["inputs"]    # expected (1, T_in, 69, 320, 320) or (T_in, 69, 320, 320)
                    targets = data["targets"]  # expected (1, T_out, 69, 320, 320) or (T_out, 69, 320, 320)
                    masks = data["masks"]      # expected (1, 320, 320) or (320, 320)
                except Exception as e:
                    print(f"  Skipping {member.name}: error reading arrays ({e})")
                    continue

                # Normalize shapes: ensure we end up with
                #   inputs:  (T_in, 69, 320, 320)
                #   targets: (T_out, 69, 320, 320)
                #   masks:   (320, 320)

                # Handle possible batch dim on inputs
                if inputs.ndim == 5 and inputs.shape[0] == 1:
                    inputs = np.squeeze(inputs, axis=0)  # (T_in, 69, 320, 320)
                elif inputs.ndim == 4:
                    # assume already (T_in, 69, 320, 320)
                    pass
                else:
                    print(f"  Skipping {member.name}: unexpected inputs shape {inputs.shape}")
                    continue

                # Handle possible batch dim on targets
                if targets.ndim == 5 and targets.shape[0] == 1:
                    targets = np.squeeze(targets, axis=0)  # (T_out, 69, 320, 320)
                elif targets.ndim == 4:
                    # assume already (T_out, 69, 320, 320)
                    pass
                else:
                    print(f"  Skipping {member.name}: unexpected targets shape {targets.shape}")
                    continue

                # Handle possible batch dim on masks
                if masks.ndim == 3 and masks.shape[0] == 1:
                    masks = np.squeeze(masks, axis=0)  # (320, 320)
                elif masks.ndim == 2:
                    # assume already (320, 320)
                    pass
                else:
                    print(f"  Skipping {member.name}: unexpected masks shape {masks.shape}")
                    continue

                # Take the first target time step
                if targets.ndim == 4 and targets.shape[0] >= 1:
                    y = targets[0]  # (69, 320, 320)
                else:
                    print(f"  Skipping {member.name}: invalid targets shape after squeeze {targets.shape}")
                    continue

                x = inputs  # (T_in, 69, 320, 320)
                mask = masks  # (320, 320)

                out_path = OUT_DIR / f"sample_{count:07d}.npz"
                np.savez_compressed(
                    out_path,
                    x=x,
                    y=y,
                    mask=mask,
                )
                count += 1

    print(f"Extracted total {count} samples.")


if __name__ == "__main__":
    main()