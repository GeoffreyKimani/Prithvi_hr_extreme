import pandas as pd
from pathlib import Path

HOME = Path.home()
PROJ_ROOT = HOME / "scratch" / "prithvi_hr_extreme"
PRITHVI_FEAT_DIR = HOME / "scratch" / "prithvi_features"

train_idx_path = PROJ_ROOT / "index_files" / "prithvi_index_train_with_latlon.csv"
test_idx_path  = PROJ_ROOT / "index_files" / "prithvi_index_test_with_latlon.csv"

out_train_csv = PROJ_ROOT / "index_files" / "hrx_prithvi_train.csv"
out_val_csv   = PROJ_ROOT / "index_files" / "hrx_prithvi_val.csv"
out_test_csv  = PROJ_ROOT / "index_files" / "hrx_prithvi_test.csv"


def add_paths_and_filter(df: pd.DataFrame, split: str) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["datetime"])

    # HR-Extreme path
    df["hrx_path"] = df.apply(
        lambda r: str(Path(r["npz_dir"]) / r["npz_filename"]), axis=1
    )

    # Prithvi features path
    if split == "train":
        feat_root = PRITHVI_FEAT_DIR / "train"
    else:  # "test"
        feat_root = PRITHVI_FEAT_DIR / "test"

    df["stem"] = df["npz_filename"].apply(lambda s: Path(s).stem)
    df["prithvi_path"] = df["stem"].apply(
        lambda stem: str(feat_root / f"{stem}_prithvi.npz")
    )

    # Keep only rows where both files exist
    df["has_hrx"] = df["hrx_path"].apply(lambda p: Path(p).is_file())
    df["has_prithvi"] = df["prithvi_path"].apply(lambda p: Path(p).is_file())
    df_valid = df[df["has_hrx"] & df["has_prithvi"]].reset_index(drop=True)

    return df_valid



def time_stratified_event_split(df, val_frac=0.2):
    """
    For each event_type, hold out the last val_frac (by datetime) as validation.
    Returns train_mask, val_mask aligned with df.index.
    """
    if not pd.api.types.is_datetime64_any_dtype(df["datetime"]):
        df = df.copy()
        df["datetime"] = pd.to_datetime(df["datetime"])

    val_indices = []

    for etype, group in df.groupby("event_type"):
        group_sorted = group.sort_values("datetime")

        n = len(group_sorted)
        n_val = max(1, int(val_frac * n))

        # last n_val rows in time for this event type → val
        val_idx = group_sorted.index[-n_val:]
        val_indices.extend(val_idx)

    val_indices = set(val_indices)
    val_mask = df.index.isin(val_indices)
    train_mask = ~val_mask
    return train_mask, val_mask


def main():
    train_df = pd.read_csv(train_idx_path)
    test_df  = pd.read_csv(test_idx_path)

    train_valid = add_paths_and_filter(train_df, split="train")
    test_valid  = add_paths_and_filter(test_df, split="test")

    # # Time-based train/val split within Jan–Jun:
    # # e.g. Jan–May = train, June = val
    # train_mask = train_valid["date"] < "2020-06-01"
    # val_mask   = (train_valid["date"] >= "2020-06-01") & (train_valid["date"] < "2020-07-01")
    train_mask, val_mask = time_stratified_event_split(train_valid, val_frac=0.2)

    final_train = train_valid[train_mask].copy()
    final_val   = train_valid[val_mask].copy()

    final_train["split"] = "train"
    final_val["split"]   = "val"
    test_valid["split"]  = "test"

    final_train.to_csv(out_train_csv, index=False)
    final_val.to_csv(out_val_csv, index=False)
    test_valid.to_csv(out_test_csv, index=False)

    print(f"Train rows: {len(final_train)}, Val rows: {len(final_val)}, Test rows: {len(test_valid)}")


if __name__ == "__main__":
    main()
