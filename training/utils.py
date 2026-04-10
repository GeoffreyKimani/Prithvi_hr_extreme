def unpack_batch(batch, device):
    """
    Unpack a batch from either:
      - HRExtremeDataset: (x_hr, y, mask, event_type)
      - HRExtremeWithPrithviDataset: (x_hr, feats_prithvi, y, mask, event_type)

    Returns:
      x_hr, feats_prithvi_or_none, y, mask, event_type_or_none
    """
    if len(batch) == 4:
        x_hr, y, mask, event_type = batch
        feats_prithvi = None
    elif len(batch) == 5:
        x_hr, feats_prithvi, y, mask, event_type = batch
    else:
        raise ValueError(f"Unexpected batch length {len(batch)}")

    x_hr = x_hr.to(device)
    y    = y.to(device)
    mask = mask.to(device)

    if feats_prithvi is not None:
        feats_prithvi = feats_prithvi.to(device)

    return x_hr, feats_prithvi, y, mask, event_type