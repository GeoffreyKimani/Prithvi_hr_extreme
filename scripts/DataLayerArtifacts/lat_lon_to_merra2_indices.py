import numpy as np

def to_minus180_180_scalar(lon: float) -> float:
    """Map a single longitude from [0,360) to [-180,180)."""
    if lon >= 180.0:
        lon -= 360.0
    return lon


def to_minus180_180_array(lon: np.ndarray) -> np.ndarray:
    """Vectorized 0–360 -> -180–180 conversion for arrays."""
    lon = lon.copy()
    mask = lon >= 180.0
    lon[mask] = lon[mask] - 360.0
    return lon

def latlon_to_merra_indices(lat_min, lat_max, lon_min, lon_max, H: int, W: int):
    """
    Map lat/lon bounds (degrees) to index ranges on a regular MERRA-2 grid.

    Assumes:
      - latitude dimension H spans [-90, 90] with step 0.5 (approx),
      - longitude dimension W spans [-180, 180] with step 0.625 (approx),
      - prithvi_out is shaped [P, H, W] with H ~ 360, W = 576.

    Returns:
      j0, j1, i0, i1  (Python slice end-exclusive, i.e. [:, j0:j1, i0:i1])
    """
    d_lat = 180.0 / H          # ~0.5
    d_lon = 360.0 / W          # ~0.625

    # Clamp to valid domain just in case
    lat_min = max(lat_min, -90.0)
    lat_max = min(lat_max, 90.0)
    # lon_min = max(lon_min, -180.0)
    # lon_max = min(lon_max, 180.0)

    # Latitude: j index increases from south to north
    # phi_j ~ -90 + (j + 0.5) * d_lat  → invert approximately
    j_min_f = (lat_min + 90.0) / d_lat - 0.5
    j_max_f = (lat_max + 90.0) / d_lat - 0.5
    j0 = int(np.floor(j_min_f))
    j1 = int(np.ceil(j_max_f)) + 1   # +1 to be inclusive at top, then end-exclusive

    # Longitude: handle wrap-around if necessary
    # lambda_i ~ -180 + (i + 0.5) * d_lon
    def lon_to_i(lon):
        return (lon + 180.0) / d_lon - 0.5

    i_min_f = lon_to_i(lon_min)
    i_max_f = lon_to_i(lon_max)
    i0 = int(np.floor(i_min_f))
    i1 = int(np.ceil(i_max_f)) + 1

    # Clip to grid bounds
    j0 = max(j0, 0)
    j1 = min(j1, H)
    i0 = max(i0, 0)
    i1 = min(i1, W)

    return j0, j1, i0, i1