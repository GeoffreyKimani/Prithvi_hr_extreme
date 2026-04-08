"""
Use Prithvi-WxC's download utilities to fetch and preprocess all
MERRA-2 data needed for the HR-Extreme train+test periods.
"""

import os
import argparse, sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from typing import List, Tuple

import requests  # for catching HTTP errors


# --- Paths ---
HOME = Path.home()
PROJ_ROOT = HOME / "scratch" / "prithvi_hr_extreme"
PRITHVI_ROOT = HOME / "scratch" / "Prithvi-WxC"      # root of Prithvi repo

# Make Prithvi code importable
sys.path.insert(0, str(PRITHVI_ROOT))

from PrithviWxC.download import get_prithvi_wxc_input

def read_index_dates(index_path: Path) -> pd.Series:
    """Read index CSV and return a Series of unique dates (no time)."""
    df = pd.read_csv(index_path)
    if "datetime" not in df.columns:
        raise ValueError(f"'datetime' column not found in {index_path}")
    dt = pd.to_datetime(df["datetime"])
    dates = dt.dt.date  # drop time, keep calendar day
    return dates.unique()

def should_retry_http_error(exc: Exception) -> tuple[bool, str]:
    """
    Decide whether to retry and what 'class' of error this is.

    Returns:
        (should_retry: bool, kind: str) where kind is '5xx', '401', or 'other'.
    """
    # Default classification
    kind = "other"

    # Requests-level errors (connection, timeout, etc.) -> treat as retryable 5xx-like
    if isinstance(exc, requests.exceptions.RequestException):
        return True, "5xx"

    if isinstance(exc, requests.HTTPError):
        code = exc.response.status_code
        if 500 <= code < 600:
            kind = "5xx"
            return True, kind
        if code == 401:
            kind = "401"
            return True, kind
        return False, kind

    return False, kind

def download_with_retry(
    time64: np.datetime64,
    input_time_step: int,
    lead_time: int,
    input_data_dir: Path,
    download_dir: Path,
    max_retries: int = 4,
    base_sleep: float = 10.0,
) -> Tuple[bool, str]:
    """
    Call get_prithvi_wxc_input with a small retry loop.

    5xx/network errors: up to max_retries attempts.
    401 errors: at most 2 attempts (first + one retry), then fail and move on.
    """
    attempt = 0
    last_exc_repr = ""
    # Track if we've already retried once on 401 for this date
    retried_401 = False

    while attempt < max_retries:
        try:
            get_prithvi_wxc_input(
                time=time64,
                input_time_step=input_time_step,
                lead_time=lead_time,
                input_data_dir=input_data_dir,
                download_dir=download_dir,
            )
            return True, "ok"
        except Exception as exc:  # noqa: BLE001
            last_exc_repr = repr(exc)
            attempt += 1

            retry, kind = should_retry_http_error(exc)

            if not retry:
                return False, f"non-retryable error: {last_exc_repr}"

            # Special handling for 401: allow only one extra attempt total.
            if kind == "401":
                if retried_401:
                    return False, f"401 unauthorized persisted after retry: {last_exc_repr}"
                retried_401 = True

            # Last allowed attempt -> stop
            if attempt >= max_retries:
                break

            sleep_s = base_sleep * (2 ** (attempt - 1))
            # Optional: clamp sleep to a maximum to avoid burning entire job on one date
            sleep_s = min(sleep_s, 120.0)  # cap at 2 minutes
            print(
                f"[RETRY] Attempt {attempt}/{max_retries} failed ({kind}): {exc}. "
                f"Sleeping {sleep_s:.1f}s before retry..."
            )
            sys.stdout.flush()
            time.sleep(sleep_s)

    return False, f"failed after {max_retries} attempts: {last_exc_repr}"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and prepare MERRA-2 data for HR-Extreme periods."
    )

    parser.add_argument(
        "--input-data-dir",
        type=str,
        default=str(PRITHVI_ROOT / "data" / "merra-2"),
        help="Directory for daily MERRA2_sfc_YYYYMMDD.nc and MERRA_pres_YYYYMMDD.nc.",
    )
    parser.add_argument(
        "--download-dir",
        type=str,
        default=str(PRITHVI_ROOT / "data" / "merra-2_raw"),
        help="Directory where raw MERRA-2 *.nc4 files are cached.",
    )
    parser.add_argument(
        "--input-time-step",
        type=int,
        default=6,
        help="Hours between inputs (matches config.data.input_time magnitude).",
    )
    parser.add_argument(
        "--lead-time",
        type=int,
        default=24,
        help="Lead time in hours up to which Prithvi input is prepared.",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=2020,
        help="Year to filter dates to (HR-Extreme is 2020).",
    )
    parser.add_argument(
        "--month",
        type=int,
        default=None,
        help="Optional month (1–12) to restrict dates to. If omitted, use all months.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=4,
        help="Max retry attempts per date for transient download errors.",
    )
    parser.add_argument(
        "--base-sleep",
        type=float,
        default=10.0,
        help="Base sleep (seconds) for exponential backoff between retries.",
    )
    parser.add_argument(
        "--log-failures",
        type=str,
        default=None,
        help="Optional path to write a CSV listing failed dates and error messages.",
    )

    return parser.parse_args()

def filter_dates(
    all_dates: List[datetime.date], year: int, month: int | None
) -> List[datetime.date]:
    dates = [d for d in all_dates if d.year == year]
    if month is not None:
        dates = [d for d in dates if d.month == month]
    return sorted(dates)

def main():
    args = parse_args()

    index_dir = PROJ_ROOT / "index_files"
    train_index = index_dir / "prithvi_index_train_with_latlon.csv"
    test_index = index_dir / "prithvi_index_test_with_latlon.csv"

    if not train_index.exists():
        raise FileNotFoundError(train_index)
    if not test_index.exists():
        raise FileNotFoundError(test_index)

    # 1) Collect unique dates from train and test
    train_dates = read_index_dates(train_index)
    test_dates = read_index_dates(test_index)
    all_dates = sorted(set(train_dates) | set(test_dates))

    # 2) Restrict to the specified year (2020) and optionally month
    # all_dates_2020 = [d for d in all_dates if d.year == args.year]
    dates = filter_dates(all_dates, year=args.year, month=args.month)
    label = f"{args.year}-{args.month:02d}" if args.month else f"{args.year}"
    print(f"Found {len(dates)} unique HR-Extreme dates in {label}.")

    # 3) Prepare directories
    input_data_dir = Path(args.input_data_dir)
    download_dir = Path(args.download_dir)
    input_data_dir.mkdir(parents=True, exist_ok=True)
    download_dir.mkdir(parents=True, exist_ok=True)

    # 4) Loop over dates; for each, call get_prithvi_wxc_input once.
    
    # Optional per-job requests_cache dir via env
    cache_dir = os.environ.get("REQUESTS_CACHE_DIR")
    if cache_dir:
        print(f"Using requests-cache directory: {cache_dir}")

    n_skipped = 0
    n_done = 0
    n_failed = 0
    failed_records : list[dict] = []

    for d in dates:
        t_mid = datetime(d.year, d.month, d.day, 12, 0)
        time64 = np.datetime64(t_mid)

        # Check whether daily files already exist
        sfc_file = input_data_dir / f"MERRA2_sfc_{d.strftime('%Y%m%d')}.nc"
        pres_file = input_data_dir / f"MERRA_pres_{d.strftime('%Y%m%d')}.nc"
        if sfc_file.exists() and pres_file.exists():
            print(f"[SKIP] {d} already has {sfc_file.name} and {pres_file.name}")
            n_skipped += 1
            continue

        print(f"[DOWNLOAD] Preparing MERRA-2 inputs for {d} ...")
        ok, msg = download_with_retry(
            time64=time64,
            input_time_step=args.input_time_step,
            lead_time=args.lead_time,
            input_data_dir=input_data_dir,
            download_dir=download_dir,
            max_retries=args.max_retries,
            base_sleep=args.base_sleep,
        )

        if ok:
            print(f"[OK] {d} completed.")
            n_done += 1
        else:
            print(f"[FAIL] {d} -> {msg}")
            n_failed += 1
            failed_records.append({"date": d.isoformat(), "error": msg})

        sys.stdout.flush()
        time.sleep(10)  # 10s pause between days

    # Summary
    print(
        f"Completed. Dates in {label}: {len(dates)}, "
        f"downloaded/processed: {n_done}, "
        f"skipped (already present): {n_skipped}, "
        f"failed: {n_failed}"
    )

    # Optional CSV of failures
    if args.log_failures and failed_records:
        log_path = Path(args.log_failures)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        df_fail = pd.DataFrame(failed_records)
        df_fail.to_csv(log_path, index=False)
        print(f"[SUMMARY] Wrote failure log to: {log_path}")

if __name__ == "__main__":
    main()