#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute typhoon intensity changes relative to the previous best-track time
before a GPM overpass, then at +12/+24/+36/+48 hours.

Edit the config section below to point to your CSVs.
"""

from __future__ import annotations

import glob
import os
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd


# =====================================================
# Config (edit here)
# =====================================================
# Set to a list of years to process separately; leave empty to use the combined/glob logic below.
YEARS = [i for i in range(2015, 2022)]  # IBTrACS season years

# Prefer a single combined file if it exists, otherwise fall back to a glob.
GPM_PASSES_CSV = "gpm_passes_from_ibtracs.csv"
GPM_PASSES_GLOB = "gpm_passes_from_ibtracs_*.csv"
GPM_PASSES_TEMPLATE = "gpm_passes_from_ibtracs_{year}.csv"

# Prefer a single combined IBTrACS file if it exists, otherwise fall back to a glob.
IBTRACS_CSV = "ibtracs_WP.csv"
IBTRACS_GLOB = "ibtracs_WP_*.csv"
IBTRACS_TEMPLATE = "ibtracs_WP_{year}.csv"
# When WRITE_BACK is True, results overwrite the source passes file(s).
# When False, results are written to OUT_CSV.
OUT_CSV = "gpm_passes_intensity_change.csv"
WRITE_BACK = True

# Use pass start/mid/end as the reference "overpass time".
PASS_TIME_REFERENCE = "mid"  # "start" | "mid" | "end"

# Intensity columns to use (first available will be used, with optional fallback).
INTENSITY_COLS = ["USA_WIND"]

# Match tolerance (hours) when looking up a target time in best-track records.
# Set to 0 for exact match only.
MATCH_TOLERANCE_HOURS = 1.0

# Offsets (hours) from the previous best-track time.
OFFSETS_HOURS = [0, 12, 24, 36, 48]


# =====================================================
# Helpers
# =====================================================
def _find_existing_csv(preferred: str, pattern: str) -> List[str]:
    if os.path.exists(preferred):
        return [preferred]
    files = sorted(glob.glob(pattern))
    return files


def _load_passes(files: Iterable[str]) -> pd.DataFrame:
    frames = []
    for path in files:
        df = pd.read_csv(path)
        df["passes_file"] = os.path.basename(path)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _load_ibtracs(files: Iterable[str]) -> pd.DataFrame:
    frames = []
    for path in files:
        df = pd.read_csv(path, low_memory=False)
        df["ibtracs_file"] = os.path.basename(path)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _choose_intensity_column(df: pd.DataFrame, candidates: List[str]) -> str:
    for col in candidates:
        if col not in df.columns:
            continue
        values = pd.to_numeric(df[col], errors="coerce")
        if values.notna().any():
            return col
    raise ValueError(f"No usable intensity column in: {candidates}")


def _compute_pass_time(df: pd.DataFrame, mode: str) -> pd.Series:
    mode = mode.lower().strip()
    if mode == "start":
        return df["pass_start_utc"]
    if mode == "end":
        return df["pass_end_utc"]
    if mode == "mid":
        return df["pass_start_utc"] + (df["pass_end_utc"] - df["pass_start_utc"]) / 2
    raise ValueError(f"Unknown PASS_TIME_REFERENCE: {mode}")


def _find_prev_time(times: np.ndarray, target: np.datetime64) -> Optional[np.datetime64]:
    if times.size == 0:
        return None
    idx = np.searchsorted(times, target, side="right") - 1
    if idx < 0:
        return None
    return times[idx]


def _get_intensity_at_time(
    track: pd.DataFrame,
    target_time: np.datetime64,
    tolerance: Optional[np.timedelta64],
) -> float:
    if track.empty or target_time is None:
        return np.nan
    times = track["time_utc_naive"].to_numpy()
    pos = np.searchsorted(times, target_time)
    candidates = []
    if pos > 0:
        candidates.append(pos - 1)
    if pos < len(times):
        candidates.append(pos)
    if not candidates:
        return np.nan
    cand_idx = np.array(candidates, dtype=int)
    diffs = np.abs(times[cand_idx] - target_time)
    best_i = cand_idx[int(np.argmin(diffs))]
    if tolerance is not None and diffs[int(np.argmin(diffs))] > tolerance:
        return np.nan
    return track["intensity"].iloc[best_i]


def run_for_files(passes_files: List[str], ibtracs_files: List[str]) -> None:
    passes = _load_passes(passes_files)
    if passes.empty:
        raise ValueError("GPM passes dataframe is empty.")

    track = _load_ibtracs(ibtracs_files)
    if track.empty:
        raise ValueError("IBTrACS dataframe is empty.")

    # Parse timestamps and normalize columns.
    passes["pass_start_utc"] = pd.to_datetime(passes["pass_start_utc"], errors="coerce", utc=True)
    passes["pass_end_utc"] = pd.to_datetime(passes["pass_end_utc"], errors="coerce", utc=True)
    passes["pass_time_utc"] = _compute_pass_time(passes, PASS_TIME_REFERENCE)
    passes["SID"] = passes["SID"].astype(str)

    track["time_utc"] = pd.to_datetime(track["ISO_TIME"], errors="coerce", utc=True)
    track["SID"] = track["SID"].astype(str)

    intensity_col = _choose_intensity_column(track, INTENSITY_COLS)
    track["intensity"] = pd.to_numeric(track[intensity_col], errors="coerce")

    # Limit to SIDs that appear in passes to reduce work.
    needed_sids = set(passes["SID"].dropna().unique())
    track = track[track["SID"].isin(needed_sids)].copy()
    track = track.dropna(subset=["time_utc"]).copy()
    # Use tz-naive UTC for searchsorted comparisons.
    track["time_utc_naive"] = track["time_utc"].dt.tz_convert("UTC").dt.tz_localize(None)
    track = track.sort_values(["SID", "time_utc_naive"])

    # Build per-SID lookup.
    track_by_sid = {sid: df for sid, df in track.groupby("SID", sort=False)}

    tolerance = None
    if MATCH_TOLERANCE_HOURS is not None:
        seconds = int(MATCH_TOLERANCE_HOURS * 3600)
        tolerance = np.timedelta64(seconds, "s")

    rows = []
    for _, prow in passes.iterrows():
        sid = prow.get("SID")
        pass_time = prow.get("pass_time_utc")
        if pd.isna(sid) or pd.isna(pass_time):
            continue
        tdf = track_by_sid.get(str(sid))
        if tdf is None or tdf.empty:
            continue

        times = tdf["time_utc_naive"].to_numpy()
        pass_time_naive = pass_time.tz_convert("UTC").tz_localize(None)
        prev_time = _find_prev_time(times, pass_time_naive.to_datetime64())
        if prev_time is None:
            continue

        out = dict(prow)
        out["bst_time_utc"] = pd.Timestamp(prev_time, tz="UTC")
        out["intensity_col"] = intensity_col

        base_intensity = None
        for hours in OFFSETS_HOURS:
            target_time = pd.Timestamp(prev_time, tz="UTC") + pd.Timedelta(hours=hours)
            target_naive = target_time.tz_localize(None)
            val = _get_intensity_at_time(tdf, target_naive.to_datetime64(), tolerance)
            key = "intensity_bst" if hours == 0 else f"intensity_{hours}h"
            out[key] = val
            if hours == 0:
                base_intensity = val
            else:
                if base_intensity is None or pd.isna(base_intensity) or pd.isna(val):
                    out[f"delta_{hours}h"] = np.nan
                else:
                    out[f"delta_{hours}h"] = val - base_intensity

        rows.append(out)

    if not rows:
        raise ValueError("No matches were produced. Check your inputs and time settings.")

    out_df = pd.DataFrame(rows)

    if WRITE_BACK:
        # Write back to source file(s); drop helper column if present.
        if len(passes_files) == 1:
            out_path = passes_files[0]
            out_df.drop(columns=["passes_file"], errors="ignore").to_csv(out_path, index=False)
            print(f"Wrote {len(out_df)} rows to {out_path}")
        else:
            if "passes_file" not in out_df.columns:
                raise ValueError("Missing passes_file; cannot split output per source file.")
            total = 0
            for name, part in out_df.groupby("passes_file", sort=False):
                out_path = name
                part.drop(columns=["passes_file"], errors="ignore").to_csv(out_path, index=False)
                total += len(part)
            print(f"Wrote {total} rows back to {len(passes_files)} files")
    else:
        out_df.to_csv(OUT_CSV, index=False)
        print(f"Wrote {len(out_df)} rows to {OUT_CSV}")


def main() -> None:
    if YEARS:
        for year in YEARS:
            passes_path = GPM_PASSES_TEMPLATE.format(year=year)
            ibtracs_path = IBTRACS_TEMPLATE.format(year=year)
            if not os.path.exists(passes_path):
                raise FileNotFoundError(f"Missing passes file: {passes_path}")
            if not os.path.exists(ibtracs_path):
                raise FileNotFoundError(f"Missing IBTrACS file: {ibtracs_path}")
            run_for_files([passes_path], [ibtracs_path])
    else:
        passes_files = _find_existing_csv(GPM_PASSES_CSV, GPM_PASSES_GLOB)
        if not passes_files:
            raise FileNotFoundError(
                f"No GPM passes files found for {GPM_PASSES_CSV} or {GPM_PASSES_GLOB}"
            )
        ibtracs_files = _find_existing_csv(IBTRACS_CSV, IBTRACS_GLOB)
        if not ibtracs_files:
            raise FileNotFoundError(
                f"No IBTrACS files found for {IBTRACS_CSV} or {IBTRACS_GLOB}"
            )
        run_for_files(passes_files, ibtracs_files)


if __name__ == "__main__":
    main()
