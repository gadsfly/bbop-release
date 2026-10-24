"""
Interpolation-based loader for BBOP merged data.

Pose data is linearly interpolated onto the miniscope timeline, so we get smooth
pose at every miniscope timestamp. Ca2+ data is NEVER interpolated — only real
measured values are used.

  - loader_interp.merge_all_data_interp()  → interpolated pose, real Ca2+
  - loader_interp.load_behavior_only()     → pose + COM on the camera clock
                                             (sessions without miniscope)

The file readers are imported from loader_new_struct.py.

Folder structure expected (same as loader_new_struct):
    rec_path/
    ├── miniscope/
    │   ├── miniscope.nc
    │   └── timeStamps.csv       ← REAL miniscope timestamps (never assume fps)
    ├── metadata/
    │   └── frame_mapping.json   ← only time_offset is used from here
    ├── videos/
    │   └── Camera1/
    │       └── frametimes.npy   ← REAL sixcam timestamps (2, N) array
    ├── annotations/
    │   ├── pose3d_mouse22.mat
    │   └── com3d.mat
    └── ...
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import scipy.io as sio
import xarray as xr
from scipy.interpolate import interp1d


# ---------------------------------------------------------------------------
#  Re-use atomic loaders from loader_new_struct (no duplication)
# ---------------------------------------------------------------------------
# We import the existing atomic loaders so that file-reading logic is not
# duplicated.  Only the MERGE strategy changes.
try:
    from utlis.sync_utlis.loader_new_struct import (
        load_pose_from_annotations,
        load_com_from_annotations,
        load_miniscope_signals,
        load_frame_mapping,
    )
except ImportError:
    # Fallback: allow running from a notebook that already has sys.path set
    from loader_new_struct import (
        load_pose_from_annotations,
        load_com_from_annotations,
        load_miniscope_signals,
        load_frame_mapping,
    )


# ===================================================================
#  Core: interpolation-based merge
# ===================================================================

def _load_sixcam_timestamps(rec_path, time_offset, camera=1):
    """
    Load REAL sixcam timestamps from frametimes.npy, apply time_offset.

    frametimes.npy shape: (2, N)
        Row 0: frame indices (1-based)
        Row 1: timestamps in seconds

    time_offset (ms): clock offset from frame_mapping.json
        = sync_time_mini_webcam - sync_time_sixcam
        (miniscope always starts later, so offset is typically negative)

    Returns
    -------
    adjusted_ts_ms : ndarray (N,)
        Sixcam timestamps in ms, aligned to miniscope clock.
        Returns None if frametimes.npy is not found.
    """
    ft_path = os.path.join(rec_path, 'videos', f'Camera{camera}', 'frametimes.npy')
    if not os.path.exists(ft_path):
        warnings.warn(f"frametimes.npy not found at {ft_path}")
        return None
    ft = np.load(ft_path)  # (2, N) — row0=frame_idx (1-based), row1=time_sec
    sixcam_ts_ms = ft[1] * 1000.0  # convert seconds → milliseconds
    return sixcam_ts_ms + time_offset  # align to miniscope clock


def _estimate_beh_timestamps_fallback(sixcam_frames, mini_timestamps, n_beh_frames):
    """
    FALLBACK ONLY — used when frametimes.npy is missing.

    Fits a linear model through the JSON mapping pairs to estimate
    behaviour timestamps.  This is approximate because the JSON pairs
    are nearest-neighbour outputs, not sparse control points.
    """
    slope, intercept = np.polyfit(sixcam_frames.astype(float),
                                  mini_timestamps.astype(float), 1)
    all_frames = np.arange(n_beh_frames, dtype=float)
    beh_ts = slope * all_frames + intercept
    return beh_ts


def _interpolate_array_onto_timeline(src_ts, src_array, dst_ts, kind='linear'):
    """
    Interpolate a (N_src, ...) array from src_ts → dst_ts along axis 0.

    Parameters
    ----------
    src_ts  : (N_src,)  source timestamps
    src_array : (N_src, ...) data array
    dst_ts  : (N_dst,)  target timestamps
    kind    : str, interpolation kind ('linear', 'nearest', 'cubic', …)

    Returns
    -------
    dst_array : (N_dst, ...) interpolated array.  Values outside src range = NaN.
    """
    orig_shape = src_array.shape
    flat = src_array.reshape(orig_shape[0], -1).astype(float)

    fn = interp1d(src_ts, flat, axis=0, kind=kind,
                  bounds_error=False, fill_value=np.nan)
    interp_flat = fn(dst_ts)
    dst_shape = (len(dst_ts),) + orig_shape[1:]
    return interp_flat.reshape(dst_shape)


def merge_all_data_interp(
    rec_path,
    pose_filename='pose3d_mouse22.mat',
    com_filename='com3d.mat',
    mapping_filename='frame_mapping.json',
    kp_names=None,
    time_col='Time Stamp (ms)',
    dff_percentile=20,
    interp_kind='linear',
    extra_cols=None,
):
    """
    Load and merge all data using **pose interpolation** onto the miniscope
    timeline.  Ca2+ signals are NEVER interpolated — only real values are kept.

    Parameters
    ----------
    rec_path : str
        Path to the session folder.
    pose_filename, com_filename : files in rec_path/annotations/.
    mapping_filename : file in rec_path/metadata/ (only time_offset is used).
    kp_names : optional keypoint names for the pose columns.
    time_col : timestamp column in miniscope/timeStamps.csv.
    dff_percentile : percentile of C used as the fixed F0 for dF/F.
    interp_kind : str
        scipy interp1d kind for pose/COM interpolation.  Default 'linear'.
    extra_cols : dict | None
        Optional dict of {col_name: array} to attach to merged_df.
        Arrays must have the same length as the merged timeline.

    Returns
    -------
    merged_df     : pd.DataFrame — full merged table (index = mini timestamp ms)
    beh_df        : pd.DataFrame — behaviour-only columns
    mini_df       : pd.DataFrame — calcium-only columns
    pose_interp   : ndarray (N_merged, A, 3, J) — interpolated pose
    calcium_array : ndarray (N_merged, n_rois)  — real calcium values
    dff_array     : ndarray (N_merged, n_rois)  — real dF/F values
    merge_info    : dict — diagnostic info (for QC / debugging)
    """

    # ── 1. Load raw data ──────────────────────────────────────────────
    pose_array, pose_df_unused = load_pose_from_annotations(
        rec_path, pose_filename, kp_names)
    com_array, com_df_unused = load_com_from_annotations(
        rec_path, com_filename)
    # Ca2+ signals + REAL miniscope timestamps from timeStamps.csv
    # (ts_full = real timestamps, never assume fps)
    ca_full, dff_full, ts_full, csv_rows = load_miniscope_signals(
        rec_path, time_col, dff_percentile)

    # Only read time_offset from the JSON — ignore mapped indices
    mapping_path = os.path.join(rec_path, 'metadata', mapping_filename)
    if not os.path.exists(mapping_path):
        raise FileNotFoundError(f"Frame mapping not found: {mapping_path}")
    with open(mapping_path, 'r') as f:
        mp = json.load(f)
    time_offset = mp['time_offset']

    n_beh = pose_array.shape[0]
    F, A, _, J = pose_array.shape
    n_rois = ca_full.shape[1]

    # ── 2. Get REAL sixcam timestamps (or fallback) ───────────────────
    adjusted_sixcam_ts = _load_sixcam_timestamps(rec_path, time_offset)

    if adjusted_sixcam_ts is not None:
        # Handle mismatch: frametimes may have more frames than pose
        # (e.g. 27300 vs 27288 — DANNCE trims a few frames)
        n_beh = min(len(adjusted_sixcam_ts), pose_array.shape[0])
        beh_ts = adjusted_sixcam_ts[:n_beh]
        ts_source = 'frametimes.npy'
    else:
        # Fallback: estimate from JSON mapping pairs (approximate)
        sixcam_frames_json = np.asarray(mp['mapped_sixcam_frame_indices'], dtype=int)
        mini_ts_json = np.asarray(mp['mini_cam_timestamps'])
        beh_ts = _estimate_beh_timestamps_fallback(
            sixcam_frames_json, mini_ts_json, n_beh)
        ts_source = 'polyfit_fallback'

    # ── 3. Define the merged timeline = REAL miniscope timestamps ─────
    #       from timeStamps.csv (only within the overlap with behaviour)
    mini_timestamps = ts_full  # real timestamps from CSV
    beh_start = beh_ts[0]
    beh_end   = beh_ts[-1]
    valid_mask = (mini_timestamps >= beh_start) & (mini_timestamps <= beh_end)
    merged_ts = mini_timestamps[valid_mask]

    if len(merged_ts) == 0:
        raise RuntimeError("No overlapping time range between pose and miniscope.")

    n_merged = len(merged_ts)

    # ── 4. Interpolate pose onto merged timeline ──────────────────────
    pose_interp = _interpolate_array_onto_timeline(
        beh_ts, pose_array[:n_beh], merged_ts, kind=interp_kind)

    # ── 5. Interpolate COM onto merged timeline ───────────────────────
    com_interp = _interpolate_array_onto_timeline(
        beh_ts, com_array[:n_beh], merged_ts, kind=interp_kind)

    # ── 6. Align Ca2+: real values only (NO interpolation) ────────────
    #   Nearest-with-tolerance matching (not exact dict lookup)
    #   This prevents silent NaN loss from floating-point mismatches.
    sorted_idx = np.searchsorted(ts_full, merged_ts)
    sorted_idx = np.clip(sorted_idx, 0, len(ts_full) - 1)
    left_idx = np.clip(sorted_idx - 1, 0, len(ts_full) - 1)
    d_right = np.abs(ts_full[sorted_idx] - merged_ts)
    d_left  = np.abs(ts_full[left_idx] - merged_ts)
    csv_row_indices = np.where(d_left < d_right, left_idx, sorted_idx)
    # Tolerance check: reject if nearest is >50ms away
    actual_ts = ts_full[csv_row_indices]
    too_far = np.abs(merged_ts - actual_ts) > 50.0
    csv_row_indices[too_far] = -1
    valid_ca = csv_row_indices >= 0

    calcium_array = np.full((n_merged, n_rois), np.nan)
    dff_array     = np.full((n_merged, n_rois), np.nan)
    calcium_array[valid_ca] = ca_full[csv_row_indices[valid_ca]]
    dff_array[valid_ca]     = dff_full[csv_row_indices[valid_ca]]

    # ── 7. Build behaviour DataFrame (interpolated) ───────────────────
    # Flatten pose: (n_merged, A*3*J)
    if kp_names is not None:
        kp_labels = list(kp_names)
    else:
        kp_labels = [f"kp{i+1}" for i in range(J)]
    coords = ('x', 'y', 'z')
    pose_cols = [
        f"{kp_labels[j]}_{coords[c]}_a{a+1}"
        for a in range(A) for c in range(3) for j in range(J)
    ]
    pose_flat = pose_interp.reshape(n_merged, -1)
    pose_df = pd.DataFrame(pose_flat, columns=pose_cols)

    # Flatten COM
    nA_com = com_interp.shape[2]
    com_cols = [f'com{a+1}_{ax}' for ax in ('x', 'y', 'z') for a in range(nA_com)]
    com_flat = com_interp.reshape(n_merged, -1)
    com_df = pd.DataFrame(com_flat, columns=com_cols)

    beh_df = pd.concat([com_df, pose_df], axis=1)
    beh_df.index = merged_ts
    beh_df.index.name = 'timestamp_ms_mini'

    # Also store which sixcam frame is closest (for video look-up)
    # Use the real beh_ts → frame index mapping (safe: no extrapolation)
    frame_of_time = interp1d(beh_ts, np.arange(n_beh, dtype=float),
                             kind='nearest', bounds_error=False,
                             fill_value=(0, n_beh - 1))
    beh_df['camera_frame_sixcam'] = np.round(
        frame_of_time(merged_ts)).astype(int)

    # ── 8. Build miniscope DataFrame (real Ca2+ only) ─────────────────
    calcium_cols = [f'calcium_roi{i}' for i in range(n_rois)]
    dff_cols     = [f'dF_F_roi{i}' for i in range(n_rois)]
    mini_df = pd.DataFrame(calcium_array, columns=calcium_cols, index=merged_ts)
    mini_df_dff = pd.DataFrame(dff_array, columns=dff_cols, index=merged_ts)
    mini_df = pd.concat([mini_df, mini_df_dff], axis=1)
    mini_df['miniscope_csv_row'] = csv_row_indices
    mini_df.index.name = 'timestamp_ms_mini'

    # ── 9. Merge ──────────────────────────────────────────────────────
    merged_df = pd.concat([beh_df, mini_df], axis=1)

    # ── 10. Attach extra columns ──────────────────────────────────────
    if extra_cols is not None:
        for col_name, col_arr in extra_cols.items():
            if len(col_arr) != n_merged:
                warnings.warn(
                    f"extra_cols['{col_name}'] length {len(col_arr)} != "
                    f"merged length {n_merged}; skipping.")
                continue
            merged_df[col_name] = col_arr

    # ── 11. Diagnostic info ───────────────────────────────────────────
    merge_info = {
        'n_beh_frames_total': n_beh,
        'n_mini_frames_total': len(ts_full),
        'n_merged': n_merged,
        'n_ca_valid': int(valid_ca.sum()),
        'n_ca_nan': int((~valid_ca).sum()),
        'n_ca_too_far': int(too_far.sum()),
        'beh_ts_range_ms': (float(beh_ts[0]), float(beh_ts[-1])),
        'merged_ts_range_ms': (float(merged_ts[0]), float(merged_ts[-1])),
        'interp_kind': interp_kind,
        'ts_source': ts_source,
        'time_offset_ms': float(time_offset),
    }

    return merged_df, beh_df, mini_df, pose_interp, calcium_array, dff_array, merge_info


# ===================================================================
#  Sessions without miniscope: behaviour on the camera clock
# ===================================================================

def load_behavior_only(
    rec_path,
    pose_filename='pose3d_mouse22.mat',
    com_filename='com3d.mat',
    kp_names=None,
    camera=1,
):
    """
    Load pose (and COM, if the session has one) on the camera clock.
    Works for every session; no interpolation, one row per pose frame.

    Returns
    -------
    beh_df     : pd.DataFrame — COM + pose columns, camera_frame_sixcam,
                 timestamp_ms_camera (Camera{camera} frametimes.npy, ms)
    pose_array : ndarray (F, A, 3, J)
    com_array  : ndarray (F, 3, nA), or None if there is no COM file
    """
    pose_array, pose_df = load_pose_from_annotations(
        rec_path, pose_filename, kp_names)
    n = pose_array.shape[0]

    com_array = None
    parts = [pose_df]
    if os.path.exists(os.path.join(rec_path, 'annotations', com_filename)):
        com_array, com_df = load_com_from_annotations(rec_path, com_filename)
        com_array = com_array[:n]
        parts.insert(0, com_df.iloc[:n])

    beh_df = pd.concat(parts, axis=1)
    beh_df['camera_frame_sixcam'] = np.arange(n)
    cam_ts = _load_sixcam_timestamps(rec_path, 0.0, camera=camera)
    if cam_ts is not None:
        ts = np.full(n, np.nan)
        m = min(n, len(cam_ts))
        ts[:m] = cam_ts[:m]
        beh_df['timestamp_ms_camera'] = ts
    beh_df.index.name = 'frame'

    return beh_df, pose_array, com_array


