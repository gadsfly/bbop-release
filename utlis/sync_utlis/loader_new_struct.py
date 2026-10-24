"""
New data structure loader for BBOP dataset.

Updated folder structure:
    rec_path/
    ├── miniscope/
    │   ├── miniscope.nc
    │   └── timeStamps.csv
    ├── metadata/
    │   └── frame_mapping.json
    ├── annotations/
    │   ├── pose3d_mouse22.mat  (or other keypoint counts)
    │   └── com3d.mat
    └── ...

Readers for single files of one session. To align pose with calcium, use
loader_interp.merge_all_data_interp().
"""

import os
import numpy as np
import pandas as pd
import scipy.io as sio
import json
import xarray as xr


def load_pose_from_annotations(
    rec_path,
    pose_filename='pose3d_mouse22.mat',
    kp_names=None
):
    """
    Load pose predictions from annotations folder **or** an absolute path.
    Dynamically handles any number of keypoints (J).

    If *pose_filename* is an absolute path, it is used directly
    (rec_path/annotations/ prefix is skipped).  Otherwise the file is
    looked up at ``rec_path/annotations/pose_filename``.

    Expected .mat shape: (F, A, 3, J)
    
    Returns:
        pose_array: (F, A, 3, J) numpy array
        pose_df: Flattened DataFrame with columns like kp1_x_a1, kp2_x_a1, ...
    """
    if os.path.isabs(pose_filename):
        pose_path = pose_filename
    else:
        pose_path = os.path.join(rec_path, 'annotations', pose_filename)
    
    if not os.path.exists(pose_path):
        raise FileNotFoundError(f"Pose file not found: {pose_path}")
    
    md = sio.loadmat(pose_path)
    if 'pred' not in md:
        raise KeyError(f"'pred' not found in {pose_path}. Keys: {list(md.keys())}")
    
    pose = md['pred']
    
    # STRICT expectation: (F, A, 3, J)
    if not (pose.ndim == 4 and pose.shape[2] == 3):
        raise ValueError(f"Expected pose shape (F, A, 3, J); got {pose.shape}")
    
    F, A, _, J = pose.shape
    
    # Keypoint names
    if kp_names is not None:
        if len(kp_names) != J:
            raise ValueError(f"kp_names length {len(kp_names)} != J {J}")
        kp_labels = list(kp_names)
    else:
        kp_labels = [f"kp{i+1}" for i in range(J)]
    
    coords = ('x', 'y', 'z')
    
    # Flatten: (F, A*3*J)
    pose_flat = pose.reshape(F, A * 3 * J)
    
    # Column names follow memory order: a -> coord -> j
    pose_cols = [
        f"{kp_labels[j]}_{coords[c]}_a{a+1}"
        for a in range(A) for c in range(3) for j in range(J)
    ]
    
    pose_df = pd.DataFrame(pose_flat, columns=pose_cols)
    pose_df.index.name = 'frame'
    
    return pose, pose_df


def load_com_from_annotations(rec_path, com_filename='com3d.mat'):
    """
    Load COM from annotations folder.
    
    Expected shape: (F, 3, nA), or (F, 3) for one animal

    Returns:
        com_array: (F, 3, nA) numpy array
        com_df: DataFrame with columns com1_x, com1_y, com1_z, com2_x, ...
    """
    com_path = os.path.join(rec_path, 'annotations', com_filename)
    
    if not os.path.exists(com_path):
        raise FileNotFoundError(f"COM file not found: {com_path}")
    
    md = sio.loadmat(com_path)
    
    # Try common variable names
    if 'com' in md:
        com = md['com']
    elif 'com3d' in md:
        com = md['com3d']
    else:
        raise KeyError(f"No 'com' or 'com3d' in {com_path}. Keys: {list(md.keys())}")
    
    # Expected: (F, 3, nA); single-animal sessions store (F, 3)
    if com.ndim == 2 and com.shape[1] == 3:
        com = com[:, :, np.newaxis]
    if com.ndim != 3 or com.shape[1] != 3:
        raise ValueError(f"Expected COM shape (F, 3, nA); got {com.shape}")
    
    F, _, nA = com.shape
    
    # Flatten: axis-first then animal
    com_cols = [f'com{a+1}_{ax}' for ax in ('x', 'y', 'z') for a in range(nA)]
    com_df = pd.DataFrame(com.reshape(F, 3 * nA), columns=com_cols)
    com_df.index.name = 'frame'
    
    return com, com_df


def load_miniscope_signals(
    rec_path,
    time_col='Time Stamp (ms)',
    dff_percentile=20
):
    """
    Load miniscope calcium signals and compute dF/F.
    
    Returns:
        calcium_array: (F, n_rois) raw calcium
        dff_array: (F, n_rois) dF/F
        timestamps: (F,) timestamps in ms
        csv_rows: (F,) row indices from timeStamps.csv
    """
    miniscope_path = os.path.join(rec_path, 'miniscope')
    ts_csv = os.path.join(miniscope_path, 'timeStamps.csv')
    nc_file = os.path.join(miniscope_path, 'miniscope.nc')
    
    if not os.path.exists(ts_csv):
        raise FileNotFoundError(f"timeStamps.csv not found: {ts_csv}")
    if not os.path.exists(nc_file):
        raise FileNotFoundError(f"miniscope.nc not found: {nc_file}")
    
    # Load timestamps
    ts_df = pd.read_csv(ts_csv)
    if time_col not in ts_df.columns:
        raise KeyError(f"Column '{time_col}' not in {ts_csv}. Got: {list(ts_df.columns)}")
    timestamps = ts_df[time_col].to_numpy()
    N_csv = len(timestamps)
    
    # Load calcium data
    with xr.open_dataset(nc_file) as ds:
        C = ds['C'].values  # (n_rois, n_frames)
    
    n_rois, n_frames = C.shape
    
    # Compute dF/F
    F0 = np.percentile(C, dff_percentile, axis=1, keepdims=True)
    F0 = np.where(F0 == 0, np.nan, F0)
    dFF = (C - F0) / F0  # (n_rois, n_frames)
    
    # Allocate full-size arrays (pad with NaN if Ca shorter than CSV)
    Ca_full = np.full((N_csv, n_rois), np.nan, dtype=float)
    dFF_full = np.full((N_csv, n_rois), np.nan, dtype=float)
    M = min(N_csv, n_frames)
    Ca_full[:M, :] = C.T[:M, :]
    dFF_full[:M, :] = dFF.T[:M, :]
    
    csv_rows = np.arange(N_csv)
    
    return Ca_full, dFF_full, timestamps, csv_rows


def load_frame_mapping(rec_path, mapping_filename='frame_mapping.json'):
    """
    Load frame mapping from metadata folder.
    
    Returns:
        sixcam_frames: Array of 6-cam frame indices
        mini_timestamps: Array of miniscope timestamps (ms)
    """
    mapping_path = os.path.join(rec_path, 'metadata', mapping_filename)
    
    if not os.path.exists(mapping_path):
        raise FileNotFoundError(f"Frame mapping not found: {mapping_path}")
    
    with open(mapping_path, 'r') as f:
        mp = json.load(f)
    
    sixcam_frames = np.asarray(mp['mapped_sixcam_frame_indices'], dtype=int)
    mini_timestamps = np.asarray(mp['mini_cam_timestamps'])
    
    return sixcam_frames, mini_timestamps
