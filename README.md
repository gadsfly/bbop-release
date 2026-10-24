# BBOP

Code for loading, synchronizing, projecting, and visualizing the BBOP dataset:
freely moving mice filmed by six cameras in a 1-m open field, with 3D pose from
social-DANNCE (22 keypoints) and, in part of the sessions, UCLA Miniscope V4
calcium imaging from V1 or M1/PMC.

This repository accompanies the BBOP dataset paper (bioRxiv, 2026; citation and
DOI added on posting). The data, including the pose-model weights, are in the
data deposit, not in this repository.

## Install

```bash
conda env create -f environment.yml
conda activate bbop
pip install -e .
```

`environment.yml` pins the versions this release was validated with
(Python 3.9, Linux). `pip install -e .` alone also works in an existing
environment.

## Quick start

Point `RELEASE_ROOT` at the downloaded dataset. A session is identified by
`cohort/kind/session`, because a few session names occur in both cohorts.

**Sessions with calcium imaging** (`oct3v1_mini`): pose and COM are
interpolated onto the miniscope timestamps; calcium and dF/F are the measured
values and are never interpolated.

```python
import os
from utlis.sync_utlis.loader_interp import merge_all_data_interp

RELEASE_ROOT = "/path/to/release"
rec_path = os.path.join(RELEASE_ROOT, "oct3v1_mini/social/20241031_VC4+UNK1_S1")
merged, beh_df, mini_df, pose_interp, ca, dff, info = merge_all_data_interp(rec_path)
# merged: one row per miniscope frame, with pose, COM, calcium and dF/F columns
```

**Behavior-only sessions** (`oct3v1_beh_only`, or any session when calcium is
not needed): one row per pose frame on the Camera1 clock.

```python
from utlis.sync_utlis.loader_interp import load_behavior_only

rec_path = os.path.join(RELEASE_ROOT, "oct3v1_beh_only/single/20241004_VC11_S1")
beh_df, pose, com = load_behavior_only(rec_path)   # com is None if the session has no COM
```

**Pose overlaid on a camera video:**

```python
from utlis.vis_valid_utlis.sdannce_vis import visualize_clip

visualize_clip(
    base_path=rec_path,
    pred_folder="annotations",
    pred_filename="pose3d_mouse22.mat",
    com_filename="com3d.mat",
    animal="mouse22",
    cammm=1,            # camera 1-6
    start_frame=1000,
    n_frames=100,
)
# writes an .mp4 to <session>/annotations/vis/
```

`notebooks/load_session.ipynb` walks through one calcium session and one
behavior-only session.

## Session layout

```
<cohort>/<single|social>/<session>/
├── label3d_dannce.mat        calibration of the six cameras, and each camera's
│                             synchronized frame index (sync.data_frame)
├── annotations/
│   ├── pose3d_mouse22.mat    3D pose, `pred`: (frames, animals, 3, 22), mm
│   ├── com3d.mat             center of mass, `com`: (frames, 3) or (frames, 3, 2)
│   └── *_provenance.json     how each product was made
├── metadata/
│   └── frame_mapping.json    miniscope-to-camera mapping, including time_offset
├── miniscope/                calcium sessions only
│   ├── miniscope.nc          MiniAn output
│   └── timeStamps.csv
└── videos/Camera1 ... Camera6/
    ├── 0.mp4
    └── frametimes.npy
```

`cohort` is `oct3v1_mini` (with calcium imaging) or `oct3v1_beh_only`.

## Conventions

**Pose arrays** are `(frames, animals, 3, keypoints)`. Social sessions have two
animals on axis 1, so do not squeeze that axis. Keypoint order:

| index | keypoint | index | keypoint | index | keypoint |
|---|---|---|---|---|---|
| 0 | EarL | 8 | ForepawL | 16 | HindpawL |
| 1 | EarR | 9 | WristL | 17 | AnkleL |
| 2 | Snout | 10 | ElbowL | 18 | KneeL |
| 3 | SpineF | 11 | ShoulderL | 19 | HindpawR |
| 4 | SpineM | 12 | ForepawR | 20 | AnkleR |
| 5 | Tail (base) | 13 | WristR | 21 | KneeR |
| 6 | Tail (mid) | 14 | ElbowR | | |
| 7 | Tail (end) | 15 | ShoulderR | | |

Skeleton edges and colors are in `utlis/connectivity.py` (`mouse22`).

**Cameras follow the MATLAB / Label3D convention, not OpenCV.**
- `K` is stored transposed: `[[fx, 0, 0], [0, fy, 0], [cx, cy, 1]]`.
- Points are row vectors: `pts_2d = pts_3d @ np.concatenate((R, t), axis=0) @ K`.
- `r` from `load_cameras` is a flattened 3×3 rotation matrix, not a Rodrigues
  vector: `R = np.asarray(r).reshape(3, 3)`.

Use `utlis.projection.load_cameras`, `project_to_2d` and `distortPoints`.

**The six cameras are not frame-locked to each other.** Pose row *i*
corresponds to video frame `sync.data_frame[i] - 1` of each camera (stored per
camera in `label3d_dannce.mat`; `data_frame` is 1-based). In the released
sessions these per-camera offsets are 0–6 frames. `sdannce_vis` applies them;
reading pose row *i* as video frame *i* puts the skeleton on the wrong frame.

## Contents

| module | purpose |
|---|---|
| `utlis/sync_utlis/loader_interp.py` | `merge_all_data_interp` (pose and COM onto the miniscope clock) and `load_behavior_only` |
| `utlis/sync_utlis/loader_new_struct.py` | readers for pose, COM, MiniAn signals and `frame_mapping.json` |
| `utlis/sync_utlis/mini_dannce_sync.py` | miniscope-to-camera alignment and `frame_mapping.json` generation |
| `utlis/sync_utlis/sync_df_utlis.py` | camera synchronization from brightness changes; `find_calib_file` |
| `utlis/sync_utlis/dropf_handler.py` | dropped-frame handling on a common camera timeline |
| `utlis/convert_calib_utlis/` | calibration and sync conversion into `label3d_dannce.mat`, intrinsics update, and the calibration wrapper |
| `utlis/exe_engine_utlis/dead_cam_rescue.py` | rescue of sessions with a missing camera |
| `utlis/projection.py` | camera loading and projection (MATLAB convention) |
| `utlis/connectivity.py` | skeleton edges and colors |
| `utlis/vis_valid_utlis/sdannce_vis.py` | pose and COM overlays on the camera videos |

## Related tools (not in this repository)

- 3D pose: social-DANNCE (<https://github.com/tqxli/sdannce>) and DANNCE
  (<https://github.com/spoonsso/dannce>).
- Center of mass: YOLO detectors with multi-view triangulation, described in
  the paper's Methods.
- Calcium extraction: MiniAn (<https://github.com/denisecailab/minian>).
- Identity-switch correction for social sessions:
  <https://github.com/gadsfly/reid-switch-corrector>.
- Cross-session ROI candidates: <https://github.com/gadsfly/roi-session-alignment>.
- Calibration: `utlis/convert_calib_utlis/chris_wrapper.py` runs Chris Axon's
  newsdannce calibration code, which is not included; pass its path.

## Credits

- `project_to_2d` and `distortPoints` in `utlis/projection.py` follow DANNCE
  (MIT license).
- The calibration wrapper runs Chris Axon's calibration code.

## License

MIT; see `LICENSE`.
