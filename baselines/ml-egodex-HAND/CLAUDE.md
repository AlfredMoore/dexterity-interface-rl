# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

EgoDex is a dataset and benchmark for egocentric dexterous manipulation (829 hours, 30 Hz, 1080p video + 3D SE(3) pose annotations for 68 hand/body joints) collected with ARKit on Apple Vision Pro. This repo contains sample code for loading, visualizing, and evaluating models on EgoDex data.

## Installation

```bash
conda create --name egodex python==3.11
conda activate egodex
conda install -c conda-forge ffmpeg=7.1.1
pip install -r requirements.txt
```

## Running the Scripts

```bash
# Visualize 3D skeletal annotations reprojected into 2D video
python visualize_2d.py --data_dir [path/to/egodex] --num_episodes 1 --output_mp4 output.mp4

# Visualize in 3D (interactive plotly)
python visualize_3d.py --data_dir [path/to/egodex]
```

## Data Format

Each task folder contains paired `.hdf5` + `.mp4` files (same index = same episode). HDF5 structure:

- `camera/intrinsic` — fixed 3×3 intrinsics (`[[736.63, 0, 960], [0, 736.63, 540], [0,0,1]]`)
- `transforms/<joint_name>` — shape `N×4×4` SE(3) poses in **ARKit origin frame** (stationary world frame set at recording start, not consistent across episodes)
- `confidences/<joint_name>` — shape `N` scalar ARKit confidence (0–1), optional
- `f.attrs['llm_description']` — natural language task description
- `f.attrs['llm_type']` — `'reversible'` or otherwise; if reversible, check `f.attrs['which_llm_description']` (1 or 2)

All 68 joint names are defined in `utils/skeleton_tfs.py` as grouped constants: `LEFT_FINGERS`, `RIGHT_FINGERS`, `LEFT_ARM`, `RIGHT_ARM`, `SPINE`, `NECK`, `WRISTS`.

## Key Architecture

**`simple_dataset.py`** — `SimpleDataset(dataset_path, query_tfs)`: PyTorch Dataset that wraps `index_episodes()` to discover all `.hdf5` files and precompute cumulative lengths. `__getitem__` returns `(tfs, cam_ext, cam_int, img, lang_instruct, confs)`. Video frames are decoded on-the-fly with `torchcodec`.

**`utils/data_utils.py`**:
- `index_episodes(path)` — walks directory tree, collects all HDF5s, returns `(file_list, lengths)`
- `convert_to_camera_frame(tfs, cam_ext)` — transforms SE(3) poses from ARKit world frame to camera frame: `inv(cam_ext) @ tfs`

**`compute_metrics.py`** — `evaluate_distance(gt_actions, gt_padding, pred_actions, best_k)`: computes best-of-K distance over action chunks. Action space assumed 48-dim (12 keypoint positions × 3D + 6D rotations interleaved); position indices `[0:3] + [9:27] + [33:48]` extract the 12×3 positions.

## Coordinate Frames Note

Transforms are in the ARKit origin frame by default. For learning, you likely want camera-frame coordinates — use `convert_to_camera_frame()`. The 2D reprojection in `visualize_2d.py` has known perspective error because the RGB video is synthesized from multiple Vision Pro cameras.

## Evaluation

The benchmark metric is **best-of-K distance** (average and final-frame Euclidean distance over 12 hand keypoints). Run `evaluate_distance()` over the full test set; only non-padded samples are counted.
