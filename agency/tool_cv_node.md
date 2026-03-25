# CV Node Overview (`cv_node.py`)

This note summarizes what the current ROS node does in:
`libs/robot_motion_interface/ros/src/robot_motion_interface_ros/robot_motion_interface_ros/cv_node.py`.

## 1. Node Role

- Node name: `cv_perception_node`
- Main purpose: run online RGB-D perception from RealSense, then execute:
  - PromptDA depth refinement (raw depth -> metric depth)
  - SAM3 semantic segmentation
- Current visible output is mainly a debug preview window.

## 2. Inputs / Configuration

- ROS parameter:
  - `config_path` (default: `libs/robot_motion_interface/config/rl_policy_node_config.yaml`)
- Config fields used:
  - `infer_rate`
  - `promptda_ckpt`, `sam3_ckpt`
  - `realsense` block: fps, intrinsics, sensor settings
- Camera input:
  - RealSense `color` + `depth` streams
  - Depth is aligned to color (`rs.align(rs.stream.color)`)

## 3. Runtime Pipeline

1. **Initialization**
- Starts RealSense pipeline (color + depth).
- Applies sensor options (auto-exposure / exposure / gain).
- Builds depth filter chain:
  - decimation (`x2`)
  - hole filling
- Starts a background capture thread to keep latest `(color, depth)` frame pair.

2. **Model loading**
- PromptDA model loaded via `PromptDAInference`.
- SAM3 model loaded via `SAM3Inference`.

3. **Periodic inference (`_infer_callback`)**
- Triggered by timer at `infer_rate`.
- Reads latest frame pair under lock.
- Runs PromptDA:
  - input: `color(BGR uint8) + depth(uint16)`
  - output: `metric_depth` tensor (meters)
- Runs SAM3 segmentation on color frame.
- Builds a debug visualization:
  - Row 1: RGB / raw depth colormap / metric depth colormap
  - Row 2: SAM3 overlays for left arm / right arm / cup
- Displays via `cv2.imshow(...)`.

## 4. ROS Topics

- Publisher created:
  - `/object_detection` (`vision_msgs/Detection3D`)
- **Current status**: no `Detection3D` message is actually published yet.
  - Pose estimation and publish logic are marked as TODO.

## 5. What Is Implemented vs TODO

Implemented:
- RealSense acquisition + alignment + filtering
- PromptDA inference
- SAM3 inference
- Timing logs and visual debugging panel

TODO / missing functionality:
- 3D object pose estimation from `metric_depth + masks/boxes`
- Packing and publishing `Detection3D` to `/object_detection`

## 6. Current Integration Risks (from code inspection)

- `cv_node.py` imports:
  - `CONCEPT_LEFT_ARM`, `CONCEPT_RIGHT_ARM`, `CONCEPT_CUP`
  from `sam3_utils.py`.
- Current `sam3_utils.py` defines `CONCEPT_ARM`, `CONCEPT_HAND`, `CONCEPT_OBJ` and requires `concept_map` in `SAM3Inference(...)`.
- `cv_node.py` currently calls `SAM3Inference(...)` without passing `concept_map`.
- This indicates a likely API mismatch between `cv_node.py` and current `sam3_utils.py` version.

In short: the node structure is complete for capture+inference+preview, but ROS 3D detection output is not finished, and SAM3 interface compatibility likely needs a quick sync fix.
