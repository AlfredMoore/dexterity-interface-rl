# Robot Kinematics Reference

## Robot: 2× Franka Panda + 2× Tesollo DG-3F = 38 DOF

**Joint order:** `[left_panda×7, left_tesollo×12, right_panda×7, right_tesollo×12]`

**URDF:** `libs/robot_description/rl/bimanual_panda_tesollo.urdf`

## Kinematic Chain (per arm)

```
panda_joint1..7 (7 revolute)
    → panda_link8 (wrist flange)
        → [fixed: xyz=0 0 0.106, rpy=0 0 -0.785]
        → delto_base_link (palm hub)
            ├─ F1M1..F1M4 (4 revolute) → F1_TIP → F1_TIP_TOP [fixed, +58mm]
            ├─ F2M1..F2M4 (4 revolute) → F2_TIP → F2_TIP_TOP [fixed, +58mm]
            └─ F3M1..F3M4 (4 revolute) → F3_TIP → F3_TIP_TOP [fixed, +58mm]
```

## Key Dimensions

| Item | Value |
|------|-------|
| Wrist → palm base (z) | 0.106 m |
| Wrist → palm base (yaw) | -0.785 rad (-45°) |
| F1 (thumb) base offset | xyz=0.0265, 0, 0 from delto_base |
| F2 (index) base offset | xyz=-0.01334, 0.023, 0 from delto_base |
| F3 (middle) base offset | xyz=-0.01334, -0.023, 0 from delto_base |
| Finger chain length | ~80-100mm (base to tip) |
| delto_offset_link | z+0.05 from delto_base (existing reference frame) |

## Finger-Human Mapping

| EgoDex Joint | Robot Link |
|-------------|------------|
| leftThumbTip / rightThumbTip | F1_TIP |
| leftIndexFingerTip / rightIndexFingerTip | F2_TIP |
| leftMiddleFingerTip / rightMiddleFingerTip | F3_TIP |

Note: Human ring + pinky fingers have no robot counterpart (Tesollo is 3-finger).

## IK Tools

| Tool | DOF | File | Method |
|------|-----|------|--------|
| PandaArmIKSolver | 7 (arm) | `retarget_episode.py:109-246` | Pinocchio DLS, position-only or SE3 |
| FixedBaseHandRetargeter | 12 (fingers) | `retarget_episode.py:327-375` | dex-retargeting NLopt, hand-base frame |
| CuRoboRetargetIK | 38 (full) | `curobo_ik.py` | cuRobo GPU IK, 6 fingertip targets |
| PinocchioBimanualFK | FK only | `libs/.../kinematics.py:65-167` | Full bimanual FK |

## Coordinate Frames

- **EgoDex camera (ARKit):** X-right, Y-up, Z-toward-viewer
- **Bimanual URDF world:** X-right, Y-back, Z-up
- **Rotation camera→URDF:** `R = [[1,0,0],[0,0,-1],[0,1,0]]`
- **Bottle in URDF:** (0.042, 0, -0.0215)
