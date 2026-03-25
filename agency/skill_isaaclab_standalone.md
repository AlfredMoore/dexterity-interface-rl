# IsaacLab Standalone Deployment Knowledge

## Overview

IsaacLab standalone mode uses `SimulationContext` + `InteractiveScene` directly,
bypassing `ManagerBasedEnv` or `DirectRLEnv`. This is ideal for replay/visualization
tasks where you don't need RL action/reward/done pipelines.

## Key Pattern

```python
from isaaclab.app import AppLauncher
# AppLauncher MUST come before any other isaaclab imports
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True  # Required for CameraCfg
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# Now import isaaclab modules
import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg, AssetBaseCfg
from isaaclab.sensors import CameraCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sim.spawners.from_files import UrdfFileCfg
from isaaclab.sim.converters.urdf_converter_cfg import UrdfConverterCfg
from isaaclab.utils import configclass

# Scene definition
@configclass
class MySceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = ...
    ground = AssetBaseCfg(...)

# Main loop
sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1/120))
scene = InteractiveScene(MySceneCfg(num_envs=1, env_spacing=2.5))
sim.reset()

while running:
    articulation.write_joint_state_to_sim(pos, vel)
    scene.write_data_to_sim()
    sim.step()
    scene.update(dt)
```

## Critical Gotchas

### 1. UrdfFileCfg `joint_drive.gains.stiffness` MISSING error
Default `JointDriveCfg()` has `PDGainsCfg(stiffness=MISSING)`. Even with
`target_type="none"`, validation runs first. Must explicitly set:
```python
joint_drive=UrdfConverterCfg.JointDriveCfg(
    target_type="none",
    gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0),
)
```

### 2. URDF joint names have no prefix
When loading separate left/right arm URDFs, joint names are the raw URDF names
(e.g., `panda_joint1`, `F1M1`) NOT `left_panda_joint1`. Must strip prefix when
mapping from 38-DOF trajectory to per-arm articulation joints.

### 3. Camera requires `--enable_cameras` flag
Without `args.enable_cameras = True` before AppLauncher, CameraCfg will throw:
`RuntimeError: A camera was spawned without the --enable_cameras flag`

### 4. Headless rendering + video
- Camera RGB data: `scene["camera"].data.output["rgb"]` → `(num_envs, H, W, 4)` uint8 tensor
- Use `imageio.get_writer()` to save frames as MP4
- Headless mode auto-detected from `--headless` flag
- First `sim.reset()` with cameras takes minutes (shader compilation)

### 5. Kinematic workstation
For a table/workstation that shouldn't move:
```python
RigidObjectCfg(
    spawn=sim_utils.CuboidCfg(
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
    ),
)
```

### 6. Teleport (zero stiffness/damping)
Set actuators with `stiffness=0.0, damping=0.0` and use `write_joint_state_to_sim()`
to directly set joint positions each frame. No PD controller involved.

## File Locations
- Standalone replay: `baselines/ml-egodex-HAND/replay/standalone_replay.py`
- Retarget tool: `baselines/ml-egodex-HAND/retarget/retarget_tool.py`
- Bottle URDF: `baselines/ml-egodex-HAND/replay/bottle_egodex/model.urdf`
- Reference: `IsaacLab-HAND/scripts/tutorials/02_scene/create_scene.py`
- Reference: `dexterity-interface/libs/robot_motion_interface/.../isaacsim_interface.py`
