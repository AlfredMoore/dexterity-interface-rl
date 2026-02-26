# Bimanual Panda Tesollo Configuration Files

This directory contains all configuration files for the bimanual Panda with Tesollo gripper system.

## 📁 Directory Structure

```
configs/
├── robot/
│   ├── bimanual_panda_tesollo.yml           # Main robot configuration
│   └── spheres/
│       └── bimanual_panda_tesollo.yml       # Collision sphere definitions
└── world/
    └── bimanual_table.yml                   # World obstacles (table)
```

## 📝 File Descriptions

### Robot Configuration
**File**: `robot/bimanual_panda_tesollo.yml`

Main robot configuration including:
- Kinematic structure (38 DoF)
- Joint names and limits
- Collision checking setup
- Self-collision pairs
- End-effector links

**Usage in code**:
```python
from curobo.wrap.reacher.motion_gen import MotionGenConfig

motion_gen_cfg = MotionGenConfig.load_from_robot_config(
    robot_cfg="robot/robot_description/configs/robot/bimanual_panda_tesollo.yml",
    # ... other parameters
)
```

### Collision Spheres
**File**: `robot/spheres/bimanual_panda_tesollo.yml`

Simplified sphere representation for fast collision checking. Each link is approximated by one or more spheres.

**Referenced by**: `robot/bimanual_panda_tesollo.yml`
```yaml
collision_spheres: 'spheres/bimanual_panda_tesollo.yml'
```

### World Configuration
**File**: `world/bimanual_table.yml`

World obstacles and environment configuration:
- Work table: 1.8288m × 0.62865m × 0.045m
- Position matches URDF mounting points

**Usage in code**:
```python
motion_gen_cfg = MotionGenConfig.load_from_robot_config(
    robot_cfg="...",
    world_cfg="robot/robot_description/configs/world/bimanual_table.yml",
)
```

## 🔧 Customization

### Modify Robot Configuration

Edit `robot/bimanual_panda_tesollo.yml`:

```yaml
cspace:
  max_jerk: 500.0           # Motion smoothness
  max_acceleration: 15.0    # Motion speed
  retract_config: [...]     # Home position
```

### Add World Obstacles

Edit `world/bimanual_table.yml` or create a new world config:

```yaml
cuboid:
  table:
    dims: [1.8288, 0.62865, 0.045]
    pose: [0.0, 0.0, 0.89175, 1, 0, 0, 0]

  # Add more obstacles
  box1:
    dims: [0.2, 0.2, 0.3]
    pose: [0.5, 0.0, 1.0, 1, 0, 0, 0]
```

### Adjust Collision Spheres

Edit `robot/spheres/bimanual_panda_tesollo.yml`:

```yaml
collision_spheres:
  left_panda_link1:
    - center: [0, 0, -0.08]
      radius: 0.08      # Increase for more conservative collision checking
```

## 🌍 Path Resolution

CuRobo resolves configuration paths relative to its content directory. The prefix `robot/` refers to the robot content root, which includes:

- `robot/robot_description/` - Your custom robot descriptions
- `robot/ur_description/` - Built-in UR robot descriptions
- etc.

## 🔗 Related Files

- **URDF Generator**: `../composites/generate_bimanual_panda_tesollo.py`
- **Generated URDF**: `../rl/bimanual_panda_tesollo.urdf`
- **Example Code**: `../../examples/bimanual_motion_gen_example.py`
- **Documentation**: `../../examples/BIMANUAL_README.md`

## ✅ Validation

Verify all configurations are correct:

```bash
# From project root
python -c "
from pathlib import Path
configs = [
    'robot_description/configs/robot/bimanual_panda_tesollo.yml',
    'robot_description/configs/robot/spheres/bimanual_panda_tesollo.yml',
    'robot_description/configs/world/bimanual_table.yml',
]
for c in configs:
    print(f'{"✓" if Path(c).exists() else "✗"} {c}')
"
```

---

**Note**: These configuration files are project-specific and stored in `robot_description/` rather than the curobo source tree. This keeps your custom configurations separate from the library code.
