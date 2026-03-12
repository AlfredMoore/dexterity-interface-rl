# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 编码规范

- 所有代码注释使用**英文**。

## 项目概述

本项目是一个用于 RL 策略真实部署的双臂灵巧操作系统，连接两个 Franka Panda 机械臂（各 7 自由度）和两个 Tesollo DG-3F 灵巧手（各 12 自由度），通过 ROS 2 通信。训练好的 RL 策略（来自 IsaacLab/RSL-RL）在策略计算机上推理，通过 ROS 2 将关节目标发送给驱动计算机。

**双机部署：**
- **驱动计算机**（`192.168.4.x` 子网）：运行 `rl_driver` 节点，通过以太网直连 Panda FCI 和 Tesollo 灵巧手。
- **策略/推理计算机**：运行 `rl_policy` 和 `cv_node`，配有 GPU 和 USB 接口的 RealSense 相机。

---

## Docker 环境

### 驱动容器（驱动计算机）
```bash
docker build -f docker/Dockerfile.driver -t dex-driver .
docker run --name handrl-driver --rm -it --privileged -v $(pwd):/workspace --net=host dex-driver
```

### 策略容器（推理计算机）
```bash
docker build -f docker/Dockerfile.policy_py312 -t dex-policy .
xhost +local:docker
docker run --name handrl-policy --rm -it --privileged --gpus all \
  -v $(pwd):/workspace --device /dev/bus/usb:/dev/bus/usb \
  --net=host -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix dex-policy
```

---

## Python 环境配置（非 Docker）

需要 Ubuntu 22.04/24.04、Python 3.11、`libeigen3-dev`。

```bash
python3.11 -m venv venv-dex && source venv-dex/bin/activate
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
pip install isaaclab[isaacsim,all]==2.2.0 --extra-index-url https://pypi.nvidia.com

# 安装所有本地库
pip install -e libs/robot_motion
pip install -e libs/robot_motion_interface
pip install -e libs/isaacsim_ui_interface/
pip install -e libs/sensor_interface/sensor_interface_py
pip install -e libs/primitives/primitives_py
pip install -e libs/planning/planning_py
```

**robot_motion** 依赖 Pinocchio（通过 robotpkg/apt 安装，需在 `~/.bashrc` 中设置环境变量）。
**robot_motion_interface** 依赖 libfranka 0.9.2（详见 `libs/robot_motion_interface/README.md`）。

---

## ROS 2 构建与运行

所有 ROS 2 代码在 `libs/robot_motion_interface/ros/` 下，建议在驱动容器内执行：

```bash
cd /workspace/libs/robot_motion_interface/ros
colcon build --symlink-install
source install/setup.bash
```

### 主要 ROS 2 节点

| 节点 | 入口文件 | 功能 |
|------|----------|------|
| `rl_driver` | `rl_driver_node.py` | 驱动计算机运行，控制 Panda + Tesollo 硬件 |
| `rl_policy` | `rl_policy_node.py` | 策略计算机运行，加载训练策略进行推理 |
| `cv_node` | `cv_node.py` | RealSense 采集 + PromptDA 度量深度 + 目标检测 |
| `interface` | `interface_node.py` | 通用双臂/Panda/Tesollo ROS 接口 |
| `test_pre_grasp` | `test_node_pre_grasp.py` | 测试预抓取轨迹 |
| `test_curobo` | `test_node_for_curobo.py` | 测试 cuRobo 运动规划 |

```bash
# 驱动计算机 — 启动硬件接口
export ROS_STATIC_PEERS='192.168.4.9;remote.com'
ros2 run robot_motion_interface_ros rl_driver

# 策略计算机 — 启动视觉感知
ros2 run robot_motion_interface_ros cv_node

# 策略计算机 — 启动 RL 策略推理
ros2 run robot_motion_interface_ros rl_policy
```

### 主要 ROS 2 话题

| 话题 | 消息类型 | 方向 |
|------|----------|------|
| `/joint_states` | `sensor_msgs/JointState` | 驱动 → 策略（1 kHz，尽力而为） |
| `/target_joint_states` | `sensor_msgs/JointState` | 策略 → 驱动（可靠） |
| `/object_detection` | `vision_msgs/Detection3D` | 视觉 → 策略（尽力而为） |

关节顺序（共 38 自由度）：`[左Panda×7, 左Tesollo×12, 右Panda×7, 右Tesollo×12]`

---

## 系统架构

```
rl_policy_node  ──/target_joint_states──►  rl_driver_node
       ▲                                         │
       │ /joint_states (1kHz)                    ├─ PandaInterface（libfranka C++ pybind）
       │                                         └─ TesolloInterface（Modbus TCP）
cv_node ──/object_detection──► rl_policy_node
  （RealSense + PromptDA 度量深度）
```

### 库结构

- **`libs/robot_motion/`** — C++/Python：底层控制器、IK（RelaxedIK、pinocchio）、机器人属性，pybind11 封装。
- **`libs/robot_motion_interface/`** — Python + C++ pybind11：`Interface` 抽象基类，实现有 `PandaInterface`、`TesolloInterface`、`IsaacSimInterface`、`BimanualInterface`。配置文件在 `config/`，ROS 2 封装在 `ros/`。
- **`libs/robot_description/`** — Panda、Tesollo DG-3F 及组合体的 URDF/xacro 文件；`rl/` 下存放预生成的 RL 专用 URDF。
- **`libs/sensor_interface/`** — Python/C++：相机接口（RealSense、Kinect）。
- **`libs/isaacsim_ui_interface/`** — Python：IsaacSim 流媒体 UI 接口。
- **`libs/primitives/`** — Python + ROS 封装：运动基元。
- **`libs/planning/`** — Python：LLM 任务规划（GPT）、YOLO 感知。
- **`dep/`** — 第三方依赖：`PromptDA-HAND`（提示式度量深度）、`sam2-HAND`（分割）。
- **`app/`** — FastAPI 后端 + Vite/JS 前端，用于浏览器操作界面（实验性）。

### Interface 抽象基类（`libs/robot_motion_interface/src/robot_motion_interface/interface.py`）

所有硬件和仿真后端均实现 `Interface`：
- `set_joint_positions(q, joint_names, blocking)` — 支持部分关节更新
- `set_cartesian_pose(x, cartesian_order, base_frame, ee_frames, blocking)`
- `joint_state()` → `np.ndarray` 形状 `(n_joints*2,)`，前半为位置，后半为速度
- `home(blocking)`、`start_loop()`、`stop_loop()`

### RL 策略节点（`rl_policy_node.py`）

1. 从 `policy_run_dir`（在 `rl_policy_node_config.yaml` 中设置）加载 RSL-RL 训练策略。
2. 订阅 `/joint_states` 和 `/object_detection`。
3. 使用 pinocchio 正向运动学计算指尖/手掌根坐标作为观测。
4. 输出经 EMA 平滑、软限位裁剪的关节位置增量目标。
5. 发布到 `/target_joint_states`。

策略运行目录须包含：`params/env.yaml`、`params/agent.yaml`、`exported/runtime_cfg.yaml`、`exported/policy.pt`。

### 配置文件

- `libs/robot_motion_interface/config/rl_bimanual_driver_config.yaml` — 硬件 IP、PD 增益、驱动节点关节名称。
- `libs/robot_motion_interface/config/rl_policy_node_config.yaml` — 策略路径、RealSense 内外参、PromptDA 检查点。

---

## 实用脚本

```bash
# 测试环境导入（在策略容器内运行）
python /workspace/init/test_env.py

# RealSense 连通性测试
python -m robot_motion_interface.utils.realsense_test

# 运动学/IK 测试（pinocchio + cuRobo）
python -m robot_motion_interface.utils.kinematics

# 预抓取 SE3 位姿采样
python -m robot_motion_interface.utils.pose_sampler

# 回原位到预抓取轨迹生成（耗时，需过滤约 5 万个关节位姿）
python -m robot_motion_interface.utils.traj_sampler

# URDF → USD 转换（用于 IsaacSim）
python3 -m robot_motion_interface.isaacsim.utils.urdf_converter \
    path/to/robot.urdf path/to/out/robot.usd \
    --fix-base --joint-stiffness 0.0 --joint-damping 0.0 --joint-target-type none
```

---

## App UI（实验性）

在仓库根目录激活 `venv-dex` 后，分别启动三个进程：

```bash
# 1. IsaacSim 流媒体模式
LIVESTREAM=2 python3 -m robot_motion_interface.isaacsim.isaacsim_interface \
  --kit_args="--/app/window/hideUi=true --/app/window/drawMouse=false"

# 2. FastAPI 后端
uvicorn ui_backend.api:app --reload

# 3. 前端开发服务器
npm run dev --prefix app/ui_frontend/
```

访问地址：API `http://127.0.0.1:8000`，前端 `http://127.0.0.1:3000`。

---

## 硬件说明

- Panda 机器人系统版本 4.2.x，仅兼容 **libfranka 0.9.2**。
- Tesollo DG-3F 通过 **Modbus TCP**（pymodbus）通信，需切换到外部控制模式。
- RealSense D4xx 用于 RGB-D 输入；内外参在 `rl_policy_node_config.yaml` 中配置。
- 驱动计算机需安装**实时内核补丁**以满足 Panda FCI 要求。
- 网络地址：左/右 Panda 分别为 `192.168.4.2` / `192.168.4.3`，左/右 Tesollo 为 `192.168.4.8` / `192.168.4.7`。
- 若 Panda 报错 "command not possible in current mode (User stopped)"，请在 Franka 控制面板检查并释放急停按钮。
