# Benchmark: EgoDex → Bimanual Tesollo DG-3F → Diffusion Policy → Real-World 部署

## 0. 前提假设（已确认）

| 项目 | 选择 | 理由 |
|------|------|------|
| 机器人 | 双臂 Franka Panda (7-DOF) + **Tesollo DG-3F 灵巧手 (12-DOF)** | 与实验室实际硬件一致 |
| 任务 | 拧瓶盖（unscrew/screw cap）— 双臂：左手扶瓶，右手拧盖 | EgoDex 中有对应 episode (`add_remove_lid`) |
| 相机 | 第三视角固定相机（RealSense D4xx）+ 可选腕部相机 | 与 Diffusion Policy 原版设置一致 |
| Action space | **关节空间 38-DOF**：`[左Panda×7, 左Tesollo×12, 右Panda×7, 右Tesollo×12]` | 与 rl_driver 接口直接对接 |

---

## 1. 数据筛选与提取

> **状态：待实现（可选，留到后期）**

EgoDex 数据集：829 小时，30Hz，1080p egocentric 视频 + ARKit 3D 骨骼姿态（194 种任务）。每个 episode 由配对的 `N.mp4` + `N.hdf5` 组成。

### 数据结构（HDF5）

- `/transforms/<joint_name>` — N×4×4 SE(3)，ARKit 世界坐标系
- `/transforms/camera` — 相机外参
- `/camera/intrinsic` — 固定内参（fx=fy=736.63, cx=960, cy=540）
- `f.attrs['llm_description']` — GPT-4 任务描述
- `/confidences/` — 关节置信度（可选）

### 筛选策略（简版，已集成到 run_retarget.py）

```
EgoDex HDF5
    ↓ llm_description 关键词匹配：cap / lid / twist / open / close / unscrew / screw
    ↓ 可视化随机抽检（待实现：filter_episodes.py）
```

注意：GPT-4 语言标注有噪声，建议人工抽检。

---

## 2. Retargeting：人手 → 双臂机器人关节空间

> **状态：✅ 已实现**

### 2.1 数据流

```
EgoDex HDF5 (ARKit world frame)
    │
    ├─ rightHand / leftHand SE(3) 4×4
    │       ↓ ARKit world → camera frame（inv(cam_ext) @ tf）
    │       ↓ CoordinateAligner（相机系 → 机器人 base frame，可配置 R,t）
    │       ↓ PandaArmIKSolver（pinocchio Jacobian IK）
    │       → Panda 7 joint angles（per arm）
    │
    └─ {right/left}{Thumb/Index/Middle}FingerTip SE(3)
            ↓ ARKit world → camera frame → robot frame（取 pos[:3,3]）
            ↓ HandRetargeter（dex-retargeting PositionOptimizer + NLopt）
            → Tesollo 12 joint angles（per hand）
    │
    ↓ 输出 (T, 38) npz：[left_panda, left_tesollo, right_panda, right_tesollo]
```

### 2.2 人手 → 机器人 关节映射

| EgoDex 关节 | 机器人目标 |
|-------------|-----------|
| `rightHand` SE(3) | 右 Franka EE → `right_panda_joint{1-7}` |
| `rightThumbTip` pos | `right_F1_TIP`（Tesollo F1 指尖） |
| `rightIndexFingerTip` pos | `right_F2_TIP` |
| `rightMiddleFingerTip` pos | `right_F3_TIP` |
| `leftHand` SE(3) | 左 Franka EE → `left_panda_joint{1-7}` |
| `leftThumbTip` pos | `left_F1_TIP` |
| `leftIndexFingerTip` pos | `left_F2_TIP` |
| `leftMiddleFingerTip` pos | `left_F3_TIP` |

### 2.3 已实现文件

| 文件 | 功能 |
|------|------|
| `retarget/utils/extract_hand_urdf.py` | 从 bimanual URDF 提取独立 Tesollo URDF（BFS 遍历） |
| `retarget/assets/tesollo_dg3f_{left,right}.urdf` | 提取出的单手 URDF（20 links，12 revolute joints） |
| `retarget/config/tesollo_{left,right}.yaml` | dex-retargeting PositionOptimizer 配置 |
| `retarget/retarget_episode.py` | 核心模块：`CoordinateAligner`、`HandRetargeter`、`PandaArmIKSolver`、`retarget_episode()` |
| `retarget/run_retarget.py` | 批处理脚本，支持关键词过滤、limit 诊断 |

### 2.4 运行方式

```bash
docker exec handrl-policy bash -c "
cd /workspace
/root/miniconda3/envs/policy/bin/python \
  baselines/ml-egodex-HAND/retarget/run_retarget.py \
  --data_dir /path/to/egodex \
  --output_dir /path/to/retargeted \
  --keywords 'cap,lid,twist' \
  --max_episodes 1 \
  --check_limits
"
```

### 2.5 关键注意事项

1. **坐标系对齐是最大的 domain gap 来源**。`CoordinateAligner` 默认 identity（相机坐标系）。真机部署前需提供标定的 R/t（`--cam_R`、`--cam_t` 参数）。
2. **dex-retargeting 环境**：需使用 `/root/miniconda3/envs/policy/bin/python`（含 torch + conda-forge pinocchio 3.9）。系统 `/opt/openrobots` 的 pinocchio 与 NumPy 2.x 不兼容，retarget 模块会自动排除。
3. **`add_dummy_free_joint: true`**：手的根部自由浮动，每个 episode 开始时调用 `warm_start()` 加速收敛。
4. **关节输出顺序**严格遵循：`[左Panda×7, 左Tesollo×12, 右Panda×7, 右Tesollo×12]`。

### 2.6 待验证

- [ ] 在 IsaacSim 中回放 retarget 轨迹，目视检查运动合理性
- [ ] `--check_limits` 统计 IK 成功率与关节速度平滑度
- [ ] 标定 `CoordinateAligner` 的 R/t（相机系 → 机器人 base frame）

---

## 3. Diffusion Policy 训练

> **状态：待实现**

### 3.1 数据格式（38-DOF 关节空间）

```python
observation = {
    "joint_positions": (T_obs, 38),   # 当前关节状态
    # 可选：
    "image": (T_obs, 3, 96, 96),      # 第三视角相机图像
}

action = {
    "joint_positions": (T_pred, 38),  # 未来 T_pred 步的关节目标
}
```

### 3.2 推荐超参数（baseline）

| 参数 | 值 |
|------|----|
| Observation horizon (T_obs) | 2 |
| Prediction horizon (T_pred) | 16 |
| Action horizon (T_act) | 8 |
| Diffusion steps | 100 (train) / 10 (DDIM infer) |
| 网络 | 1D temporal U-Net |
| Batch size | 256 |
| LR | 1e-4, cosine decay |

### 3.3 两阶段方案

| 阶段 | 数据源 | 目的 |
|------|--------|------|
| **Phase 1** | EgoDex retarget 轨迹 | 学习任务运动模式，减少真机 demo 需求 |
| **Phase 2** | 真机 teleoperation demo（10–50 条） | 弥补 retargeting domain gap |

---

## 4. 评估指标

| 指标 | 说明 |
|------|------|
| **Retargeting 质量** | IK 成功率、关节速度平滑度（`--check_limits`） |
| **Offline best-of-K** | `compute_metrics.py` 中的距离指标（可选，需预测模型） |
| **Sim 成功率** | IsaacSim 中闭环测试 100 次成功率 |
| **Real-world 成功率** | 真机 20 次试验成功率（核心指标） |

---

## 5. Real-World 部署 Pipeline

```
RealSense (30Hz)
    ↓ 观测编码（joint_positions + 可选 image）
┌──────────────────────┐
│   Diffusion Policy   │
│  (DDIM 10步, ~50ms)  │
└──────────────────────┘
    ↓ action chunk (8步 × 38-DOF)
rl_driver（/target_joint_states，200Hz 插值执行）
    ├─ Franka Panda（libfranka FCI）
    └─ Tesollo DG-3F（Modbus TCP）
```

**部署注意：**
- 推理：DDIM 10 步 GPU 约 50ms，满足实时性
- 安全：力矩阈值 + workspace boundary，超限立即停止
- 初期：夹具固定瓶身，验证右臂拧盖；验证通过后放开左臂扶瓶

---

## 6. 实验路线（最小可行）

```
[✅] Step 1  数据了解    — EgoDex 数据结构、CLAUDE.md、简单数据加载测试
[✅] Step 2  Retargeting — 实现完成，待真实数据验证
[ ]  Step 2' 坐标标定    — 标定 CoordinateAligner R/t，IsaacSim 回放验证
[ ]  Step 3  Policy 训练 — Diffusion Policy 在 retarget 数据上 offline 训练
[ ]  Step 4  Sim 验证    — IsaacSim 闭环测试
[ ]  Step 5  真机 fine-tune — 采集少量真机 demo，fine-tune
[ ]  Step 6  真机部署    — 真机测试，统计成功率
```

---

## 附：环境配置

```bash
# 安装 dex-retargeting 到 policy 环境（已完成）
docker exec handrl-policy bash -c "
  /root/miniconda3/envs/policy/bin/pip install -e /workspace/baselines/dex-retargeting/
  /root/miniconda3/bin/conda install -n policy -c conda-forge pinocchio -y
"

# 重新生成 Tesollo URDF（如 bimanual URDF 更新后）
docker exec handrl-policy bash -c "
  cd /workspace && python3 baselines/ml-egodex-HAND/retarget/utils/extract_hand_urdf.py
"
```
