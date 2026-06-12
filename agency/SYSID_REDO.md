# SYSID 重做计划 — Franka Panda + Tesollo DG-3F (cap-twist sim2real)

本计划**取代并不参照**旧的 `.claude/SYSID.md` 与 `.claude/SYSID_PLAN.md`(两者已废弃)。
目标:基于在 `hand_mjlab` 训练的 cap-twist 策略,对真机系统做一次**认真的** system
identification —— 上一次 sysid 比较随意(直接用 LSE 选 kp/kd,没考虑重力补偿和其它
参数),这次要把重力补偿/负载、plant 层参数、控制延迟逐项查清并整轨匹配。

- 真机驱动仓库:`/home/hci-lab/mo/dexterity-interface-rl`(ROS2 双臂驱动 + libfranka + Tesollo)
- sim 仓库:`/home/hci-lab/mo/hand_mjlab`(mjlab / MuJoCo-Warp)
- 状态:**真机采集代码(第 0/1 步)已写完,未编译**;sim 拟合(第 2 步)待真机数据回来再写。

---

## 真机数据采集(简版)

环境:**dex 驱动容器**,工作目录 `/workspace`,先 `pip install -e libs/robot_motion_interface`
编好 pybind。**全程不开 `rl_driver`/`rl_policy`**(sysid 脚本独占该臂 FCI 连接)。
脚本是纯 Python(非 ROS 节点),`python -m robot_motion_interface.sysid.<...>`。

1. **重力/负载诊断**(第 0 步,只读,即用):左右各一次 `sysid_gravity_diag --gravity-at-home
   --out ...`,再 `--compare` 出左右对比 → 定位右臂过补;可选 `--set-load-mass --verify` 试代码修正。
2. **激励采集**(第 1 步):`sysid_excitation --arm {left,right} --mode {simple,random} --out ...`
   → 每臂每模式一份 npz(记录 `q_target/q/dq`)。
   ⚠️ **前置**:需 `runtime/runtime_cfg.yaml`(dt/EMA/vel_scale/软限位)—— 仓库里没有,从训练策略导出
   (`<policy_run_dir>/exported/runtime_cfg.yaml`)放过去或 `--runtime_cfg` 指过去。
3. **交接**:把 `runtime/system_id/` 下的 `*.npz`、`*.json` copy 到 `hand_mjlab/logs/sysid/`,
   sim 侧拟合(第 2 步)再处理。

完整命令见第 8 节;每步的设计/依据见第 0–3 节。

---

## 0. 已核实事实(读代码,带出处)

### 真机侧 `dexterity-interface-rl`
- 控制律 [joint_torque_controller.cpp:39]:`τ = coriolis(q,dq) + kp·e + kd·(−dq) (+ gravity 仅当 flag)`;
  误差 `e` 先 clamp 到 ±`max_joint_delta=0.3` 再进 PD [joint_torque_controller.cpp:33-35]。
- 控制器构造 `gravity_compensation = false` [panda_interface.cpp:28] → 控制器**自己不补重力**,
  全靠 libfranka FCI 内部模型补(torque 控制模式下 FCI 把命令力矩叠加在按配置 load 算出的
  重力之上)。两臂**都没有 `setLoad`** → 重力补偿完全由 Desk(FCI web)里配置的 load 决定。
- FCI 回调里 `robot_state` 已含 `tau_J`(实测连杆力矩)与 `tau_ext_hat_filtered`(扣模型外力矩估计)
  [panda_interface.cpp:68-83];但当前 `joint_state()` 只回 14 维 (q,q̇) [panda_interface.cpp:38-54],
  **ROS / 接口层都没有 τ**。
- 增益(= sim 标称):arm kp=[60,80,60,80,25,70,15] / kd=[20,20,20,20,10,20,5];
  finger kp=4.7 / kd=0.2 [rl_bimanual_driver_config.yaml:7-14]。
- 关节顺序 **38 维 = [左Panda7, 左Tesollo12, 右Panda7, 右Tesollo12]** [rl_bimanual_driver_config.yaml:45-60]。
- `rl_driver_node` 启动时**两臂都拉起并 home**,`target_callback` 期望**完整 38 维**并硬切给四个 interface
  [rl_driver_node.py:73-99, 168-177];`/joint_states` 由 timer 以 **200Hz** 发布
  (`joint_state_pub_timer_freq:200`,非 CLAUDE.md 表里写的 1kHz)[rl_driver_node.py:118]。
- `BimanualInterface` 支持 `enable_left`/`enable_right` **单臂拉起**,`set_joint_positions(q, joint_names=…)`
  支持**部分关节**更新 [bimanual_interface.py:56-77, 161-189]。已有 `examples/oscillating_ex.py`
  用 `BimanualInterface` 直接做逐关节正弦的模式(本计划新节点的雏形)。
- `utils/sysid_action_traj_gen.py`:逐关节激励 raw action 生成器 + policy↔real 关节序映射。
  **仅作概念参考,本次不复用**(自己写新代码)。

### sim 侧 `hand_mjlab`
- action 管线 [actions.py:140-160]:`a=clamp(raw)·ema+prev·(1-ema)`;
  `tgt=alpha·prev_tgt+(1-alpha)·q + a·dt·vel_scale·action_scale`;`tgt=clamp(tgt, soft_limits)`。
  driver 顺序(7 arm + 12 finger),`preserve_order=True`。
- 速率:物理 200Hz(`timestep=0.005`)/ decimation 10 / policy 20Hz / `step_dt=0.05`(hand_env_cfg)。
- 待标参数标称 [panda_tesollo_constants.py]:arm armature=0.1;finger armature=0.005、
  frictionloss=0.05、viscous_damping=0(已删 URDF 1.1 泄漏);全 body `gravcomp=1.0`(复刻真机重力补偿)。

---

## 1. 设计决策(已与用户收敛)

| 决策 | 选择 | 理由 |
|---|---|---|
| 激励空间 | **action 空间**,但**同时记录 q_target** | 端到端忠实策略闭环;记 q_target 使 sim 复放可在 target 空间干净拟合 |
| 节点架构 | **B:单臂 in-process 节点** | 只拉起待测一条臂(另一臂不上电),无 `/target_joint_states` 双发布者竞争,时间戳一跳到位 |
| τ / load 读出 | **新增独立只读出口**(如 `tau_state()` / `load_info()`) | 不破坏现有 14 维/臂 `joint_state()` 契约(`rl_driver_node`/`BimanualInterface` 都按它切分) |
| 重力/负载 | **第 0 步先诊断**(只读,先于激励) | 重力基线不对,armature/frictionloss 全标偏;并验证右臂"过补"假设 + 确认修正杠杆 |
| 旧脚本 | **不复用 `sysid_action_traj_gen.py`** | 自写更仔细的新 sysid 代码 |

### 架构 B 说明
新节点 `sysid_excitation_node` 内部直接 `BimanualInterface(enable_left=True/False, enable_right=…)`
只拉起**一条臂**(7 arm + 12 finger),自己跑激励 → `set_joint_positions(q_target, joint_names=[单关节])`
→ 直接 `joint_state()` / 新 `tau_state()` 读回 → 自录 csv/npz。它是该臂控制器在 sysid 期间的**唯一上游**,
与 `rl_driver` / `rl_policy` **互斥**(三者绝不并跑)。这就是"与 robot_motion 协调"的含义 = 独占 + 复用同一 PD 律。

---

## 2. ⚠️ 必须复刻到 sim 的真机非线性(否则拟合错系统)

1. **误差 clamp ±0.3**:真机 PD 前把 `e` clamp 到 ±`max_joint_delta` [joint_torque_controller.cpp:33-35];
   sim 的 position actuator 无此 clamp。→ 激励保证 `e<0.3`,或在 sim 复放里补同一 clamp。
2. **Coriolis 前馈**:真机控制器**加** `coriolis(q,dq)` 前馈 [joint_torque_controller.cpp:37,39];
   sim 的 MuJoCo plant 中 Coriolis 自然存在、未补偿。单关节慢速里很小,但属真实 deviation,验证阶段留意。
3. **重力补偿基线**:真机靠 FCI 按 Desk 配置 load 补重力(可能左右不一致 / 右臂过补);
   sim 用 `gravcomp=1.0` 假设"完美补偿"。第 0 步的实测结论决定:真机 `setLoad` 修对后 sim 维持
   `gravcomp=1`,还是 sim 复刻残余偏差。

---

## 3. 分步计划

### 第 0 步 —— 重力补偿 / 负载诊断(只读 + 修正验证,**独立先做**)
**确认:独立先上真机执行,先于激励节点。** 一个小诊断脚本/节点,主体不发运动指令,单臂逐个跑:
> **单臂执行**:每次只连一条臂只读 + dump,**左右对比离线做**(`--arm left/right --out`
> 各跑一次,再 `--compare left.json right.json`),不在一个进程里连双臂。
1. `robot_.readOnce()` 读配置负载:`m_load` / `F_x_Cload`(质心) / `I_load`(惯量) / `m_ee` / `m_total`,
   **左右臂对比**(就是 Desk/FCI web 里设的那套)。
2. `robot_.loadModel()` → `model.gravity(q)` 用 **config home q 显式重载**算重力力矩(同姿态公平对比),
   不传 q 则用当前物理姿态;= **FCI 实际施加的重力补偿力矩**,左右对比。
3. 静止读 `tau_ext_hat_filtered` 残余力矩,左右对比。
4. **产出**:右臂"过补"来源(Desk 配置不一致 / load 偏大 / 物理标定不对称)的结论。
5. **修正验证(代码优先,失败回退 Desk)**:
   - 先尝试**代码注入** `robot_.setLoad(m_load, F_x_Cload, I_load)` 写入修正后的负载;
   - 注入后**重读步骤 1–3**,确认 `m_load` / `model.gravity` / `tau_ext` 是否真的改变;
   - **若 setLoad 不生效**(读回无变化,FCI 可能只认 Desk 配置),则**由用户在 Desk 侧注入** load,再重读确认。
> 这一步定了,后面 plant 拟合才在正确的重力基线上做。

### 第 1 步 —— 新激励节点 `sysid_excitation_node`(架构 B,全新写)
- 自写激励(不复用旧脚本),**当前阶段两类轨迹**:
  - **简单轨迹**:逐关节(arm 7 → finger 12),一次只激励一个关节,其余钉 home;
    波形 step(多幅值,保证 `e<0.3`)+ chirp(0–10Hz,覆盖策略频段)。先单关节解耦。
  - **随机轨迹**:随机 action 序列(限幅/限速、保证 `e<0.3`),抓多关节耦合与交叉惯量。
  - 注:**端到端(随训练策略闭环/真实任务片段)暂不做**,等 end2end 成熟再加(见第 3 步)。
- **action 空间下发,但每步同时记录**:`raw_action` / 管线产出的 `q_target` / 实测 `q` / `q̇` /
  `τ`(`tau_J` + `tau_ext_hat_filtered`),按控制周期打时间戳 → csv/npz。
- 管线(EMA→速度积分→clamp)在节点内跑,**先核对它与 `rl_policy_node` 部署管线、与 sim [actions.py:140-160]
  逐字一致**。
- 复用 `BimanualInterface(enable_单臂)` + 同一 PD/控制器 + `max_joint_delta=0.3` + 软限位 + collision 阈值急停。
- 辨识采样可高于 policy 的 20Hz(如 200Hz 下发与记录)以抓 armature/高频;端到端验证段回到 20Hz 管线。

### 第 2 步 —— sim 复放拟合 `hand_mjlab/scripts/sysid_replay_fit.py`(待建)
- 单 env 拉起对应单臂 entity,喂记录的同一串 `q_target`,整轨匹配。
- 拟合 θ = **{armature, frictionloss, dof_damping, kp/kd 复核, 控制延迟}**;
  **kp/kd 是真机标称,不重新 LSE 选**,只复核微调。
- loss:`Σ_t Σ_j w_j[(q_sim−q_real)² + λ(q̇_sim−q̇_real)²]`(有 τ 加力矩项),逐关节归一化。
- 优化:CMA-ES / Optuna(或自由空间可微 rollout 梯度);**先逐关节、再全局微调**;
  **显式检查可辨识性**(armature↔kd、frictionloss↔damping 易简并)。
- 复刻第 2 节的两个非线性(误差 clamp、Coriolis),重力基线用第 0 步结论。
- 验证:留出集(没标过的幅值/频率 + 真实任务片段)。

### 第 3 步 —— 写回 + DR(端到端验证推迟)
- **当前**:用简单/随机轨迹的**留出集**验证拟合(没标过的幅值/频率/随机种子),不做端到端。
- 拟合标称写回 `panda_tesollo_constants.py`;Phase 4 DR(`pd_gains`/`joint_friction`/`geom_friction` 等)
  围绕标称随机化,覆盖残余失配。
- 接触段(指↔盖↔瓶)单独验证后再进 DR(自由空间拟合好之后)。
- **推迟**:端到端验证(同一条 action 轨迹过真机管线 vs sim 管线 / 真实任务片段 reach/grasp/twist),
  等 end2end 成熟再加。

---

## 4. 待辨识参数表

| 参数 | 位置 | 标称 | 来源 | 标定方式 |
|---|---|---|---|---|
| arm armature | actuator | 0.1 | menagerie 粗值 | 整轨高频拟合(P1) |
| finger armature | actuator | 0.005 | 粗猜 | 整轨高频/振荡拟合 |
| arm/finger frictionloss | actuator | 0 / 0.05 | 占位 | breakaway/stiction 力矩 + 拟合 |
| dof_damping | joint | 0 | kd 已提供阻尼 | 拟合(防与 kd 重复计) |
| kp / kd | actuator | 真机值 | 配置 | **仅复核**,不重选 |
| 控制延迟 | action term | ~1 步 | 经验 | 通信/控制滞后拟合 |
| 重力补偿 / load | FCI(Desk) | 见第 0 步 | Desk 配置 | readOnce + Model.gravity 诊断 |

---

## 5. 真机代码改动(本计划涉及,均待 review 后再动)
> **命名约定:本次在 `dexterity-interface-rl` 新增的所有代码(文件、节点、脚本、新读出口/方法)
> 一律加 `sysid_` 前缀**,与既有代码区分。
- **新增**:`sysid_excitation_node`(架构 B)+ 第 0 步诊断脚本(如 `sysid_gravity_diag` / `sysid_load_diag`)。
- **只读扩展**:`panda_interface`(.cpp/.hpp + pybind)新增独立 τ / load 读出口(如 `sysid_tau_state()` /
  `sysid_load_info()`),**不改现有 14 维 `joint_state()` 契约**,故 `rl_driver_node` /
  `BimanualInterface` 的切分逻辑不受影响。
- **可能**:`robot_.setLoad(...)` 修正右臂重力补偿(取决于第 0 步结论;已包成 `sysid_set_load`)。

### 实施进度
- ✅ **第 0 步**:C++ 只读出口 `sysid_load_info/gravity/tau_ext/tau_measured/set_load`
  (`panda_interface.hpp/.cpp`,新增 `model_` 成员 + `<franka/model.h>`)、pybind 5 个 `.def`、
  `panda/panda_interface.py` 薄包装、诊断脚本 `sysid/sysid_gravity_diag.py`
  (单臂读 + dump + 离线 `--compare` + set_load 验证)。
- ✅ **第 1 步**:`sysid/sysid_excitation.py`(架构 B 单臂 in-process;simple+random 激励;
  **逐字复刻真机 `compute_targets` 管线**;记录 `raw_action/ema_action/q_target/q/dq` → npz)。
  动态 τ 不记(`sysid_tau_*` 要求控制环停;静态 τ 是第 0 步)。
- ❌ **控制环重连不做**(已移除,`start_loop`/`stop_loop` 保持原样)。
- ⚠️ **本机无 franka 头文件/build,未编译验证** → 需 `pip install -e libs/robot_motion_interface`
  重编译 pybind 扩展(纯 Python 的 `sysid/` editable 即时可见)。`Franka::Franka` 已链接,无需改 CMake。

> **发现的 sim↔real 管线偏差**(记录,影响理解,不影响拟合——因为我们记录 q_target):
> 真机 `compute_targets`(rl_policy_node)= `tgt = prev_tgt + a·dt·vel·scale`,**无** alpha/joint_pos
> 重锚;每臂**单个** vel_action_scale。sim `actions.py` 有 `alpha·prev_tgt+(1-alpha)·joint_pos`、
> 且 arm/finger 分开 vel_scale。拟合走 `q_target→q`,与此无关;但作为 port 偏差另行核对。

## 6. sim 代码改动(待 review 后再动)
- **新增**:`hand_mjlab/scripts/sysid_replay_fit.py`(单 env 单臂复放 + 整轨拟合;plain MuJoCo,
  读 `logs/sysid/` 下从 dex copy 过来的 npz)。
- 拟合输出**写回** `panda_tesollo_constants.py`(armature/frictionloss/damping/delay/重力处理)。

## 6.5 数据流(已确认)
- **mjlab 不进 container**(py3.13+Warp+GPU,与 dex 容器不兼容)。
- **dex 容器**:只采集 → 写 `runtime/system_id/*.npz`、`*.json`(repo volume 挂到 `/workspace`,
  结果直接落在宿主机 dex repo)。
- **交接**:把 npz/json 从 dex repo **copy 到 `hand_mjlab/logs/sysid/`**。
- **mjlab venv**:`sysid_replay_fit.py` 读 `logs/sysid/`、跑 sim、拟合、写回常量。

---

## 7. 已确认决定(原开放问题)
- **第 0 步独立先上真机**(先于激励节点)。✓
- **端到端推迟**:当前只做简单 + 随机轨迹的 sysid,端到端等 end2end 成熟再说。✓
- **右臂重力修正**:代码 `setLoad` 优先注入并重读验证;若不生效,由用户在 Desk 侧注入。✓
- **命名**:`dexterity-interface-rl` 本次新增代码一律 `sysid_` 前缀(见第 5 节)。✓
  (`hand_mjlab` 侧沿用本仓库约定,如 `scripts/sysid_replay_fit.py`。)

---

## 8. 运行顺序(脚本怎么跑)

> dex 侧脚本是**纯 Python**(非 ROS 节点),在**驱动容器**里跑;repo volume 挂到 `/workspace`,
> 故路径前缀是 `/workspace/...`,产物落在宿主机 dex repo 同路径。先 `pip install -e
> libs/robot_motion_interface` 编好 pybind 扩展。**sysid 期间不要跑 `rl_driver`/`rl_policy`**
> (sysid 脚本独占该臂的 FCI 连接)。

**第 0 步 — 重力/负载诊断(dex 驱动容器,单臂逐个,只读)**
```bash
cd /workspace
python -m robot_motion_interface.sysid.sysid_gravity_diag --arm left  --gravity-at-home --out runtime/system_id/grav_left.json
python -m robot_motion_interface.sysid.sysid_gravity_diag --arm right --gravity-at-home --out runtime/system_id/grav_right.json
python -m robot_motion_interface.sysid.sysid_gravity_diag --compare runtime/system_id/grav_left.json runtime/system_id/grav_right.json
# 若要试代码修正右臂 load:
python -m robot_motion_interface.sysid.sysid_gravity_diag --arm right --set-load-mass <kg> --verify
```

**第 1 步 — 激励采集(dex 驱动容器,单臂,simple + random)**
```bash
cd /workspace
python -m robot_motion_interface.sysid.sysid_excitation --arm right --mode simple --out runtime/system_id/right_simple.npz
python -m robot_motion_interface.sysid.sysid_excitation --arm right --mode random --out runtime/system_id/right_random.npz
# 左臂同理 --arm left
```

**交接 — 把结果 copy 到 mjlab repo**
```bash
# 在宿主机:dex repo 的 runtime/system_id/ → hand_mjlab/logs/sysid/
cp <dex-repo>/libs/robot_motion_interface/runtime/system_id/*.npz  <...>/hand_mjlab/logs/sysid/
cp <dex-repo>/libs/robot_motion_interface/runtime/system_id/*.json <...>/hand_mjlab/logs/sysid/
```

**第 2 步 — sim 复放 + 拟合(hand_mjlab venv)**
```bash
cd <...>/hand_mjlab
uv run python scripts/sysid_replay_fit.py --selftest --arm right          # 先 sim-to-sim 自洽测试
uv run python scripts/sysid_replay_fit.py --arm right \
    --rec logs/sysid/right_simple.npz --holdout logs/sysid/right_random.npz --out logs/sysid/fit_right.json
# 拟合结果写回 panda_tesollo_constants.py(armature/frictionloss/damping/...)
```

**第 3 步 — 重训 + DR**(围绕标称随机化,见第 3 节)。

---

## 引用文件
- 真机:`libs/robot_motion/cpp/src/controllers/joint_torque_controller.cpp`、
  `libs/robot_motion_interface/cpp/src/panda_interface.cpp`、
  `libs/robot_motion_interface/src/robot_motion_interface/bimanual_interface.py`、
  `.../ros/.../rl_driver_node.py`、`.../config/rl_bimanual_driver_config.yaml`、
  `.../examples/oscillating_ex.py`、`.../utils/sysid_action_traj_gen.py`(参考)。
- sim:`src/hand_mjlab/assets/panda_tesollo/panda_tesollo_constants.py`、
  `src/hand_mjlab/tasks/hand/mdp/actions.py`、`src/hand_mjlab/tasks/hand/hand_env_cfg.py`。
