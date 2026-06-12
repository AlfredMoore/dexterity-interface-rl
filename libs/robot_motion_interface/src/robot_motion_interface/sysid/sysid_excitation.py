"""SYSID Step 1 -- single-arm excitation + recording (architecture A, via ROS topics).

Publishes excitation targets to ``/target_joint_states`` and subscribes to
``/joint_states`` for measured feedback. **rl_driver must be running** on the driver
computer (it owns the FCI/Tesollo connections and the real-time control loops).

Design (per SYSID_REDO.md):
  - Excite ONE arm at a time; the other arm is held at home in every published message.
  - Excite in ACTION space (raw in [-1,1]); the pipeline integrates it into a joint
    target. Two trajectory families: "simple" (per-joint step + chirp, decoupled)
    and "random" (smooth full-chain noise, captures cross-joint coupling).
  - Record raw_action, ema_action, q_target (pipeline output), q, dq per step.

Safety: targets are clipped to URDF soft limits; amplitudes are modest. The arm's PD
controller (inside rl_driver) enforces max_joint_delta=0.3 on top. Do NOT run
rl_policy at the same time (conflicting targets on /target_joint_states).

Run (inside the POLICY container, ROS workspace sourced, rl_driver running on driver):
    python -m robot_motion_interface.sysid.sysid_excitation --arm right --mode simple \
        --out runtime/system_id/right_simple.npz
    python -m robot_motion_interface.sysid.sysid_excitation --arm right --mode random \
        --out runtime/system_id/right_random.npz
"""

from __future__ import annotations

import argparse
import threading
import time
from pathlib import Path

import numpy as np
import yaml

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from robot_motion_interface.utils.qos import HIGH_PERF_QOS, HIGH_RELIA_QOS

# RMI root = libs/robot_motion_interface (has config/).
_RMI_ROOT = Path(__file__).resolve().parents[3]

_N_PANDA = 7
_N_TESOLLO = 12
_N_CHAIN = _N_PANDA + _N_TESOLLO  # 19 per arm
_N_FULL = _N_CHAIN * 2             # 38 total (left + right)

# ---------------------------------------------------------------------------
# Pipeline constants (from hand_mjlab, no runtime_cfg dependency).
# ---------------------------------------------------------------------------
_DT = 0.05          # timestep(0.005) * decimation(10) = 20 Hz policy
_EMA = 0.5          # actions.py SmoothedJointPositionActionCfg.ema
_ACTION_SCALE = 1.0  # actions.py SmoothedJointPositionActionCfg.action_scale

# Per-joint velocity scale: arm=1.0, finger=1.5 (actions.py arm_vel_scale/finger_vel_scale).
# Order: panda_joint1..7, F1M1..F1M4, F2M1..F2M4, F3M1..F3M4 (= driver config order).
_VEL_SCALE = np.array(
    [1.0] * _N_PANDA + [1.5] * _N_TESOLLO, dtype=np.float32
)

# Soft joint limits from URDF (panda safety_controller + tesollo <limit>).
# soft_joint_pos_limit_factor = 1.0 -> soft = hard.
# Order: panda_joint1..7, F1M1..F1M4, F2M1..F2M4, F3M1..F3M4.
_SOFT_LOWER = np.array([
    -2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973,  # panda
    -1.04720, -1.76278, -0.15708, -0.22689,  # F1
    -1.91986, -1.76278, -0.15708, -0.22689,  # F2
    -0.08727, -1.76278, -0.15708, -0.22689,  # F3
], dtype=np.float32)

_SOFT_UPPER = np.array([
    +2.8973, +1.7628, +2.8973, -0.0698, +2.8973, +3.7525, +2.8973,  # panda
    +1.04720, +1.76278, +2.53073, +2.02458,  # F1
    +0.13963, +1.76278, +2.53073, +2.02458,  # F2
    +2.00713, +1.76278, +2.53073, +2.02458,  # F3
], dtype=np.float32)

# Joint names in driver order (for npz metadata).
_JOINT_NAMES = [
    "panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4",
    "panda_joint5", "panda_joint6", "panda_joint7",
    "F1M1", "F1M2", "F1M3", "F1M4",
    "F2M1", "F2M2", "F2M3", "F2M4",
    "F3M1", "F3M2", "F3M3", "F3M4",
]


# ---------------------------------------------------------------------------
# Action pipeline (same as before).
# ---------------------------------------------------------------------------
def _step_pipeline(
    raw_action: np.ndarray,
    prev_ema: np.ndarray,
    prev_targets: np.ndarray,
    dt: float,
    ema: float,
    vel_scale: np.ndarray,
    action_scale: float,
    soft_lower: np.ndarray,
    soft_upper: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (new_targets, new_ema_action) for one chain (19,)."""
    a = np.clip(raw_action, -1.0, 1.0) * ema + prev_ema * (1.0 - ema)
    targets = np.clip(
        prev_targets + a * dt * vel_scale * action_scale,
        soft_lower,
        soft_upper,
    )
    return targets, a


# ---------------------------------------------------------------------------
# Excitation waveforms (unchanged).
# ---------------------------------------------------------------------------
def _pulse(amp: float, hold: int, gap: int) -> np.ndarray:
    return np.concatenate([
        np.full(hold, amp, dtype=np.float32),
        np.zeros(gap, dtype=np.float32),
        np.full(hold, -amp, dtype=np.float32),
        np.zeros(gap, dtype=np.float32),
    ])


def _chirp(amp: float, n: int, f0: float, f1: float, dt: float) -> np.ndarray:
    t = np.arange(n, dtype=np.float32) * dt
    T = max(n * dt, 1e-6)
    k = (f1 - f0) / T
    phase = 2.0 * np.pi * (f0 * t + 0.5 * k * t * t)
    return (amp * np.sin(phase)).astype(np.float32)


def _build_simple(args, dt: float) -> tuple[np.ndarray, list[str]]:
    pulse = _pulse(args.amp, args.pulse_hold, args.gap)
    chirp = _chirp(args.amp, args.chirp_steps, args.chirp_f0, args.chirp_f1, dt)
    pre = np.zeros(args.pre_hold, dtype=np.float32)
    core = np.concatenate([pre, pulse, np.zeros(args.gap, dtype=np.float32), chirp, pre])
    seg = core.shape[0]
    actions = np.zeros((seg * _N_CHAIN, _N_CHAIN), dtype=np.float32)
    labels: list[str] = []
    for j in range(_N_CHAIN):
        scale = args.arm_amp_scale if j < _N_PANDA else 1.0
        actions[j * seg:(j + 1) * seg, j] = core * scale
        labels.extend([f"joint{j}"] * seg)
    return actions, labels


def _build_random(args, dt: float) -> tuple[np.ndarray, list[str]]:
    rng = np.random.default_rng(args.seed)
    n = args.random_steps
    raw = rng.uniform(-args.amp, args.amp, size=(n, _N_CHAIN)).astype(np.float32)
    alpha = 0.85
    smooth = np.zeros_like(raw)
    for i in range(1, n):
        smooth[i] = alpha * smooth[i - 1] + (1.0 - alpha) * raw[i]
    smooth[:, :_N_PANDA] *= args.arm_amp_scale
    pre = np.zeros((args.pre_hold, _N_CHAIN), dtype=np.float32)
    actions = np.concatenate([pre, smooth, pre], axis=0)
    return actions, ["random"] * actions.shape[0]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SYSID single-arm excitation + recording (via ROS).")
    p.add_argument("--arm", choices=["left", "right"], required=True)
    p.add_argument("--mode", choices=["simple", "random"], required=True)
    p.add_argument("--driver_cfg", type=str, default=None,
                   help="rl_bimanual_driver_config.yaml. Default: package config/.")
    p.add_argument("--out", type=str, required=True, help="Output .npz path.")
    # excitation shape
    p.add_argument("--amp", type=float, default=0.3, help="Action amplitude in [0,1].")
    p.add_argument("--arm-amp-scale", type=float, default=0.5,
                   help="Extra scale on arm (panda) action vs fingers.")
    p.add_argument("--pre-hold", type=int, default=20, help="Zero-action settle steps.")
    p.add_argument("--pulse-hold", type=int, default=15)
    p.add_argument("--gap", type=int, default=20)
    p.add_argument("--chirp-steps", type=int, default=200)
    p.add_argument("--chirp-f0", type=float, default=0.2)
    p.add_argument("--chirp-f1", type=float, default=8.0)
    p.add_argument("--random-steps", type=int, default=600)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def _load_driver_cfg(args) -> dict:
    drv_path = Path(args.driver_cfg) if args.driver_cfg else \
        _RMI_ROOT / "config" / "rl_bimanual_driver_config.yaml"
    if not drv_path.exists():
        raise FileNotFoundError(f"driver config not found: {drv_path}")
    with open(drv_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# 38-dim target construction: active arm = excitation, inactive arm = home.
# Joint order: [left_panda(7), left_tesollo(12), right_panda(7), right_tesollo(12)]
# ---------------------------------------------------------------------------
def _make_full_target(side: str, arm_targets: np.ndarray, home_38: np.ndarray) -> np.ndarray:
    """Build a 38-dim target: active arm gets `arm_targets`, inactive arm stays home."""
    full = home_38.copy()
    if side == "left":
        full[:_N_CHAIN] = arm_targets
    else:
        full[_N_CHAIN:] = arm_targets
    return full


def _extract_arm_state(side: str, full_pos: np.ndarray, full_vel: np.ndarray):
    """Extract the active arm's (q, dq) from 38-dim joint_states."""
    if side == "left":
        return full_pos[:_N_CHAIN].copy(), full_vel[:_N_CHAIN].copy()
    else:
        return full_pos[_N_CHAIN:].copy(), full_vel[_N_CHAIN:].copy()


def main() -> None:
    args = parse_args()
    side = args.arm
    drv = _load_driver_cfg(args)
    dt = _DT

    # Home position (38-dim): [left_panda_home, left_tesollo_home, right_panda_home, right_tesollo_home]
    panda_home = np.array(drv["panda_home_joint_positions"], dtype=np.float32)
    tesollo_home = np.array(drv["tesollo_home_joint_positions"], dtype=np.float32)
    chain_home = np.concatenate([panda_home, tesollo_home])  # (19,)
    home_38 = np.concatenate([chain_home, chain_home])       # (38,) same home for both arms

    # Build excitation
    if args.mode == "simple":
        actions, labels = _build_simple(args, dt)
    else:
        actions, labels = _build_random(args, dt)

    # --- ROS setup ---
    rclpy.init()
    node = Node("sysid_excitation")
    target_pub = node.create_publisher(JointState, "/target_joint_states", HIGH_RELIA_QOS)

    # Latest joint_states (updated by subscriber callback in a spin thread).
    latest_pos = np.zeros(_N_FULL, dtype=np.float32)
    latest_vel = np.zeros(_N_FULL, dtype=np.float32)
    state_received = threading.Event()
    lock = threading.Lock()

    def _js_cb(msg: JointState):
        nonlocal latest_pos, latest_vel
        with lock:
            latest_pos[:] = np.array(msg.position, dtype=np.float32)
            latest_vel[:] = np.array(msg.velocity, dtype=np.float32)
        state_received.set()

    node.create_subscription(JointState, "/joint_states", _js_cb, HIGH_PERF_QOS)

    # Spin in background so the subscriber fires.
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    # Wait for first joint_states from rl_driver.
    print("[sysid] Waiting for /joint_states from rl_driver ...")
    if not state_received.wait(timeout=10.0):
        raise RuntimeError("No /joint_states received — is rl_driver running?")
    print("[sysid] Got joint_states. Holding home for 2s ...")

    # Hold home for 2 seconds (settle).
    for _ in range(int(2.0 / dt)):
        msg = JointState()
        msg.header.stamp = node.get_clock().now().to_msg()
        msg.position = home_38.tolist()
        target_pub.publish(msg)
        time.sleep(dt)

    # --- Excitation loop ---
    prev_ema = np.zeros(_N_CHAIN, dtype=np.float32)
    targets = chain_home.copy()  # active arm starts at home
    rec_t, rec_raw, rec_ema, rec_tgt, rec_q, rec_dq = [], [], [], [], [], []
    t0 = time.perf_counter()

    print(f"[sysid] Starting excitation: arm={side} mode={args.mode} steps={actions.shape[0]}")
    try:
        for k in range(actions.shape[0]):
            raw = actions[k]
            targets, prev_ema = _step_pipeline(
                raw, prev_ema, targets, dt, _EMA, _VEL_SCALE,
                _ACTION_SCALE, _SOFT_LOWER, _SOFT_UPPER,
            )

            # Publish full 38-dim target (active arm = excitation, inactive = home).
            full_target = _make_full_target(side, targets, home_38)
            msg = JointState()
            msg.header.stamp = node.get_clock().now().to_msg()
            msg.position = full_target.tolist()
            target_pub.publish(msg)

            # Record measured state.
            with lock:
                q_arm, dq_arm = _extract_arm_state(side, latest_pos, latest_vel)

            rec_t.append(time.perf_counter() - t0)
            rec_raw.append(raw.copy())
            rec_ema.append(prev_ema.copy())
            rec_tgt.append(targets.copy())
            rec_q.append(q_arm)
            rec_dq.append(dq_arm)

            time.sleep(dt)
    finally:
        # Return to home.
        print("[sysid] Returning to home ...")
        for _ in range(int(2.0 / dt)):
            msg = JointState()
            msg.header.stamp = node.get_clock().now().to_msg()
            msg.position = home_38.tolist()
            target_pub.publish(msg)
            time.sleep(dt)
        node.destroy_node()
        rclpy.try_shutdown()

    # --- Save ---
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        arm=side, mode=args.mode, dt=dt, action_EMA=_EMA,
        vel_scale=_VEL_SCALE, action_scale=_ACTION_SCALE,
        soft_lower=_SOFT_LOWER, soft_upper=_SOFT_UPPER, home=chain_home,
        joint_names=np.array(_JOINT_NAMES),
        labels=np.array(labels),
        t=np.array(rec_t),
        raw_action=np.array(rec_raw),
        ema_action=np.array(rec_ema),
        q_target=np.array(rec_tgt),
        q=np.array(rec_q),
        dq=np.array(rec_dq),
    )
    print(f"[saved] {out_path}  steps={len(rec_t)}  arm={side}  mode={args.mode}")


if __name__ == "__main__":
    main()
