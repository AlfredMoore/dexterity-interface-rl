"""SYSID Step 1 -- single-arm excitation + recording (architecture B, in-process).

Owns ONE arm's controller directly (BimanualInterface with only one side enabled)
and drives excitation through the SAME deployment action pipeline the policy uses
(``compute_targets`` -- replicated verbatim from rl_policy_node, see NOTE), then
records the closed-loop response for offline sysid fitting (Step 2).

Design (per SYSID_REDO.md):
  - Single arm only: the other arm is NOT powered (enable_<other>=False).
  - Excite in ACTION space (raw in [-1,1]); the pipeline integrates it into a joint
    target. Two trajectory families: "simple" (per-joint step + chirp, decoupled)
    and "random" (smooth full-chain noise, captures cross-joint coupling).
  - Record raw_action, ema_action, q_target (pipeline output), q, dq per step.
    (Dynamic tau during motion is NOT captured here -- the sysid_tau_* readouts
    require the loop STOPPED; static gravity tau is Step 0. A dynamic-tau capture
    would need a small control-loop extension -- separate decision.)

Safety: respects the controller's max_joint_delta clamp and the soft joint limits
(targets are clipped to runtime_cfg soft limits). Keep amplitudes modest so the PD
error stays within max_joint_delta. SOLE owner of the arm -- never run alongside
rl_driver / rl_policy.

Run (inside the DRIVER container, repo volume-mounted at /workspace):
    python -m robot_motion_interface.sysid.sysid_excitation --arm right --mode simple \
        --out runtime/system_id/right_simple.npz
    python -m robot_motion_interface.sysid.sysid_excitation --arm right --mode random \
        --out runtime/system_id/right_random.npz
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import yaml

from robot_motion_interface.bimanual_interface import BimanualInterface

# RMI root = libs/robot_motion_interface (has config/ and runtime/).
_RMI_ROOT = Path(__file__).resolve().parents[3]

_N_PANDA = 7
_N_TESOLLO = 12
_N_CHAIN = _N_PANDA + _N_TESOLLO  # 19 per arm


# ---------------------------------------------------------------------------
# Deployment action pipeline.
# NOTE: copied VERBATIM from rl_policy_node.compute_targets (per-arm slice) so the
# identified system matches the policy's closed loop. Keep in sync with that file.
#   a   = clip(raw,-1,1)*ema + prev*(1-ema)
#   tgt = clip(prev_tgt + a*dt*vel_action_scale*action_scale, soft_lower, soft_upper)
# (No alpha/joint_pos re-anchoring; single per-arm vel scale -- both differ from the
#  sim actions.py pipeline. Recording q_target makes the Step-2 fit pipeline-agnostic.)
# ---------------------------------------------------------------------------
def _step_pipeline(
    raw_action: np.ndarray,
    prev_ema: np.ndarray,
    prev_targets: np.ndarray,
    dt: float,
    ema: float,
    vel_action_scale: float,
    action_scale: float,
    soft_lower: np.ndarray,
    soft_upper: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (new_targets, new_ema_action) for one chain (19,)."""
    a = np.clip(raw_action, -1.0, 1.0) * ema + prev_ema * (1.0 - ema)
    targets = np.clip(
        prev_targets + a * dt * vel_action_scale * action_scale,
        soft_lower,
        soft_upper,
    )
    return targets, a


# ---------------------------------------------------------------------------
# Excitation waveforms (ACTION space, in [-1, 1]).
# A constant action is a velocity command -> the target RAMPS; so bounded motions
# use sign-alternating pulses or sinusoids (integral stays bounded).
# ---------------------------------------------------------------------------
def _pulse(amp: float, hold: int, gap: int) -> np.ndarray:
    """+amp, gap, -amp -> a there-and-back motion (target returns near start)."""
    return np.concatenate([
        np.full(hold, amp, dtype=np.float32),
        np.zeros(gap, dtype=np.float32),
        np.full(hold, -amp, dtype=np.float32),
        np.zeros(gap, dtype=np.float32),
    ])


def _chirp(amp: float, n: int, f0: float, f1: float, dt: float) -> np.ndarray:
    """Linear sine sweep f0->f1 Hz over n steps (bounded oscillating target)."""
    t = np.arange(n, dtype=np.float32) * dt
    T = max(n * dt, 1e-6)
    k = (f1 - f0) / T
    phase = 2.0 * np.pi * (f0 * t + 0.5 * k * t * t)
    return (amp * np.sin(phase)).astype(np.float32)


def _build_simple(args, dt: float) -> tuple[np.ndarray, list[str]]:
    """Per-joint sequence: for each of the 19 joints, pulse then chirp; others 0.

    Returns actions (T, 19) and a per-step label list for bookkeeping.
    """
    pulse = _pulse(args.amp, args.pulse_hold, args.gap)
    chirp = _chirp(args.amp, args.chirp_steps, args.chirp_f0, args.chirp_f1, dt)
    pre = np.zeros(args.pre_hold, dtype=np.float32)
    core = np.concatenate([pre, pulse, np.zeros(args.gap, dtype=np.float32), chirp, pre])
    seg = core.shape[0]

    actions = np.zeros((seg * _N_CHAIN, _N_CHAIN), dtype=np.float32)
    labels: list[str] = []
    for j in range(_N_CHAIN):
        # arm joints (0..6) get a smaller amplitude than fingers if requested.
        scale = args.arm_amp_scale if j < _N_PANDA else 1.0
        actions[j * seg:(j + 1) * seg, j] = core * scale
        labels.extend([f"joint{j}"] * seg)
    return actions, labels


def _build_random(args, dt: float) -> tuple[np.ndarray, list[str]]:
    """Smooth full-chain band-limited noise (captures cross-joint coupling).

    Deterministic per --seed; arm joints scaled down. The action is a low-pass
    filtered zero-mean noise so the integrated target stays bounded.
    """
    rng = np.random.default_rng(args.seed)
    n = args.random_steps
    raw = rng.uniform(-args.amp, args.amp, size=(n, _N_CHAIN)).astype(np.float32)
    # simple EMA low-pass to band-limit the velocity command
    alpha = 0.85
    smooth = np.zeros_like(raw)
    for i in range(1, n):
        smooth[i] = alpha * smooth[i - 1] + (1.0 - alpha) * raw[i]
    smooth[:, :_N_PANDA] *= args.arm_amp_scale
    pre = np.zeros((args.pre_hold, _N_CHAIN), dtype=np.float32)
    actions = np.concatenate([pre, smooth, pre], axis=0)
    return actions, ["random"] * actions.shape[0]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SYSID single-arm excitation + recording.")
    p.add_argument("--arm", choices=["left", "right"], required=True)
    p.add_argument("--mode", choices=["simple", "random"], required=True)
    p.add_argument("--driver_cfg", type=str, default=None,
                   help="rl_bimanual_driver_config.yaml. Default: package config/.")
    p.add_argument("--runtime_cfg", type=str, default=None,
                   help="runtime_cfg.yaml (dt, action_EMA, action-scale dict, soft "
                        "limits). Default: <RMI_ROOT>/runtime/runtime_cfg.yaml.")
    p.add_argument("--action-scale", type=float, default=1.0,
                   help="Deployment env_cfg['action_scale'] (faithful default 1.0).")
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
    p.add_argument("--record-rate", type=float, default=None,
                   help="Hz to poll joint_state between commands (default = command "
                        "rate 1/dt). Higher captures faster dynamics for armature ID.")
    return p.parse_args()


def _load_cfgs(args):
    drv_path = Path(args.driver_cfg) if args.driver_cfg else \
        _RMI_ROOT / "config" / "rl_bimanual_driver_config.yaml"
    rt_path = Path(args.runtime_cfg) if args.runtime_cfg else \
        _RMI_ROOT / "runtime" / "runtime_cfg.yaml"
    for pth in (drv_path, rt_path):
        if not pth.exists():
            raise FileNotFoundError(f"config not found: {pth}")
    with open(drv_path, "r", encoding="utf-8") as f:
        drv = yaml.safe_load(f)
    with open(rt_path, "r", encoding="utf-8") as f:
        rt = yaml.safe_load(f)
    return drv, rt


def _build_single_arm(side: str, drv: dict) -> BimanualInterface:
    """Single-arm BimanualInterface (only `side` enabled; other not powered)."""
    urdf = str((_RMI_ROOT / drv["panda_urdf_path"]).resolve())
    common = dict(
        panda_urdf_path=urdf,
        panda_home_joint_positions=np.array(drv["panda_home_joint_positions"], float),
        panda_kp=np.array(drv["panda_kp"], float),
        panda_kd=np.array(drv["panda_kd"], float),
        tesollo_home_joint_positions=np.array(drv["tesollo_home_joint_positions"], float),
        tesollo_control_loop_frequency=drv["tesollo_control_loop_frequency"],
        tesollo_kp=np.array(drv["tesollo_kp"], float),
        tesollo_kd=np.array(drv["tesollo_kd"], float),
    )
    # Pass empty joint-name lists for the disabled side so the 19-dim command layout
    # maps only onto the enabled chain (see BimanualInterface.set_joint_positions).
    side_kwargs = dict(
        left_panda_hostname=None, left_panda_joint_names=[],
        right_panda_hostname=None, right_panda_joint_names=[],
        left_tesollo_ip=None, left_tesollo_port=None, left_tesollo_joint_names=[],
        right_tesollo_ip=None, right_tesollo_port=None, right_tesollo_joint_names=[],
    )
    side_kwargs[f"{side}_panda_hostname"] = drv[f"{side}_panda_hostname"]
    side_kwargs[f"{side}_panda_joint_names"] = drv[f"{side}_panda_joint_names"]
    side_kwargs[f"{side}_tesollo_ip"] = drv[f"{side}_tesollo_ip"]
    side_kwargs[f"{side}_tesollo_port"] = drv[f"{side}_tesollo_port"]
    side_kwargs[f"{side}_tesollo_joint_names"] = drv[f"{side}_tesollo_joint_names"]
    return BimanualInterface(
        enable_left=(side == "left"), enable_right=(side == "right"),
        **common, **side_kwargs,
    )


def main() -> None:
    args = parse_args()
    side = args.arm
    drv, rt = _load_cfgs(args)

    dt = float(rt["dt"])
    ema = float(rt["action_EMA"])
    vel_action_scale = float(rt["robot_action_scale_dict"][f"{side}_joint_vel_action"])
    soft_lower = np.array(rt["robot_joint_limits_dict"][f"{side}_joint_pose_soft_lower"], float)
    soft_upper = np.array(rt["robot_joint_limits_dict"][f"{side}_joint_pose_soft_upper"], float)

    home = np.concatenate([
        np.array(drv["panda_home_joint_positions"], float),
        np.array(drv["tesollo_home_joint_positions"], float),
    ])  # (19,)

    if args.mode == "simple":
        actions, labels = _build_simple(args, dt)
    else:
        actions, labels = _build_random(args, dt)

    record_dt = 1.0 / args.record_rate if args.record_rate else dt
    polls_per_step = max(1, int(round(dt / record_dt)))

    arms = _build_single_arm(side, drv)
    rec_t, rec_raw, rec_ema, rec_tgt, rec_q, rec_dq = [], [], [], [], [], []

    try:
        arms.start_loop()
        arms.home(blocking=True)
        time.sleep(1.0)

        prev_ema = np.zeros(_N_CHAIN, dtype=np.float32)
        targets = home.copy()
        t0 = time.perf_counter()

        for k in range(actions.shape[0]):
            raw = actions[k]
            targets, prev_ema = _step_pipeline(
                raw, prev_ema, targets, dt, ema, vel_action_scale,
                args.action_scale, soft_lower, soft_upper,
            )
            arms.set_joint_positions(targets)  # command the 19-dim chain

            # Poll the measured state up to the next command instant.
            for _ in range(polls_per_step):
                st = np.asarray(arms.joint_state(), dtype=float)  # [q(19), dq(19)]
                rec_t.append(time.perf_counter() - t0)
                rec_raw.append(raw.copy())
                rec_ema.append(prev_ema.copy())
                rec_tgt.append(targets.copy())
                rec_q.append(st[:_N_CHAIN].copy())
                rec_dq.append(st[_N_CHAIN:].copy())
                time.sleep(record_dt)
    finally:
        arms.stop_loop()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        arm=side, mode=args.mode, dt=dt, action_EMA=ema,
        vel_action_scale=vel_action_scale, action_scale=args.action_scale,
        soft_lower=soft_lower, soft_upper=soft_upper, home=home,
        joint_names=np.array(drv[f"{side}_panda_joint_names"] + drv[f"{side}_tesollo_joint_names"]),
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
