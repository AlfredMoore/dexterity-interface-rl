"""Generate per-joint action trajectories for system identification.

Output format is compatible with replay_action_torch.py:
  actions: [T, N, A] float32
  dones:   [T, N] bool

Where:
  T = number of steps
  N = number of trajectories (one trajectory per stimulated joint)
  A = action dimension (dual-chain joint count)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate action trajectories for real-world system identification.")
    parser.add_argument(
        "--runtime_cfg",
        type=str,
        default=None,
        help="Path to runtime_cfg.yaml. Default: <RMI_ROOT>/runtime/runtime_cfg.yaml",
    )
    parser.add_argument(
        "--handenv_cfg",
        type=str,
        default=None,
        help="Path to HandEnv.yaml. Default: <RMI_ROOT>/runtime/HandEnv.yaml",
    )
    parser.add_argument(
        "--output_pt",
        type=str,
        default=None,
        help="Output .pt path. Default: <RMI_ROOT>/runtime/actions_sysid_traj.pt",
    )
    parser.add_argument(
        "--waveform",
        type=str,
        choices=["sine", "step", "both"],
        default="sine",
        help="Stimulus type.",
    )
    parser.add_argument("--amplitude", type=float, default=1.0, help="Action amplitude in [0, 1].")
    parser.add_argument(
        "--startup_home_steps",
        type=int,
        default=10,
        help="Always prepend this many startup HOME-hold steps (zero action).",
    )
    parser.add_argument("--pre_hold_steps", type=int, default=30, help="Zero-action hold steps before stimulus.")
    parser.add_argument("--post_hold_steps", type=int, default=30, help="Zero-action hold steps after stimulus.")
    parser.add_argument("--sine_period_steps", type=int, default=120, help="Sine period in steps.")
    parser.add_argument("--sine_cycles", type=int, default=3, help="Number of sine cycles.")
    parser.add_argument("--step_hold_steps", type=int, default=90, help="Step plateau steps per level.")
    parser.add_argument("--step_gap_steps", type=int, default=30, help="Zero-action gap between +step and -step.")
    parser.add_argument("--both_gap_steps", type=int, default=60, help="Zero-action gap between sine and step in 'both'.")
    parser.add_argument(
        "--joint_indices",
        type=str,
        default="all",
        help='Stimulated action indices, e.g. "0,1,19" or "all".',
    )
    return parser.parse_args()


def _build_sine_signal(amplitude: float, period_steps: int, cycles: int) -> np.ndarray:
    n = period_steps * cycles
    t = np.arange(n, dtype=np.float32)
    return amplitude * np.sin(2.0 * np.pi * t / float(period_steps))


def _build_step_signal(amplitude: float, hold_steps: int, gap_steps: int) -> np.ndarray:
    pos = np.full((hold_steps,), amplitude, dtype=np.float32)
    gap = np.zeros((gap_steps,), dtype=np.float32)
    neg = np.full((hold_steps,), -amplitude, dtype=np.float32)
    return np.concatenate([pos, gap, neg], axis=0)


def _build_core_signal(args: argparse.Namespace) -> np.ndarray:
    if args.waveform == "sine":
        core = _build_sine_signal(args.amplitude, args.sine_period_steps, args.sine_cycles)
    elif args.waveform == "step":
        core = _build_step_signal(args.amplitude, args.step_hold_steps, args.step_gap_steps)
    else:
        sine = _build_sine_signal(args.amplitude, args.sine_period_steps, args.sine_cycles)
        step = _build_step_signal(args.amplitude, args.step_hold_steps, args.step_gap_steps)
        core = np.concatenate([sine, np.zeros((args.both_gap_steps,), dtype=np.float32), step], axis=0)
    return np.clip(core.astype(np.float32), -1.0, 1.0)


def _parse_joint_indices(spec: str, action_dim: int) -> list[int]:
    if spec.strip().lower() == "all":
        return list(range(action_dim))
    indices = [int(x.strip()) for x in spec.split(",") if x.strip()]
    for idx in indices:
        if idx < 0 or idx >= action_dim:
            raise ValueError(f"Invalid joint index {idx}, valid range: [0, {action_dim - 1}]")
    if len(set(indices)) != len(indices):
        raise ValueError("joint_indices contains duplicates.")
    return indices


def main() -> None:
    args = parse_args()

    rmi_root = Path(__file__).resolve().parents[3]
    runtime_dir = rmi_root / "runtime"
    runtime_cfg_path = Path(args.runtime_cfg) if args.runtime_cfg else runtime_dir / "runtime_cfg.yaml"
    handenv_cfg_path = Path(args.handenv_cfg) if args.handenv_cfg else runtime_dir / "HandEnv.yaml"
    output_pt_path = Path(args.output_pt) if args.output_pt else runtime_dir / "actions_sysid_traj.pt"

    with open(runtime_cfg_path, "r", encoding="utf-8") as f:
        runtime_cfg = yaml.safe_load(f)
    with open(handenv_cfg_path, "r", encoding="utf-8") as f:
        handenv_cfg = yaml.safe_load(f)

    dt = float(runtime_cfg["dt"])
    left_indices = runtime_cfg["policy_action_indices_dict"]["left"]
    right_indices = runtime_cfg["policy_action_indices_dict"]["right"]
    action_dim = len(left_indices) + len(right_indices)

    chain_joint_names = handenv_cfg["env"]["robot"]["jointNames"]["arm"] + handenv_cfg["env"]["robot"]["jointNames"]["hand"]
    if len(chain_joint_names) * 2 != action_dim:
        raise ValueError(
            f"Action dim mismatch: action_dim={action_dim}, "
            f"chain_joint_names*2={len(chain_joint_names) * 2}"
        )
    action_labels = [f"left_{name}" for name in chain_joint_names] + [f"right_{name}" for name in chain_joint_names]

    home_left_pose = handenv_cfg["experiment"]["home"]["left_joint_pose"]
    home_right_pose = handenv_cfg["experiment"]["home"]["right_joint_pose"]
    home_joint_pos = np.array(
        [home_left_pose[name] for name in chain_joint_names] + [home_right_pose[name] for name in chain_joint_names],
        dtype=np.float32,
    )

    core = _build_core_signal(args)
    signal = np.concatenate(
        [
            np.zeros((args.startup_home_steps,), dtype=np.float32),
            np.zeros((args.pre_hold_steps,), dtype=np.float32),
            core,
            np.zeros((args.post_hold_steps,), dtype=np.float32),
        ],
        axis=0,
    )
    total_steps = int(signal.shape[0])

    stim_joint_indices = _parse_joint_indices(args.joint_indices, action_dim)
    num_traj = len(stim_joint_indices)
    actions = np.zeros((total_steps, num_traj, action_dim), dtype=np.float32)
    for i, joint_idx in enumerate(stim_joint_indices):
        actions[:, i, joint_idx] = signal

    dones = np.zeros((total_steps, num_traj), dtype=bool)
    dones[-1, :] = True

    payload = {
        "actions": torch.from_numpy(actions),                      # [T, N, A]
        "dones": torch.from_numpy(dones),                          # [T, N]
        "dt": float(dt),
        "total_steps": total_steps,
        "num_traj": num_traj,
        "joint_names": action_labels,
        "stim_joint_indices": stim_joint_indices,
        "stim_joint_names": [action_labels[i] for i in stim_joint_indices],
        "waveform": args.waveform,
        "startup_home_steps": int(args.startup_home_steps),
        "signal": torch.from_numpy(signal),                        # [T]
        "home_joint_pos": torch.from_numpy(home_joint_pos),        # [A]
    }

    output_pt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_pt_path)

    print(f"[saved] {output_pt_path}")
    print(
        f"[info] waveform={args.waveform}, amplitude={args.amplitude}, "
        f"steps={total_steps}, traj_count={num_traj}, action_dim={action_dim}, dt={dt}"
    )
    print(f"[info] stimulated joints: {payload['stim_joint_names'][:6]} ... total={num_traj}")


if __name__ == "__main__":
    main()
