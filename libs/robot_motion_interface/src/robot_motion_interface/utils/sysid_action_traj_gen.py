"""Generate per-joint action trajectories for system identification.

Output format is a large dict saved to one .pt file:
  {
    "__meta__": { "available_combo_keys": [...] , ... },
    "wfsine_amp0p2": { ...single-trajectory-set payload... },
    "wfboth_amp1":   { ... },
    ...
  }

Each combo payload includes:
  actions: [T, N, A] float32
  dones:   [T, N] bool
where:
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

try:
    from robot_motion_interface.utils.sim2real import joint_mapping
except ModuleNotFoundError:
    try:
        from sim2real import joint_mapping
    except ModuleNotFoundError:
        from .sim2real import joint_mapping


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
        "--driver_cfg",
        type=str,
        default=None,
        help="Path to rl_bimanual_driver_config.yaml. Default: <RMI_ROOT>/config/rl_bimanual_driver_config.yaml",
    )
    parser.add_argument(
        "--output_pt",
        type=str,
        default=None,
        help="Output .pt path. Default: <RMI_ROOT>/runtime/system_id/actions_sysid_traj_grid.pt",
    )
    # Deprecated single-combo args kept for compatibility.
    parser.add_argument(
        "--waveform",
        type=str,
        choices=["sine", "step", "both"],
        default=None,
        help="(Deprecated) Single stimulus type. If set, overrides --waveforms.",
    )
    parser.add_argument(
        "--amplitude",
        type=float,
        default=None,
        help="(Deprecated) Single action amplitude in [0,1]. If set, overrides --amplitudes.",
    )
    parser.add_argument(
        "--waveforms",
        type=str,
        default="sine,step,both",
        help='Comma-separated waveform list, e.g. "sine,step,both".',
    )
    parser.add_argument(
        "--amplitudes",
        type=str,
        default="0.2,0.4,0.6,0.8,1.0",
        help='Comma-separated amplitude list, e.g. "0.2,0.4,1.0".',
    )
    parser.add_argument(
        "--startup_home_steps",
        type=int,
        default=10,
        help="Always prepend this many startup HOME-hold steps (zero action).",
    )
    parser.add_argument("--pre_hold_steps", type=int, default=30, help="Zero-action hold steps before stimulus.")
    parser.add_argument("--post_hold_steps", type=int, default=30, help="Zero-action hold steps after stimulus.")
    parser.add_argument("--sine_period_steps", type=int, default=120, help="Sine period in steps.")
    parser.add_argument("--sine_cycles", type=int, default=2, help="Number of sine cycles.")
    parser.add_argument("--step_hold_steps", type=int, default=90, help="Step plateau steps per level.")
    parser.add_argument("--step_gap_steps", type=int, default=30, help="Zero-action gap between +step and -step.")
    parser.add_argument("--both_gap_steps", type=int, default=60, help="Zero-action gap between sine and step in 'both'.")
    parser.add_argument(
        "--joint_indices",
        type=str,
        default="all",
        help='Stimulated action indices in real/driver order, e.g. "0,1,19" or "all".',
    )
    return parser.parse_args()


def _format_amp_token(value: float) -> str:
    token = f"{value:.6g}"
    return token.replace("-", "m").replace(".", "p")


def _with_waveform_amp_suffix(path: Path, waveform: str, amplitude: float) -> Path:
    if path.suffix == "":
        path = path.with_suffix(".pt")
    suffix = f"_wf{waveform}_amp{_format_amp_token(amplitude)}"
    if suffix in path.stem:
        return path
    return path.with_name(f"{path.stem}{suffix}{path.suffix}")


def _build_sine_signal(amplitude: float, period_steps: int, cycles: int) -> np.ndarray:
    n = period_steps * cycles
    t = np.arange(n, dtype=np.float32)
    return amplitude * np.sin(2.0 * np.pi * t / float(period_steps))


def _build_step_signal(amplitude: float, hold_steps: int, gap_steps: int) -> np.ndarray:
    pos = np.full((hold_steps,), amplitude, dtype=np.float32)
    gap = np.zeros((gap_steps,), dtype=np.float32)
    neg = np.full((hold_steps,), -amplitude, dtype=np.float32)
    return np.concatenate([pos, gap, neg], axis=0)


def _build_core_signal(args: argparse.Namespace, waveform: str, amplitude: float) -> np.ndarray:
    if waveform == "sine":
        core = _build_sine_signal(amplitude, args.sine_period_steps, args.sine_cycles)
    elif waveform == "step":
        core = _build_step_signal(amplitude, args.step_hold_steps, args.step_gap_steps)
    else:
        sine = _build_sine_signal(amplitude, args.sine_period_steps, args.sine_cycles)
        step = _build_step_signal(amplitude, args.step_hold_steps, args.step_gap_steps)
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


def _parse_waveforms(args: argparse.Namespace) -> list[str]:
    allowed = {"sine", "step", "both"}
    if args.waveform is not None:
        return [args.waveform]
    waveforms = [w.strip() for w in args.waveforms.split(",") if w.strip()]
    if not waveforms:
        raise ValueError("No waveforms provided.")
    invalid = [w for w in waveforms if w not in allowed]
    if invalid:
        raise ValueError(f"Invalid waveforms: {invalid}. Allowed: {sorted(allowed)}")
    return waveforms


def _parse_amplitudes(args: argparse.Namespace) -> list[float]:
    if args.amplitude is not None:
        amps = [float(args.amplitude)]
    else:
        amps = [float(x.strip()) for x in args.amplitudes.split(",") if x.strip()]
    if not amps:
        raise ValueError("No amplitudes provided.")
    for amp in amps:
        if amp < 0.0 or amp > 1.0:
            raise ValueError(f"Amplitude out of range [0,1]: {amp}")
    return amps


def _driver_joint_names(driver_cfg: dict) -> list[str]:
    if "all_joint_names" in driver_cfg:
        return list(driver_cfg["all_joint_names"])
    left = ["left_" + n for n in driver_cfg["left_panda_joint_names"] + driver_cfg["left_tesollo_joint_names"]]
    right = ["right_" + n for n in driver_cfg["right_panda_joint_names"] + driver_cfg["right_tesollo_joint_names"]]
    return left + right


def _driver_home_joint_pos(driver_cfg: dict) -> np.ndarray:
    left_chain = np.array(driver_cfg["panda_home_joint_positions"] + driver_cfg["tesollo_home_joint_positions"], dtype=np.float32)
    right_chain = np.array(driver_cfg["panda_home_joint_positions"] + driver_cfg["tesollo_home_joint_positions"], dtype=np.float32)
    return np.concatenate([left_chain, right_chain], axis=0)


def main() -> None:
    args = parse_args()

    rmi_root = Path(__file__).resolve().parents[3]
    runtime_dir = rmi_root / "runtime"
    system_id_dir = runtime_dir / "system_id"
    runtime_cfg_path = Path(args.runtime_cfg) if args.runtime_cfg else runtime_dir / "runtime_cfg.yaml"
    handenv_cfg_path = Path(args.handenv_cfg) if args.handenv_cfg else runtime_dir / "HandEnv.yaml"
    driver_cfg_path = Path(args.driver_cfg) if args.driver_cfg else rmi_root / "config" / "rl_bimanual_driver_config.yaml"
    output_pt_path = Path(args.output_pt) if args.output_pt else system_id_dir / "actions_sysid_traj_grid.pt"
    if output_pt_path.suffix == "":
        output_pt_path = output_pt_path.with_suffix(".pt")

    with open(runtime_cfg_path, "r", encoding="utf-8") as f:
        runtime_cfg = yaml.safe_load(f)
    with open(handenv_cfg_path, "r", encoding="utf-8") as f:
        handenv_cfg = yaml.safe_load(f)
    with open(driver_cfg_path, "r", encoding="utf-8") as f:
        driver_cfg = yaml.safe_load(f)

    dt = float(runtime_cfg["dt"])
    left_indices = runtime_cfg["policy_action_indices_dict"]["left"]
    right_indices = runtime_cfg["policy_action_indices_dict"]["right"]
    policy_action_dim = len(left_indices) + len(right_indices)

    # Policy/sim order (HandEnv): used as source indexing for --joint_indices.
    chain_joint_names = handenv_cfg["env"]["robot"]["jointNames"]["arm"] + handenv_cfg["env"]["robot"]["jointNames"]["hand"]
    policy_action_labels = [f"left_{name}" for name in chain_joint_names] + [f"right_{name}" for name in chain_joint_names]
    if len(policy_action_labels) != policy_action_dim:
        raise ValueError(
            f"Action dim mismatch: policy_action_dim={policy_action_dim}, "
            f"policy_joint_names={len(policy_action_labels)}"
        )

    # Real/driver order: primary order for replay to real robot.
    real_action_labels = _driver_joint_names(driver_cfg)
    if len(real_action_labels) != policy_action_dim:
        raise ValueError(
            f"Action dim mismatch: policy_action_dim={policy_action_dim}, "
            f"real_joint_names={len(real_action_labels)} from {driver_cfg_path}"
        )

    policy2real_indexing, real2policy_indexing = joint_mapping(policy_action_labels, real_action_labels)

    # Priority: driver cfg home for real-system replay.
    home_joint_pos = _driver_home_joint_pos(driver_cfg)
    if home_joint_pos.shape[0] != policy_action_dim:
        raise ValueError(
            f"Home dim mismatch: home_joint_pos={home_joint_pos.shape[0]}, "
            f"policy_action_dim={policy_action_dim}"
        )

    # Input joint indices are interpreted in real/driver order.
    stim_joint_indices = _parse_joint_indices(args.joint_indices, policy_action_dim)
    stim_joint_indices_policy = [int(real2policy_indexing[i]) for i in stim_joint_indices]
    num_traj = len(stim_joint_indices)

    waveforms = _parse_waveforms(args)
    amplitudes = _parse_amplitudes(args)

    payload_grid: dict[str, dict] = {}
    for waveform in waveforms:
        for amplitude in amplitudes:
            core = _build_core_signal(args, waveform, amplitude)
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

            actions = np.zeros((total_steps, num_traj, policy_action_dim), dtype=np.float32)
            for i, joint_idx in enumerate(stim_joint_indices):
                actions[:, i, joint_idx] = signal

            dones = np.zeros((total_steps, num_traj), dtype=bool)
            dones[-1, :] = True

            combo_key = f"wf{waveform}_amp{_format_amp_token(amplitude)}"
            payload_grid[combo_key] = {
                "actions": torch.from_numpy(actions),                          # [T, N, A]
                "dones": torch.from_numpy(dones),                              # [T, N]
                "dt": float(dt),
                "total_steps": total_steps,
                "num_traj": num_traj,
                "joint_names": real_action_labels,                             # legacy key: replay uses this
                "joint_names_real": real_action_labels,
                "joint_names_policy": policy_action_labels,
                "policy_to_real_indexing": [int(i) for i in policy2real_indexing],
                "real_to_policy_indexing": [int(i) for i in real2policy_indexing],
                "stim_joint_indices": stim_joint_indices,                      # legacy key: real indices
                "stim_joint_indices_real": stim_joint_indices,
                "stim_joint_indices_policy": stim_joint_indices_policy,
                "stim_joint_names": [real_action_labels[i] for i in stim_joint_indices],
                "stim_joint_names_real": [real_action_labels[i] for i in stim_joint_indices],
                "stim_joint_names_policy": [policy_action_labels[i] for i in stim_joint_indices_policy],
                "waveform": waveform,
                "amplitude": float(amplitude),
                "startup_home_steps": int(args.startup_home_steps),
                "signal": torch.from_numpy(signal),                            # [T]
                "home_joint_pos": torch.from_numpy(home_joint_pos),            # [A]
            }

    combo_keys = sorted(payload_grid.keys())
    output_payload: dict[str, dict] = {
        "__meta__": {
            "available_combo_keys": combo_keys,
            "num_combos": len(combo_keys),
            "waveforms": waveforms,
            "amplitudes": amplitudes,
            "joint_indices_real": stim_joint_indices,
            "joint_indices_policy": stim_joint_indices_policy,
            "policy_action_dim": policy_action_dim,
            "dt": float(dt),
        }
    }
    output_payload.update(payload_grid)

    output_pt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output_payload, output_pt_path)

    print(f"[saved] {output_pt_path}")
    print(f"[info] combos={len(combo_keys)}, waveforms={waveforms}, amplitudes={amplitudes}")
    print(f"[info] traj_count={num_traj}, policy_action_dim={policy_action_dim}, dt={dt}")
    print(f"[info] stimulated joints(real): {[real_action_labels[i] for i in stim_joint_indices][:6]} ... total={num_traj}")
    print(
        f"[info] stimulated joints(policy): "
        f"{[policy_action_labels[i] for i in stim_joint_indices_policy][:6]} ... total={num_traj}"
    )


if __name__ == "__main__":
    main()
