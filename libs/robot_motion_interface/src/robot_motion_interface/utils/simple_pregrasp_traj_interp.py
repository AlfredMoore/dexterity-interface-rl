"""Generate a simple two-stage pre-grasp trajectory from runtime HandEnv.yaml.

Stage 1 (100 steps): move left chain HOME -> PRE_GRASP, keep right chain at HOME.
Stage 2 (100 steps): keep left chain at PRE_GRASP, move right chain HOME -> PRE_GRASP.

Outputs (.npy) are saved to libs/robot_motion_interface/runtime:
  - traj_pregrasp_left_100.npy   (100, 38)
  - traj_pregrasp_right_100.npy  (100, 38)
  - traj_pregrasp_full_200.npy   (200, 38)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from robot_motion_interface.utils.sim2real import joint_mapping


def _read_chain_pose(
    cfg: dict,
    mode: str,
    side: str,
    chain_joint_names: list[str],
) -> np.ndarray:
    joint_pose = cfg["experiment"][mode][f"{side}_joint_pose"]
    return np.array([joint_pose[name] for name in chain_joint_names], dtype=np.float32)


def main() -> None:
    rmi_root = Path(__file__).resolve().parents[3]
    runtime_dir = rmi_root / "runtime"
    cfg_path = runtime_dir / "HandEnv.yaml"
    driver_cfg_path = rmi_root / "config" / "rl_bimanual_driver_config.yaml"

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    with open(driver_cfg_path, "r", encoding="utf-8") as f:
        driver_cfg = yaml.safe_load(f)

    chain_joint_names = cfg["env"]["robot"]["jointNames"]["arm"] + cfg["env"]["robot"]["jointNames"]["hand"]
    action_per_chain = int(cfg["env"]["action"]["actionPerChain"])
    action_space = int(cfg["env"]["action"]["actionSpace"])

    # sim (HandEnv) hand joint order differs from real driver order — reindex per chain
    real_chain_names = driver_cfg["left_panda_joint_names"] + driver_cfg["left_tesollo_joint_names"]
    chain2real_idx, _ = joint_mapping(chain_joint_names, real_chain_names)
    # build full bimanual reindex: [left_chain reordered, right_chain reordered]
    full_reindex = chain2real_idx + [i + action_per_chain for i in chain2real_idx]

    l_home = _read_chain_pose(cfg, mode="home", side="left", chain_joint_names=chain_joint_names)
    r_home = _read_chain_pose(cfg, mode="home", side="right", chain_joint_names=chain_joint_names)
    l_pre = _read_chain_pose(cfg, mode="pre_grasp", side="left", chain_joint_names=chain_joint_names)
    r_pre = _read_chain_pose(cfg, mode="pre_grasp", side="right", chain_joint_names=chain_joint_names)

    if action_per_chain != l_home.shape[0] or action_space != action_per_chain * 2:
        raise ValueError(
            f"Action shape mismatch: action_per_chain={action_per_chain}, action_space={action_space}, "
            f"left_chain={l_home.shape[0]}"
        )

    alphas = np.linspace(0.0, 1.0, 100, dtype=np.float32)

    traj_left = np.zeros((100, action_space), dtype=np.float32)
    traj_left[:, :action_per_chain] = (1.0 - alphas[:, None]) * l_home[None, :] + alphas[:, None] * l_pre[None, :]
    traj_left[:, action_per_chain:] = r_home[None, :]

    traj_right = np.zeros((100, action_space), dtype=np.float32)
    traj_right[:, :action_per_chain] = l_pre[None, :]
    traj_right[:, action_per_chain:] = (1.0 - alphas[:, None]) * r_home[None, :] + alphas[:, None] * r_pre[None, :]

    traj_full = np.concatenate([traj_left, traj_right], axis=0)

    # reorder sim joint order → real driver order
    traj_left = traj_left[:, full_reindex]
    traj_right = traj_right[:, full_reindex]
    traj_full = traj_full[:, full_reindex]

    out_left = runtime_dir / "traj_pregrasp_left_100.npy"
    out_right = runtime_dir / "traj_pregrasp_right_100.npy"
    out_full = runtime_dir / "traj_pregrasp_full_200.npy"

    np.save(out_left, traj_left)
    np.save(out_right, traj_right)
    np.save(out_full, traj_full)

    print(f"[saved] {out_left} shape={traj_left.shape}")
    print(f"[saved] {out_right} shape={traj_right.shape}")
    print(f"[saved] {out_full} shape={traj_full.shape}")


if __name__ == "__main__":
    main()
