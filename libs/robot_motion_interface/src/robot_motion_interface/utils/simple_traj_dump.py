"""Dump one trajectory (by traj_id) from runtime/actions_done_trace.pt to .npy.

Rules:
- Select one trajectory by traj_id (default: 0).
- Truncate at the first done step (inclusive).
- Save as numpy array in runtime directory.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dump one trajectory from actions_done_trace.pt")
    parser.add_argument("--traj_id", type=int, default=0, help="Trajectory index in env dimension.")
    parser.add_argument(
        "--input_pt",
        type=str,
        default=None,
        help="Input trace .pt path. Default: <RMI_ROOT>/runtime/actions_done_trace.pt",
    )
    parser.add_argument(
        "--output_npy",
        type=str,
        default=None,
        help="Output .npy path. Default: <RMI_ROOT>/runtime/traj_trace_id{traj_id}.npy",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    rmi_root = Path(__file__).resolve().parents[3]
    runtime_dir = rmi_root / "runtime"
    input_pt = Path(args.input_pt) if args.input_pt is not None else runtime_dir / "actions_done_trace.pt"
    output_npy = (
        Path(args.output_npy)
        if args.output_npy is not None
        else runtime_dir / f"traj_trace_id{args.traj_id}.npy"
    )

    trace = torch.load(input_pt, map_location="cpu")
    actions = torch.as_tensor(trace["actions"], dtype=torch.float32)  # [T, N, A] or [T, A]
    dones = torch.as_tensor(trace["dones"], dtype=torch.bool) if "dones" in trace else None

    if actions.ndim == 3:
        traj = actions[:, args.traj_id, :]  # [T, A]
        done_vec = dones[:, args.traj_id] if dones is not None else None  # [T]
    else:
        traj = actions  # [T, A]
        done_vec = dones if dones is not None else None

    if done_vec is not None:
        done_idx = torch.where(done_vec)[0]
        if len(done_idx) > 0:
            end = int(done_idx[0].item()) + 1
            traj = traj[:end]

    traj_np = traj.cpu().numpy().astype(np.float32)
    np.save(output_npy, traj_np)
    print(f"[saved] {output_npy} shape={traj_np.shape}")


if __name__ == "__main__":
    main()
