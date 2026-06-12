"""SYSID Step 0 -- gravity-compensation / load diagnostic (READ-ONLY).

Purpose
-------
Confirm the suspected right-arm gravity over-compensation and the fix lever,
WITHOUT commanding any motion:
  1. Read each arm's configured load (m_load / F_x_Cload / I_load / m_ee / m_total)
     -- whatever is set in the Franka Desk (FCI web) UI.
  2. Compute the model-based gravity torque (franka::Model::gravity), optionally at
     the config home pose (explicit-q overload) for a fair pose-matched L/R compare.
  3. Read the resting external-torque estimate (tau_ext_hat_filtered) and tau_J.
  4. Compare LEFT vs RIGHT (offline, from two dumps) -> locate the asymmetry.
  5. (optional) Try code-side correction via sysid_set_load(), re-read to verify it
     actually changed; if it does NOT, the user sets the load in Desk instead.

Single-arm by design: each run connects ONE arm and dumps its diagnostic; the
LEFT-vs-RIGHT comparison is offline from two dumps (no dual-arm process).

Safety: NEVER starts the control loop and NEVER sends targets. Only opens an FCI
connection and calls the read-only sysid_* methods (readOnce()/loadModel()).

Run:
    python -m robot_motion_interface.sysid.sysid_gravity_diag --arm left  --out left.json
    python -m robot_motion_interface.sysid.sysid_gravity_diag --arm right --out right.json
    python -m robot_motion_interface.sysid.sysid_gravity_diag --compare left.json right.json
    python -m robot_motion_interface.sysid.sysid_gravity_diag --arm right \
        --set-load-mass 0.0 --verify   # try code injection + re-read
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from robot_motion_interface.panda.panda_interface import PandaInterface

# RMI root = libs/robot_motion_interface (has config/ and the urdf relative path).
_RMI_ROOT = Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SYSID gravity/load diagnostic (read-only).")
    parser.add_argument(
        "--arm",
        choices=["left", "right"],
        default=None,
        help="Single arm to diagnose. Connects ONE robot, read-only.",
    )
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("LEFT_DUMP", "RIGHT_DUMP"),
        default=None,
        help="Offline mode: read two prior --out dumps and print a LEFT-vs-RIGHT table.",
    )
    parser.add_argument(
        "--driver_cfg",
        type=str,
        default=None,
        help="Path to rl_bimanual_driver_config.yaml. Default: package config/.",
    )
    parser.add_argument(
        "--gravity-at-home",
        action="store_true",
        help="Evaluate sysid_gravity at the CONFIG home q (explicit-q overload) for a "
        "fair pose-matched L/R compare, instead of the robot's current physical pose.",
    )
    # --- optional correction-verification path (Step 0.5) ---
    parser.add_argument(
        "--set-load-mass",
        type=float,
        default=None,
        help="If set, attempt code-side sysid_set_load with this mass (kg) on --arm.",
    )
    parser.add_argument(
        "--set-load-com",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 0.0),
        help="Load CoM (x y z, m) used with --set-load-mass.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="After sysid_set_load, re-read load/gravity/tau to confirm the change took.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Optional path to dump the diagnostic (json) for later --compare.",
    )
    return parser.parse_args()


def _load_driver_cfg(path: str | None) -> dict:
    cfg_path = Path(path) if path else _RMI_ROOT / "config" / "rl_bimanual_driver_config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"driver config not found: {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _connect_arm(arm: str, cfg: dict) -> PandaInterface:
    """Build a read-only PandaInterface for one arm from the driver config.

    Does NOT call start_loop()/home() -- read-only diagnostic only.
    """
    hostname = cfg[f"{arm}_panda_hostname"]
    joint_names = cfg[f"{arm}_panda_joint_names"]
    urdf_path = str((_RMI_ROOT / cfg["panda_urdf_path"]).resolve())
    home = np.array(cfg["panda_home_joint_positions"], dtype=float)
    kp = np.array(cfg["panda_kp"], dtype=float)
    kd = np.array(cfg["panda_kd"], dtype=float)
    return PandaInterface(hostname, urdf_path, joint_names, home, kp, kd)


def _read_diag(panda: PandaInterface, cfg: dict, gravity_at_home: bool) -> dict:
    """Read the read-only diagnostic quantities from one arm -> dict of lists."""
    home_q = np.array(cfg["panda_home_joint_positions"], dtype=float)
    grav_q = home_q if gravity_at_home else None
    js = np.asarray(panda.joint_state(), dtype=float)  # (14,) = [q(7), dq(7)]
    return {
        "load_info": np.asarray(panda.sysid_load_info(), dtype=float).tolist(),
        "gravity": np.asarray(panda.sysid_gravity(grav_q), dtype=float).tolist(),
        "gravity_at": "home" if gravity_at_home else "current",
        "tau_ext": np.asarray(panda.sysid_tau_ext(), dtype=float).tolist(),
        "tau_measured": np.asarray(panda.sysid_tau_measured(), dtype=float).tolist(),
        "q": js[:7].tolist(),
        "dq": js[7:].tolist(),
    }


def _fmt_load(load_info: list[float]) -> str:
    li = np.asarray(load_info, dtype=float)
    # [m_ee, m_load, m_total, F_x_Cload(3), I_load(9)]
    return (
        f"m_ee={li[0]:.4f}  m_load={li[1]:.4f}  m_total={li[2]:.4f}  "
        f"F_x_Cload={np.round(li[3:6], 4).tolist()}"
    )


def _print_one(tag: str, d: dict) -> None:
    print(f"\n=== {tag} ===")
    print(f"  load  : {_fmt_load(d['load_info'])}")
    print(f"  gravity[{d['gravity_at']}] (Nm): {np.round(d['gravity'], 3).tolist()}")
    print(f"  tau_ext (Nm)     : {np.round(d['tau_ext'], 3).tolist()}")
    print(f"  tau_measured (Nm): {np.round(d['tau_measured'], 3).tolist()}")
    print(f"  q (rad)          : {np.round(d['q'], 3).tolist()}")


def _compare(left: dict, right: dict) -> None:
    """Print a LEFT-vs-RIGHT table and flag asymmetries (offline, from two dumps)."""
    _print_one("LEFT", left)
    _print_one("RIGHT", right)

    ll = np.asarray(left["load_info"], dtype=float)
    rl = np.asarray(right["load_info"], dtype=float)
    print("\n=== LEFT vs RIGHT ===")
    print(f"  d(m_load)   = {rl[1] - ll[1]:+.4f} kg  (right - left)")
    print(f"  d(m_total)  = {rl[2] - ll[2]:+.4f} kg")

    if left["gravity_at"] != right["gravity_at"]:
        print("  [warn] gravity evaluated at DIFFERENT poses -> not comparable. "
              "Re-run both with --gravity-at-home.")
    else:
        dg = np.asarray(right["gravity"]) - np.asarray(left["gravity"])
        print(f"  d(gravity)  (Nm) = {np.round(dg, 3).tolist()}")

    lext = np.linalg.norm(left["tau_ext"])
    rext = np.linalg.norm(right["tau_ext"])
    print(f"  |tau_ext| left={lext:.3f}  right={rext:.3f}  (resting residual)")
    if rext > 1.5 * max(lext, 1e-6):
        print("  [flag] RIGHT resting tau_ext markedly larger -> load/gravity mismatch "
              "(consistent with over-compensation). Check m_load / Desk config.")


def _try_set_load_and_verify(panda: PandaInterface, cfg: dict, gravity_at_home: bool,
                             mass: float, com: tuple[float, float, float]) -> None:
    """Attempt code-side load correction, then re-read to confirm it changed."""
    before = _read_diag(panda, cfg, gravity_at_home)
    _print_one("BEFORE set_load", before)

    inertia = np.zeros((3, 3), dtype=float)  # point-mass approx for the diagnostic
    ok = panda.sysid_set_load(mass, np.asarray(com, dtype=float), inertia)
    print(f"\nsysid_set_load(mass={mass}, com={com}) -> accepted={ok}")

    after = _read_diag(panda, cfg, gravity_at_home)
    _print_one("AFTER set_load", after)

    changed = not np.allclose(before["load_info"], after["load_info"])
    if changed:
        print("\n[result] load read-back CHANGED -> code-side setLoad is the fix lever.")
    else:
        print("\n[result] load read-back UNCHANGED -> FCI likely honors only the "
              "Desk-configured load. Set the correct load in Franka Desk instead.")


def main() -> None:
    args = parse_args()

    # Offline compare branch.
    if args.compare is not None:
        with open(args.compare[0], "r", encoding="utf-8") as f:
            left = json.load(f)
        with open(args.compare[1], "r", encoding="utf-8") as f:
            right = json.load(f)
        _compare(left, right)
        return

    # Online single-arm branch.
    if args.arm is None:
        raise SystemExit("Provide --arm {left,right} (or --compare A B for offline).")

    cfg = _load_driver_cfg(args.driver_cfg)
    panda = _connect_arm(args.arm, cfg)

    if args.set_load_mass is not None:
        _try_set_load_and_verify(panda, cfg, args.gravity_at_home,
                                 args.set_load_mass, tuple(args.set_load_com))
        return

    diag = _read_diag(panda, cfg, args.gravity_at_home)
    _print_one(args.arm.upper(), diag)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"arm": args.arm, **diag}, f, indent=2)
        print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
