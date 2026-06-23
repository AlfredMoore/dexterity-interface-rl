#!/usr/bin/env python3
"""Read the current bimanual joint state and step-interpolate to the pre-grasp pose.

Unlike replay_target_np (which streams a precomputed, time-paced trajectory), this
script builds the path online from wherever the arms currently are:

  1. Read the latest /joint_states (current 38-DoF position).
  2. Use the pre-grasp target (hardcoded DEFAULT_PREGRASP, or --pregrasp override),
     or the HOME pose (DEFAULT_HOME) when --home is passed.
  3. Linearly interpolate current -> target with a fixed per-step joint increment
     ("step size" in rad), NOT a fixed time budget. The number of waypoints is
     ceil(max|target - current| / step_size), so no joint moves more than
     step_size per published setpoint regardless of how far the start pose is.
  4. Publish each waypoint to /target_joint_states.

publish_hz only paces playback; it does not change the interpolation shape, which
is governed entirely by step_size.

Joint order (38 DoF) matches the driver's all_joint_names (positional contract;
the driver reads /target_joint_states by index, not by name):
  left panda(7) | left tesollo(12) | right panda(7) | right tesollo(12)

Example:
  python -m robot_motion_interface.utils.interp_to_pregrasp --step_size 0.01 --publish_hz 20 -y
  python -m robot_motion_interface.utils.interp_to_pregrasp --home --step_size 0.01 --publish_hz 20 -y   # retract to HOME
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import rclpy
import yaml
from rclpy.node import Node
from sensor_msgs.msg import JointState

from robot_motion_interface.utils.qos import HIGH_PERF_QOS, HIGH_RELIA_QOS

# libs/robot_motion_interface (this file lives in .../src/robot_motion_interface/utils)
RMI_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = RMI_ROOT / "config" / "rl_bimanual_driver_config.yaml"

# Hardcoded 38-DoF pre-grasp target (refined real-robot pose, mirrors cli.sh "Step 1").
# Order: left panda(7) | left tesollo(12) | right panda(7) | right tesollo(12).
# Override at runtime with --pregrasp.
DEFAULT_PREGRASP = [
    # left panda (7)
    -1.0795, 0.9081, 0.1157, -1.6087, 1.0009, 1.317, 0.0918,
    # left tesollo (12): F1/F2/F3 x M1..M4
    -0.0069, -0.008, 0.6158, 0.2031,
    -0.0096, 0.0008, 0.6095, 0.214,
    -0.0088, -0.0017, 0.614, 0.2053,
    # right panda (7)
    0.1002, 0.1566, 0.0014, -1.6126, 0.0015, 1.6172, 0.75,
    # right tesollo (12): F1/F2/F3 x M1..M4
    0.0375, -0.0375, 0.5625, 0.2343,
    -0.0375, 0.0375, 0.5625, 0.2375,
    0.0375, -0.0375, 0.5625, 0.1625,
]

# Hardcoded 38-DoF HOME (retracted/neutral) pose, selected with --home instead of the
# pre-grasp. Values from config/bimanual_arm_config.yaml (panda_home_joint_positions +
# tesollo all-zero), applied to BOTH arms.
DEFAULT_HOME = [
    # left panda (7) -- Franka neutral
    0.0, -0.7854, 0.0, -2.3562, 0.0, 1.5708, 0.7854,
    # left tesollo (12) -- open hand
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    # right panda (7)
    0.0, -0.7854, 0.0, -2.3562, 0.0, 1.5708, 0.7854,
    # right tesollo (12)
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
]


def load_joint_names(config_path: Path) -> list[str]:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return list(cfg["all_joint_names"])


def build_step_interp(q0: np.ndarray, qT: np.ndarray, step_size: float) -> np.ndarray:
    """Linear interpolation q0 -> qT with a bounded per-step joint increment.

    Returns waypoints of shape (num_steps, dof), EXCLUDING q0 and INCLUDING qT.
    num_steps = ceil(max|qT - q0| / step_size), so every joint moves at most
    step_size per step. Granularity is set by step_size (rad), not by time.
    """
    delta = qT - q0
    max_abs = float(np.max(np.abs(delta))) if delta.size else 0.0
    num_steps = max(1, math.ceil(max_abs / step_size))
    alphas = np.arange(1, num_steps + 1, dtype=np.float64) / num_steps  # (num_steps,) in (0, 1]
    return q0[None, :] + alphas[:, None] * delta[None, :]


class InterpToPregraspNode(Node):
    def __init__(self, joint_names: list[str], state_topic: str, target_topic: str, publish_name: bool):
        super().__init__("interp_to_pregrasp_node")
        self.joint_names = joint_names
        self.dof = len(joint_names)
        self.publish_name = publish_name
        self.state_topic = state_topic
        self.target_topic = target_topic

        self._latest_pos: np.ndarray | None = None

        # /joint_states is published BEST_EFFORT by the driver (HIGH_PERF_QOS); a
        # RELIABLE subscriber would not match it, so use the same profile.
        self.state_sub = self.create_subscription(
            JointState, state_topic, self._state_cb, HIGH_PERF_QOS
        )
        # /target_joint_states is consumed RELIABLE by the driver (HIGH_RELIA_QOS).
        self.target_pub = self.create_publisher(JointState, target_topic, HIGH_RELIA_QOS)

    def _state_cb(self, msg: JointState) -> None:
        if len(msg.position) != self.dof:
            self.get_logger().error(
                f"{self.state_topic} DoF mismatch: got {len(msg.position)}, expected {self.dof}"
            )
            return
        self._latest_pos = np.asarray(msg.position, dtype=np.float64)

    def wait_for_state(self, timeout_sec: float) -> np.ndarray:
        """Spin until the first valid /joint_states snapshot arrives."""
        self.get_logger().info(f"Waiting for current state on {self.state_topic} ...")
        start = time.monotonic()
        while rclpy.ok() and self._latest_pos is None:
            rclpy.spin_once(self, timeout_sec=0.1)
            if time.monotonic() - start > timeout_sec:
                raise TimeoutError(
                    f"No message on {self.state_topic} within {timeout_sec:.1f}s. Is rl_driver running?"
                )
        self.get_logger().info("Current state received.")
        return self._latest_pos.copy()

    def wait_for_subscriber(self, timeout_sec: float) -> None:
        """Wait until the driver subscribes to the target topic.

        QoS is RELIABLE + VOLATILE, so any setpoint published before DDS discovery
        matches the subscriber is dropped and never resent; without this wait the
        first interpolation steps would be lost.
        """
        self.get_logger().info(f"Waiting for a subscriber on {self.target_topic} ...")
        start = time.monotonic()
        while rclpy.ok() and self.target_pub.get_subscription_count() == 0:
            time.sleep(0.1)
            if time.monotonic() - start > timeout_sec:
                raise TimeoutError(
                    f"No subscriber on {self.target_topic} within {timeout_sec:.1f}s. Is rl_driver running?"
                )
        time.sleep(0.5)  # margin for the link to fully establish
        self.get_logger().info(f"Subscriber connected ({self.target_pub.get_subscription_count()}).")

    def publish_target(self, q: np.ndarray) -> None:
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        if self.publish_name:
            msg.name = self.joint_names
        msg.position = q.tolist()
        self.target_pub.publish(msg)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Read current bimanual state and step-interpolate to the pre-grasp pose."
    )
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    parser.add_argument("--step_size", type=float, default=0.01,
                        help="Max per-step joint increment in rad (sets interp granularity).")
    parser.add_argument("--publish_hz", type=float, default=30.0,
                        help="Waypoint playback rate; paces output only, not interp shape.")
    parser.add_argument("--state_topic", type=str, default="/joint_states")
    parser.add_argument("--target_topic", type=str, default="/target_joint_states")
    parser.add_argument("--publish_name", action="store_true",
                        help="Populate JointState.name (driver reads by index; off by default).")
    parser.add_argument("--settle_sec", type=float, default=0.5,
                        help="Keep republishing the final target for this long at the end.")
    parser.add_argument("-y", "--yes", action="store_true",
                        help="Skip the confirmation prompt before moving.")
    parser.add_argument("--state_timeout", type=float, default=10.0)
    parser.add_argument("--sub_timeout", type=float, default=30.0)
    parser.add_argument("--pregrasp", type=float, nargs="+", default=None,
                        help="Explicit pre-grasp target as DoF floats; overrides DEFAULT_PREGRASP.")
    parser.add_argument("--home", action="store_true",
                        help="Interpolate to the hardcoded HOME (retracted) pose instead of pre-grasp.")
    args = parser.parse_args(argv)

    if args.step_size <= 0.0:
        raise ValueError(f"--step_size must be > 0, got {args.step_size}")
    if args.publish_hz <= 0.0:
        raise ValueError(f"--publish_hz must be > 0, got {args.publish_hz}")

    joint_names = load_joint_names(Path(args.config))
    dof = len(joint_names)
    if args.home:
        target, dest_name = DEFAULT_HOME, "HOME"
    else:
        target = args.pregrasp if args.pregrasp is not None else DEFAULT_PREGRASP
        dest_name = "PRE-GRASP"
    if len(target) != dof:
        raise ValueError(f"{dest_name} target expects {dof} values, got {len(target)}")
    qT = np.asarray(target, dtype=np.float64)

    rclpy.init()
    node = InterpToPregraspNode(
        joint_names=joint_names,
        state_topic=args.state_topic,
        target_topic=args.target_topic,
        publish_name=args.publish_name,
    )
    try:
        q0 = node.wait_for_state(args.state_timeout)
        node.wait_for_subscriber(args.sub_timeout)

        waypoints = build_step_interp(q0, qT, args.step_size)
        delta = qT - q0
        max_abs = float(np.max(np.abs(delta)))
        worst = int(np.argmax(np.abs(delta)))
        num_steps = waypoints.shape[0]

        node.get_logger().info(
            "Step-interp plan:\n"
            f"  DoF             : {dof}\n"
            f"  step_size (rad) : {args.step_size}\n"
            f"  max joint delta : {max_abs:.4f} rad @ idx {worst} ({joint_names[worst]})\n"
            f"  num steps       : {num_steps}\n"
            f"  publish_hz      : {args.publish_hz}\n"
            f"  est. duration   : {num_steps / args.publish_hz:.2f} s"
        )

        if not args.yes:
            input(
                f">>> Robot will move from CURRENT pose to {dest_name}. "
                "Keep a hand on the e-stop. Press Enter to start (Ctrl-C to abort)..."
            )

        dt = 1.0 / args.publish_hz
        deadline = time.perf_counter()
        for i in range(num_steps):
            node.publish_target(waypoints[i])
            if i % 20 == 0 or i == num_steps - 1:
                node.get_logger().info(f"step {i + 1}/{num_steps}")
            deadline += dt
            sleep_s = deadline - time.perf_counter()
            if sleep_s > 0:
                time.sleep(sleep_s)

        # Hold the final target briefly so the driver settles on it.
        settle_end = time.perf_counter() + max(0.0, args.settle_sec)
        while rclpy.ok() and time.perf_counter() < settle_end:
            node.publish_target(qT)
            time.sleep(dt)
        node.get_logger().info(f"Reached {dest_name} target.")

    except KeyboardInterrupt:
        node.get_logger().warn("Interrupted; stopping (driver holds last sent target).")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
