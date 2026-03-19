"""
Replay one retargeted trajectory NPZ to rl_driver_node at a fixed publish rate.

Workflow
--------
1. Load one trajectory NPZ (expects key: traj_full).
2. Parse (task, episode) from trajectory path.
3. Validate against report JSON:
   - pregrasp_plan_success == 1
   - new_ik_success_both_rate >= 0.05
4. If valid: publish each waypoint to /target_joint_states at 30 Hz and exit.
5. If invalid: log reason and abort without publishing.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np
import rclpy
import yaml
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState

# Keep this node independent from robot_motion_interface pybind import chain.
T_JS_QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)
_TRJ_NAME_RE = re.compile(r"^(?P<episode>.+?)_curobo_2stage(?:-interpolated)?$")


def _find_project_root() -> Path:
    candidates: list[Path] = []
    env_var = os.environ.get("DEXTERITY_PROJECT_ROOT")
    if env_var:
        candidates.append(Path(env_var).expanduser().resolve())

    # Priority 2: infer from current file / cwd by walking up to repo markers.
    candidates.extend([Path(__file__).resolve(), Path.cwd().resolve()])

    for c in candidates:
        probe = c if c.is_dir() else c.parent
        for parent in [probe, *probe.parents]:
            if (parent / ".git").exists() and (parent / "models").exists() and (parent / "libs").exists():
                return parent

    # Fallback to common container location.
    if Path("/workspace").exists():
        return Path("/workspace").resolve()
    raise RuntimeError("Cannot determine project root. Set DEXTERITY_PROJECT_ROOT.")


PROJECT_ROOT = _find_project_root()
RMI_ROOT = PROJECT_ROOT / "libs" / "robot_motion_interface"


def _resolve_project_path(raw: str) -> Path:
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p.resolve()


def _episode_equal(a: str, b: str) -> bool:
    if a == b:
        return True
    if a.isdigit() and b.isdigit():
        return int(a) == int(b)
    return False


class RetargetTrajPlayerNode(Node):
    def __init__(self) -> None:
        super().__init__("retarget_traj_player_node")

        default_cfg = str(RMI_ROOT / "config" / "rl_bimanual_driver_config.yaml")
        default_report = str(
            PROJECT_ROOT
            / "models"
            / "egodex"
            / "traj-retarging"
            / "reports"
            / "curobo_2stage_vs_baseline.json"
        )

        self.declare_parameter("config_path", default_cfg)
        self.declare_parameter("traj_path", "")
        self.declare_parameter("report_path", default_report)
        self.declare_parameter("publish_hz", 30.0)

        cfg_path = _resolve_project_path(
            self.get_parameter("config_path").get_parameter_value().string_value
        )
        traj_raw = self.get_parameter("traj_path").get_parameter_value().string_value
        report_path = _resolve_project_path(
            self.get_parameter("report_path").get_parameter_value().string_value
        )
        self._publish_hz = float(self.get_parameter("publish_hz").value)
        if self._publish_hz <= 0.0:
            raise ValueError(f"publish_hz must be > 0, got {self._publish_hz}")

        if not traj_raw:
            raise ValueError("traj_path is required and cannot be empty")
        traj_path = _resolve_project_path(traj_raw)

        self.get_logger().info(f"config_path: {cfg_path}")
        self.get_logger().info(f"traj_path:   {traj_path}")
        self.get_logger().info(f"report_path: {report_path}")
        self.get_logger().info(f"publish_hz:  {self._publish_hz}")

        self.joint_names = self._load_joint_names(cfg_path)
        self._traj = self._load_traj(traj_path)

        task, episode = self._parse_task_episode(traj_path)
        report_row = self._find_report_row(report_path, task, episode)
        self._validate_row_or_raise(report_row, task, episode)

        self._idx = 0
        self._pub = self.create_publisher(JointState, "/target_joint_states", T_JS_QOS)
        self._timer = self.create_timer(1.0 / self._publish_hz, self._tick)
        self.get_logger().info(
            f"Validation passed. Start publishing {len(self._traj)} waypoints at {self._publish_hz:.1f} Hz."
        )

    def _load_joint_names(self, cfg_path: Path) -> list[str]:
        if not cfg_path.exists():
            raise FileNotFoundError(f"Config path not found: {cfg_path}")
        with cfg_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        l_names = [
            "left_" + n
            for n in cfg["left_panda_joint_names"] + cfg["left_tesollo_joint_names"]
        ]
        r_names = [
            "right_" + n
            for n in cfg["right_panda_joint_names"] + cfg["right_tesollo_joint_names"]
        ]
        names = l_names + r_names
        if len(names) != 38:
            raise ValueError(f"Expected 38 joint names, got {len(names)}")
        return names

    def _load_traj(self, traj_path: Path) -> np.ndarray:
        if not traj_path.exists():
            raise FileNotFoundError(f"Trajectory not found: {traj_path}")
        with np.load(str(traj_path), allow_pickle=True) as npz:
            if "traj_full" not in npz.files:
                raise KeyError(f"traj_full not found in {traj_path}")
            traj = np.asarray(npz["traj_full"], dtype=np.float32)
        if traj.ndim != 2:
            raise ValueError(f"traj_full must be 2D, got shape {traj.shape}")
        if traj.shape[1] != 38:
            raise ValueError(f"traj_full must have 38 DoF, got shape {traj.shape}")
        return traj

    def _parse_task_episode(self, traj_path: Path) -> tuple[str, str]:
        task = traj_path.parent.name
        m = _TRJ_NAME_RE.match(traj_path.stem)
        if m is None:
            raise ValueError(
                "Trajectory filename must match '<episode>_curobo_2stage(.npz|"
                "-interpolated.npz)'"
            )
        episode = m.group("episode")
        return task, episode

    def _find_report_row(self, report_path: Path, task: str, episode: str) -> dict:
        if not report_path.exists():
            raise FileNotFoundError(f"Report not found: {report_path}")
        with report_path.open("r", encoding="utf-8") as f:
            report = json.load(f)
        rows = report.get("episodes", [])
        for row in rows:
            row_task = str(row.get("task", ""))
            row_ep = str(row.get("episode", ""))
            if row_task == task and _episode_equal(row_ep, episode):
                return row
        raise RuntimeError(
            f"No matching report row found for task='{task}', episode='{episode}' in {report_path}"
        )

    def _validate_row_or_raise(self, row: dict, task: str, episode: str) -> None:
        plan_ok = int(row.get("pregrasp_plan_success", 0)) == 1
        both_rate = float(row.get("new_ik_success_both_rate", 0.0))
        valid = plan_ok and (both_rate >= 0.05)
        if valid:
            self.get_logger().info(
                f"[VALID] task={task} episode={episode} pregrasp_plan_success=1 "
                f"both_rate={both_rate:.4f}"
            )
            return

        reason = row.get("failure_reason_codes", "")
        self.get_logger().error(
            "[INVALID] Reject playback: "
            f"task={task} episode={episode} "
            f"pregrasp_plan_success={int(row.get('pregrasp_plan_success', 0))} "
            f"both_rate={both_rate:.4f} "
            f"failure_reason_codes={reason}"
        )
        raise RuntimeError("Trajectory did not pass validity gate.")

    def _tick(self) -> None:
        if self._idx >= self._traj.shape[0]:
            self.get_logger().info("Trajectory publish complete. Shutting down.")
            self._timer.cancel()
            rclpy.shutdown()
            return
        self._publish(self._traj[self._idx])
        self._idx += 1

    def _publish(self, q: np.ndarray) -> None:
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.joint_names
        msg.position = q.tolist()
        self._pub.publish(msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node: RetargetTrajPlayerNode | None = None
    try:
        node = RetargetTrajPlayerNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        print(f"[retarget_traj_player_node] ERROR: {exc}")
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
