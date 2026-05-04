"""Launch the depth + fk + aux_policy stack.

Topology:
    rl_driver  -> /joint_states  ->  fk_node  -> /fk/poses
                  /joint_states  ->  aux_policy_2
    depth_node -> CUDA IPC handle (Trigger srv) -> aux_policy_2

rl_driver is hardware-specific and is *not* started here — bring it up
separately. This file only launches the three software nodes that depend
on /joint_states or on each other.

Startup order:
  - depth_node first (DA3 compile + warmup is the slowest, ~5–15s).
  - fk_node and aux_policy_2 start in parallel; aux_policy_2 blocks in
    its _fetch_depth_handle() service call until depth_node is ready.

Usage:
    ros2 launch robot_motion_interface_ros aux_policy_stack.launch.py
"""

from launch import LaunchDescription
from launch.actions import LogInfo
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    depth_node = Node(
        package="robot_motion_interface_ros",
        executable="depth_node",
        name="depth_node",
        output="screen",
        emulate_tty=True,
    )

    fk_node = Node(
        package="robot_motion_interface_ros",
        executable="fk_node",
        name="fk_node",
        output="screen",
        emulate_tty=True,
    )

    aux_policy = Node(
        package="robot_motion_interface_ros",
        executable="aux_policy_2",
        name="aux_policy_node",
        output="screen",
        emulate_tty=True,
    )

    return LaunchDescription([
        LogInfo(msg="Starting aux_policy stack: depth_node + fk_node + aux_policy_2"),
        depth_node,
        fk_node,
        aux_policy,
    ])
