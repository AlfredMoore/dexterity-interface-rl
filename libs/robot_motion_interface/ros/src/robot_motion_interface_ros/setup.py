import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'robot_motion_interface_ros'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='root@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
    },
    entry_points={
        'console_scripts': [
            'interface = robot_motion_interface_ros.interface_node:main',
            'rl_driver = robot_motion_interface_ros.rl_driver_node:main',
            'rl_policy = robot_motion_interface_ros.rl_policy_node:main',
            'aux_policy = robot_motion_interface_ros.aux_policy:main',
            'aux_policy_2 = robot_motion_interface_ros.aux_policy_v2:main',
            'depth_feat_policy = robot_motion_interface_ros.depth_feat_policy:main',
            'depth_feat_node = robot_motion_interface_ros.depth_feat_node:main',
            'depth_sam_feat_node = robot_motion_interface_ros.depth_sam_feat_node:main',
            'depth_feat_node_collect = robot_motion_interface_ros.depth_feat_node_collect:main',
            'depth_feat_node_all_data = robot_motion_interface_ros.depth_feat_node_all_data:main',
            'bottle_apriltag_node = robot_motion_interface_ros.bottle_apriltag_node:main',
            'bottle_apriltag_node_all_data = robot_motion_interface_ros.bottle_apriltag_node_all_data:main',
            'apriltag_policy = robot_motion_interface_ros.apriltag_policy:main',
            'fk_node = robot_motion_interface_ros.fk_node:main',
            'depth_node = robot_motion_interface_ros.depth_node:main',
            'kinect_node = robot_motion_interface_ros.kinect_node:main',
            'cv_node = robot_motion_interface_ros.cv_node:main',
            'test_pre_grasp = robot_motion_interface_ros.test_node_pre_grasp:main',
            'test_curobo = robot_motion_interface_ros.test_node_for_curobo:main',
            'test_retarget_traj_player = robot_motion_interface_ros.test_node_retarget_traj_player:main',
            'run_traj = robot_motion_interface_ros.node_run_traj:main',
            'replay_target_pregrasp = robot_motion_interface_ros.replay_target_np:main',
            'replay_target_policy = robot_motion_interface_ros.replay_target_torch:main',
            'replay_action_torch = robot_motion_interface_ros.replay_action_torch:main',
            'yolo_node_bbox = robot_motion_interface_ros.yolo_node_bbox:main',
            'yolo_node_seg = robot_motion_interface_ros.yolo_node_seg:main',
            'gsam2_node = robot_motion_interface_ros.gsam2_node:main',
            'sam_node = robot_motion_interface_ros.sam_node:main',
            "kinect_sam_c2d_node = robot_motion_interface_ros.kinect_sam_c2d_node:main",
        ],
    },
)
