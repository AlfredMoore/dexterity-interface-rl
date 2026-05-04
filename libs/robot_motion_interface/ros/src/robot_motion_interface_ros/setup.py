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
            'fk_node = robot_motion_interface_ros.fk_node:main',
            'cv_node = robot_motion_interface_ros.cv_node:main',
            'test_pre_grasp = robot_motion_interface_ros.test_node_pre_grasp:main',
            'test_curobo = robot_motion_interface_ros.test_node_for_curobo:main',
            'test_retarget_traj_player = robot_motion_interface_ros.test_node_retarget_traj_player:main',
            'run_traj = robot_motion_interface_ros.node_run_traj:main',
            'replay_target_pregrasp = robot_motion_interface_ros.replay_target_np:main',
            'replay_target_policy = robot_motion_interface_ros.replay_target_torch:main',
            'replay_action_torch = robot_motion_interface_ros.replay_action_torch:main'
        ],
    },
)
