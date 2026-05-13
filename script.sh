# CAMERA
# 1. realsense calibration through artag
python -m robot_motion_interface.utils.realsense_artag_cali \
  --marker-size 0.1 \
  --target-tag-id 0 \
  --aruco-dict DICT_4X4_50

# 2. cycle gan real world data collection
python -m robot_motion_interface.utils.realsense_real_img real_run
python -m robot_motion_interface.utils.realsense_real_record real_stream_real_jar



# ROBOT
# 1. left hand pregrasp, right hand home
ros2 topic pub --once --wait-matching-subscriptions 1 \
    --qos-reliability reliable --qos-durability volatile --qos-history keep_last --qos-depth 1 \
    /target_joint_states sensor_msgs/msg/JointState \
    "{position: [-1.10, 0.90, 0.10, -1.60, 1.0, 1.3, 0.1,
                 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                 0.0, -0.7854, 0.0, -2.3562, 0.0, 1.5708, 0.7854,
                 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}"

# 2. left hand pregrasp, right hand pregrasp
ros2 topic pub --once --wait-matching-subscriptions 1 \
    --qos-reliability reliable --qos-durability volatile --qos-history keep_last --qos-depth 1 \
    /target_joint_states sensor_msgs/msg/JointState \
    "{position: [-1.10, 0.90, 0.10, -1.60, 1.0, 1.3, 0.1,
                 0.0, 0.0, 0.60, 0.20, 0.0, 0.0, 0.60, 0.20, 0.0, 0.0, 0.60, 0.20,
                 0.1, 0.15, 0.0, -1.6, 0.0, 1.6, 0.75,
                 0.0, 0.0, 0.60, 0.20, 0.0, 0.0, 0.60, 0.20, 0.0, 0.0, 0.60, 0.20]}"
