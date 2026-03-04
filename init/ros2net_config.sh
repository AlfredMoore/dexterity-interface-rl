# server
fastdds discovery --server-id 0

export ROS_DISCOVERY_SERVER=192.168.4.9:11811
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DOMAIN_ID=22
export ROS_LOCALHOST_ONLY=0

# client
export ROS_DISCOVERY_SERVER=192.168.4.9:11811
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DOMAIN_ID=22
export ROS_LOCALHOST_ONLY=0
