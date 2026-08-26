cmake_minimum_required(VERSION 3.5)
project(mujoco)

find_package(ament_cmake REQUIRED)
find_package(mujoco_ros2_control_plugins REQUIRED)
find_package(mujoco_ros2_control_msgs REQUIRED)
find_package(mujoco_vendor REQUIRED)
find_package(pluginlib REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_srvs REQUIRED)
find_package(trajectory_msgs REQUIRED)

add_library(mujoco_scene_plugin SHARED src/mujoco_scene_plugin.cpp)
target_compile_features(mujoco_scene_plugin PUBLIC cxx_std_17)
ament_target_dependencies(
  mujoco_scene_plugin
  mujoco_ros2_control_plugins
  mujoco_ros2_control_msgs
  pluginlib
  rclcpp
  std_srvs
  trajectory_msgs
)
target_link_libraries(mujoco_scene_plugin mujoco::mujoco)

pluginlib_export_plugin_description_file(mujoco_ros2_control_plugins xml/mujoco_scene_plugin.xml)

install(DIRECTORY launch config urdf xml
  DESTINATION share/${PROJECT_NAME}
)

install(TARGETS mujoco_scene_plugin
  LIBRARY DESTINATION lib
)

ament_package()
