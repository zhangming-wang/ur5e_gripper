cmake_minimum_required(VERSION 3.5)
project(mujoco)

find_package(ament_cmake REQUIRED)

install(DIRECTORY launch config urdf xml
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
