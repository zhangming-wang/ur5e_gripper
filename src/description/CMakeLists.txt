cmake_minimum_required(VERSION 3.5)
project(description)

find_package(ament_cmake REQUIRED)

install(
  DIRECTORY config launch meshes rviz urdf
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)

  ament_add_pytest_test(ur5e_urdf_xacro test/test_ur5e_urdf_xacro.py)
  ament_add_pytest_test(view_ur5e_launch test/test_view_ur5e_launch.py)
endif()

ament_package()
