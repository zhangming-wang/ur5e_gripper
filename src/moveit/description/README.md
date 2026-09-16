# ur5e_description

ROS2 package containing URDF description and meshes for the **Universal Robots UR5e** manipulator.

This package was extracted from the [`ur_description`](https://github.com/UniversalRobots/Universal_Robots_ROS2_Description) package and simplified to support only the UR5e model.

## Usage

To visualize the UR5e robot model, install this package in your ROS2 workspace and run:

```bash
ros2 launch ur5e_description view_ur5e.launch.xml
```

You can also visualize it with a custom tf prefix:

```bash
ros2 launch ur5e_description view_ur5e.launch.xml tf_prefix:=my_ur_
```

## Package Structure

```
ur5e_description/
├── config/ur5e/         # UR5e kinematics, joint limits, physics, visual parameters
├── meshes/ur5e/         # UR5e collision (.stl) and visual (.dae) meshes
├── urdf/                # XACRO description files
├── launch/              # Launch files for visualization
├── rviz/                # RViz configuration
└── test/                # Pytest-based tests
```

## License

All code is licensed under the BSD-3-Clause license.
