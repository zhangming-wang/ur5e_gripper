from setuptools import setup

package_name = "act"

setup(
    name=package_name,
    version="1.0.0",
    packages=[package_name],
    data_files=[
        ("share/" + package_name, ["package.xml"]),
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    entry_points={
        "console_scripts": [
            "act_orchestrator = act.act_orchestrator_node:main",
            "act_ros_adapter = act.act_ros_adapter:main",
        ],
    },
)
