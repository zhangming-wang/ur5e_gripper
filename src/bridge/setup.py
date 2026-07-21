import os
from setuptools import setup

package_name = "bridge"

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
            "bridge_node = bridge.bridge_node:main",
            "planning_node = bridge.planning_node:main",
        ],
    },
)
