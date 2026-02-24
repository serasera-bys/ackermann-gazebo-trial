from glob import glob
from setuptools import find_packages, setup

package_name = "hybrid_nav_autonomy_bringup"

setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", glob("launch/*.launch.py")),
        (f"share/{package_name}/config", glob("config/*.yaml")),
        (f"share/{package_name}/config/bt", glob("config/bt/*.xml")),
        (f"share/{package_name}/worlds", glob("worlds/*.sdf")),
        (f"share/{package_name}/rviz", glob("rviz/*.rviz")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="bernard",
    maintainer_email="you@example.com",
    description="Bringup package for semantic RL explorer stack.",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "semantic_run_metrics_node = hybrid_nav_autonomy_bringup.semantic_run_metrics_node:main",
            "scan_retimestamp_node = hybrid_nav_autonomy_bringup.scan_retimestamp_node:main",
        ],
    },
)
