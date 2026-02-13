from setuptools import find_packages, setup

package_name = "hybrid_nav_rl_planner"

setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="bernard",
    maintainer_email="you@example.com",
    description="Placeholder RL planner with the same interface as rule planner.",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "rl_planner_node = hybrid_nav_rl_planner.rl_planner_node:main",
        ],
    },
)

