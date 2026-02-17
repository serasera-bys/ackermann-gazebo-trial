from setuptools import find_packages, setup

package_name = "hybrid_nav_rl_planner"

setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/config", ["config/rl_policy.json"]),
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
            "dataset_collector_node = hybrid_nav_rl_planner.dataset_collector_node:main",
            "train_stub = hybrid_nav_rl_planner.train_stub:main",
        ],
    },
)
