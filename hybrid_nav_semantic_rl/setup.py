from setuptools import find_packages, setup

package_name = "hybrid_nav_semantic_rl"

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
    description="Offline semantic RL decision layer for frontier ranking.",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "semantic_rl_decider_node = hybrid_nav_semantic_rl.semantic_rl_decider_node:main",
            "semantic_dataset_collector_node = hybrid_nav_semantic_rl.semantic_dataset_collector_node:main",
            "train_semantic_rl = hybrid_nav_semantic_rl.train_semantic_rl:main",
        ],
    },
)
