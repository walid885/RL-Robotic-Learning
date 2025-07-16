from setuptools import setup, find_packages

setup(
    name="robot-simulation",
    version="0.1.0",
    description="Modular robot simulation with PyBullet",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "pybullet>=3.2.0",
        "robot-descriptions>=1.0.0",
        "numpy>=1.21.0",
    ],
    python_requires=">=3.7",
)
