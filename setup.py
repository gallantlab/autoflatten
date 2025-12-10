#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import find_packages, setup

setup(
    name="autoflatten",
    version="0.2.0",
    description="Automatic Surface Flattening Pipeline",
    author="Matteo Visconti di Oleggio Castello",
    author_email="mvdoc@berkeley.edu",
    url="",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        # Core dependencies (autoflatten projection)
        "numpy",
        "networkx",
        "pycortex",
        "scikit-learn",
        # pyflatten dependencies (flattening algorithm)
        "jax>=0.4.0",
        "libigl>=2.5.0",
        "nibabel>=5.0.0",
        "numba>=0.58.0",
        "scipy>=1.10.0",
        "tqdm>=4.65.0",
        # Visualization
        "matplotlib>=3.7.0",
    ],
    extras_require={
        "test": [
            "pytest",
            "pytest-cov",
            "codecov",
        ],
        "cuda": [
            "jax[cuda12]>=0.4.0",
        ],
        "viz3d": [
            "polyscope>=2.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "autoflatten=autoflatten.cli:main",
        ],
    },
    python_requires=">=3.10",
)
