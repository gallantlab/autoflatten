#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import find_packages, setup

setup(
    name="autoflatten",
    version="0.1.0",
    description="Automatic Surface Flattening Pipeline",
    author="Mateo Visconti di Oleggio Castello",
    author_email="mvdoc@berkeley.edu",
    url="",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "numpy",
        "networkx",
        "pycortex",
    ],
    entry_points={
        "console_scripts": [
            "autoflatten=autoflatten.cli:main",
        ],
    },
    python_requires=">=3.7",
)
