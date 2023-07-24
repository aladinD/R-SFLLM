#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="isac_pl",
    version="0.0.1",
    description="Deep Learning template for Integrated Sensing and Communications Research",
    author="Vlad Andrei",
    author_email="vlad.andrei@tum.de",
    url="https://github.com/user/project",  # REPLACE WITH YOUR OWN GITHUB PROJECT LINK
    install_requires=["pytorch-lightning", "hydra-core"],
    packages=find_packages(),
)
