# Copyright 2022 PIERRE LASSALLE
# All rights reserved

from setuptools import setup, find_packages

with open("requirements.txt") as fp:
    install_requires = fp.read().strip().split("\n")

setup(
    install_requires=install_requires,
    packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
)