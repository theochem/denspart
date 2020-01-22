#!/usr/bin/env python3
# DensPart performs Atoms-in-molecules density partitioning.
# Copyright (C) 2011-2019 The DensPart Development Team
#
# This file is part of DensPart.
#
# DensPart is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 3
# of the License, or (at your option) any later version.
#
# DensPart is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>
# --
"""Installation script for DensPart.

Directly calling this script is only needed by DensPart developers in special
circumstances. End users are recommended to install DensPart with pip or conda.
"""


import os

from setuptools import setup


def get_readme():
    """Load README.rst for display on PyPI."""
    with open("README.md") as fhandle:
        return fhandle.read()


setup(
    name="denspart",
    version="0.0.0",
    description="Atoms-in-molecules density partitioning",
    long_description=get_readme(),
    author="HORTON-ChemTools Dev Team",
    author_email="horton.chemtools@gmail.com",
    url="https://github.com/theochem/denspart",
    package_dir={"denspart": "denspart"},
    packages=["denspart", "denspart.test"],
    classifiers=[
        "Environment :: Console",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Intended Audience :: Science/Research",
    ],
    install_requires=["numpy>=1.0"],
)
