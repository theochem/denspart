#!/usr/bin/env python3
# DensPart performs Atoms-in-molecules density partitioning.
# Copyright (C) 2011-2020 The DensPart Development Team
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
    with open("README.rst") as fhandle:
        return fhandle.read()


def get_version_info():
    """Read __version__ and DEV_CLASSIFIER from version.py, using exec, not import."""
    try:
        with open(os.path.join("denspart", "version.py"), "r") as f:
            myglobals = {}
            exec(f.read(), myglobals)  # pylint: disable=exec-used
        return myglobals["__version__"], myglobals["DEV_CLASSIFIER"]
    except IOError:
        return "0.0.0.post0", "Development Status :: 2 - Pre-Alpha"


VERSION, DEV_CLASSIFIER = get_version_info()


setup(
    name="denspart",
    version=VERSION,
    description="Atoms-in-molecules density partitioning",
    long_description=get_readme(),
    author="HORTON-ChemTools Dev Team",
    author_email="horton.chemtools@gmail.com",
    url="https://github.com/theochem/denspart",
    package_dir={"denspart": "denspart"},
    packages=[
        "denspart",
        "denspart.adapters",
        "denspart.adapters.test",
        "denspart.test",
        "denspart.utils",
    ],
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "denspart-from-horton3 = denspart.adapters.horton3:main",
            "denspart-from-gpaw = denspart.adapters.gpaw:main",
            "denspart-from-adf = denspart.adapters.adf:main",
            "denspart = denspart.__main__:main",
            "denspart-write-extxyz = denspart.utils.write_extxyz:main",
        ]
    },
    classifiers=[
        DEV_CLASSIFIER,
        "Environment :: Console",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Intended Audience :: Science/Research",
    ],
    install_requires=["numpy>=1.0"],  # , "qc-grid"],
)
