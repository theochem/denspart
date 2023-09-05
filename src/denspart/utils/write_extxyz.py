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
"""Write denspart results into a nice extended XYZ."""


import argparse

import numpy as np
from iodata import IOData, dump_one
from iodata.formats.xyz import DEFAULT_ATOM_COLUMNS
from iodata.utils import angstrom

__all__ = ["main"]

# pylint: disable=unnecessary-lambda
ATOM_COLUMNS = [
    *DEFAULT_ATOM_COLUMNS,
    ("atffparams", "charges", (), float, lambda word: float(word), lambda value: f"{value:15.10f}"),
    (
        "atffparams",
        "rcubed",
        (),
        float,
        lambda word: float(word) * angstrom**3,
        lambda value: f"{value / angstrom**3:15.10f}",
    ),
    (
        "atffparams",
        "valence_charges",
        (),
        float,
        lambda word: float(word),
        lambda value: f"{value:15.10f}",
    ),
    (
        "atffparams",
        "core_charges",
        (),
        float,
        lambda word: float(word),
        lambda value: f"{value:15.10f}",
    ),
    (
        "atffparams",
        "valence_widths",
        (),
        float,
        lambda word: float(word * angstrom),
        lambda value: f"{value / angstrom:15.10f}",
    ),
]


def main():
    """Convert results to extended XYZ, main script."""
    args = parse_args()
    results = np.load(args.out_npz)
    iodata = IOData(
        title="Properties=species:S:1:pos:R:3:charges:R:1:rcubed:R:1:"
        "valence_charges:R:1:core_charges:R:1:valence_widths:R:1",
        atnums=results["atnums"],
        atcoords=results["atcoords"],
        atffparams={
            "charges": results["charges"],
            "rcubed": results["radial_moments"][:, 3],
            "valence_charges": results["valence_charges"],
            "core_charges": results["core_charges"],
            "valence_widths": results["valence_widths"],
        },
    )
    dump_one(iodata, args.out_xyz, fmt="xyz", atom_columns=ATOM_COLUMNS)


def parse_args():
    """Parse command-line arguments."""
    description = "Write denspart results as extended XYZ file."
    parser = argparse.ArgumentParser(prog="denspart-write-extxyz", description=description)
    parser.add_argument("out_npz", help="The results from the main denspart script.")
    parser.add_argument("out_xyz", help="The extended XZY file.")
    return parser.parse_args()


if __name__ == "__main__":
    main()
