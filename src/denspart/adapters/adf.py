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
"""Convert ADF results input a density.npz file.

ADF calculations should have the following two settings:

  Symmetry NOSYM
  Save TAPE10

Example input file that works (tested with AMS 2021.202 revision 95049).


Task SinglePoint

System
  Atoms
    O  -0.00000000   0.00000000   0.46983780
    H   0.00000000   0.63395481  -0.23491890
    H   0.00000000  -0.63395481  -0.23491890
  end
end

Engine ADF
  Basis
     Type DZP
  End
  Symmetry NOSYM
  Save TAPE10
  XC
     GGA PBE
  End
EndEngine


"""


import argparse
import contextlib
import os
import tempfile

import numpy as np
from scm.plams import AMSJob, KFReader, finish, init


def main():
    """Run the main program."""
    args = parse_args()
    # extract data from ADF files
    data = extract_adf(args.dn_ams_results)
    # Write all data to a denspart H5 file
    write_output(args.fn_npz, data)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="denspart-from-adf",
        description="Extract molecular structure, integration grid, and density at grid "
        "points from ADF AMS job.",
    )
    parser.add_argument("dn_ams_results", help="The ams.results directory.")
    parser.add_argument("fn_npz", help="The DensPart density.npz file.")
    return parser.parse_args()


@contextlib.contextmanager
def plams_session():
    """Safely call init and finish when working with PLAMS."""
    with tempfile.TemporaryDirectory("plams_workdir", "denspart-from-adf") as path:
        init(path=path)
        yield
        finish()


def extract_adf(dn_ams_results):
    """Take the relevant arrays from TAPE files.

    This function uses the native Python KF reader to get the relevant data from the
    ams.rkf and TAPE10 files. For more info on PLAMS, see
    https://www.scm.com/documentation/Scripting/PLAMS/PLAMS/
    Thanks to Mirko Franchini and Erik van Lenthe from SCM for providing the
    necessary information.
    """
    result = {}
    with plams_session():
        # Use the AMSJob interface to check if the input was correct.
        job = AMSJob.load_external(dn_ams_results)
        job.check()
        if "adf" not in job.settings["input"]:
            raise OSError("Only ADF Jobs are supported at the moment.")
        if (
            "symmetry" not in job.settings["input"]["adf"]
            or job.settings["input"]["adf"]["symmetry"].lower() != "nosym"
        ):
            raise OSError("Symmetry must be set to NOSYM in the ADF job.")
        if (
            "save" not in job.settings["input"]["adf"]
            or "tape10" not in job.settings["input"]["adf"]["save"][0].lower()
        ):
            raise OSError("Tape 10 file must be saved.")

        # Read all other stuff directly from RKF files because PLAMS does not support
        # TAPE files and has the poor habit of converting away from atomic units. :(

        # Read the geometry from the ams.rkf file
        rkf = KFReader(os.path.join(dn_ams_results, "ams.rkf"))
        natom = rkf.read("Molecule", "nAtoms")
        result["atcoords"] = np.array(rkf.read("Molecule", "Coords")).reshape(-1, 3)
        assert result["atcoords"].shape == (natom, 3)
        result["atnums"] = np.array(rkf.read("Molecule", "AtomicNumbers"))
        assert result["atnums"].shape == (natom,)
        # Fitted all-electron density is put on grid, so no need to use effective core charges.
        result["atcorenums"] = result["atnums"][:]

        # Read the grid and density on grid from the TAPE10 file
        r10 = KFReader(os.path.join(dn_ams_results, "TAPE10"))

        block_sizes = np.array(r10.read("Points", "Length of Blocks"))
        grid_data = np.array(r10.read("Points", "Data"))
        scf_data = np.array(r10.read("SCF Data", "Data"))

        # Process the integration grid. Collect blocked data into contiguous array
        npoint = block_sizes.sum()
        result["points"] = np.zeros((npoint, 3))
        result["weights"] = np.zeros(npoint)
        ipoint = 0
        offset = 0
        for block_size in block_sizes:
            result["points"][ipoint : ipoint + block_size, 0] = grid_data[
                offset : offset + block_size
            ]
            offset += block_size
            result["points"][ipoint : ipoint + block_size, 1] = grid_data[
                offset : offset + block_size
            ]
            offset += block_size
            result["points"][ipoint : ipoint + block_size, 2] = grid_data[
                offset : offset + block_size
            ]
            offset += block_size
            result["weights"][ipoint : ipoint + block_size] = grid_data[
                offset : offset + block_size
            ]
            offset += block_size
            ipoint += block_size

        # process the density on the grid
        result["density"] = np.zeros(npoint)
        if r10.read("General", "nspin") == 1:
            # restricted case
            ipoint = 0
            offset = 0
            for block_size in block_sizes:
                result["density"][ipoint : ipoint + block_size] = scf_data[
                    offset : offset + block_size
                ]
                ipoint += block_size
                offset += 2 * block_size
        else:
            assert r10.read("General", "nspin") == 2
            # unrestricted case
            ipoint = 0
            offset = 0
            for block_size in block_sizes:
                result["density"][ipoint : ipoint + block_size] = scf_data[
                    offset : offset + block_size
                ]
                offset += block_size
                result["density"][ipoint : ipoint + block_size] += scf_data[
                    offset : offset + block_size
                ]
                ipoint += block_size
                offset += 2 * block_size

    return result


def write_output(fn_npz, data):
    """Write the density.npz file."""
    # Only store the most relevant data.
    debye = 0.3934303
    for i, char in enumerate("xyz"):
        dipole_moment = np.dot(data["atnums"], data["atcoords"][:, i]) - np.dot(
            data["density"] * data["points"][:, i], data["weights"]
        )
        print(f"Dipole moment {char} [a.u.]:  {dipole_moment:10.3f}")
        print(f"Dipole moment {char} [Debye]: {dipole_moment/debye:10.3f}")
    nelec = np.dot(data["density"], data["weights"])
    charge = data["atnums"].sum() - nelec
    print(f"Number of electrons:     {nelec:10.3f}")
    print(f"Total charge:            {charge:10.3e}")

    mask = data["weights"] != 0.0
    np.savez_compressed(
        fn_npz,
        atnums=data["atnums"],
        atcorenums=data["atcorenums"],
        atcoords=data["atcoords"],
        density=data["density"][mask],
        points=data["points"][mask],
        weights=data["weights"][mask],
    )


if __name__ == "__main__":
    main()
