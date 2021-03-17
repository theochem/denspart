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
"""Main command-line interface to denspart."""

import argparse

import numpy as np

from grid.basegrid import Grid
from grid.periodicgrid import PeriodicGrid

from denspart.mbis import partition
from denspart.properties import compute_rexp


__all__ = ["main"]


def main():
    """Partitioning command-line interface."""
    args = parse_args()
    data = np.load(args.in_npz)
    if data["cellvecs"].size == 0:
        grid = Grid(data["points"], data["weights"])
    else:
        print('Using Periodic Grid')
        grid = PeriodicGrid(data["points"], data["weights"], data["cellvecs"], wrap=True)
    rho = data["rho"]
    print("MBIS partitioning --")
    pro_model, localgrids = partition(
        data["atnums"],
        data["atcoords"],
        grid,
        rho,
        args.gtol,
        args.ftol,
        args.rho_cutoff,
    )
    print("Promodel")
    pro_model.pprint()
    print("Computing additional properties")
    results = pro_model.get_results()
    results.update(
        {
            "atnums": data["atnums"],
            "atcoords": data["atcoords"],
            "charges": pro_model.charges,
            "radial_moments": compute_rexp(pro_model, grid, rho, localgrids),
            "gtol": args.gtol,
            "ftol": args.ftol,
            "rho_cutoff": args.rho_cutoff,
        }
    )
    # TODO: make it easy to reconstruct a pro-model from the saved results.
    # This would at least require the class name of the pro-model to be stored.
    # Storing as a pickle would make this easy, but is insufficient for
    # long-term archival.
    np.savez(args.out_npz, **results)
    print('Sum of charges: ', sum(pro_model.charges))


def parse_args():
    """Parse command-line arguments."""
    description = "Density partitioning of a given density on a grid."
    parser = argparse.ArgumentParser(prog="denspart", description=description)
    parser.add_argument("in_npz", help="The NPZ file with grid and density.")
    parser.add_argument("out_npz", help="The NPZ file in which resutls will be stored.")
    parser.add_argument(
        "--gtol",
        type=float,
        default=1e-8,
        help="gtol convergence criterion for L-BFGS-B. [default=%(default)s]",
    )
    parser.add_argument(
        "--ftol",
        type=float,
        default=1e-12,
        help="ftol convergence criterion for L-BFGS-B. [default=%(default)s]",
    )
    parser.add_argument(
        "--rhocut",
        type=float,
        default=1e-10,
        dest="rho_cutoff",
        help="Cutoff density, used to estimate local grid sizes. "
        "Set to zero for while grid integrations (molecules only).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
