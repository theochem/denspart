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

from denspart.mbis import MBISProModel
from denspart.vh import optimize_reduce_pro_model
from denspart.properties import compute_radial_moments, compute_multipole_moments


__all__ = ["main"]


def main(args=None):
    """Partitioning command-line interface."""
    args = parse_args(args)
    data = np.load(args.in_npz)
    if "cellvecs" not in data or data["cellvecs"].size == 0:
        grid = Grid(data["points"], data["weights"])
    else:
        print("Using periodic grid")
        grid = PeriodicGrid(
            data["points"], data["weights"], data["cellvecs"], wrap=True
        )
    density = data["density"]
    print("MBIS partitioning --")
    pro_model_init = MBISProModel.from_geometry(data["atnums"], data["atcoords"])
    pro_model, localgrids = optimize_reduce_pro_model(
        pro_model_init,
        grid,
        density,
        args.gtol,
        args.maxiter,
        args.density_cutoff,
    )
    print("Promodel")
    pro_model.pprint()
    print("Computing additional properties")
    results = pro_model.to_dict()
    results.update(
        {
            "charges": pro_model.charges,
            "radial_moments": compute_radial_moments(
                pro_model, grid, density, localgrids
            ),
            "multipole_moments": compute_multipole_moments(
                pro_model, grid, density, localgrids
            ),
            "gtol": args.gtol,
            "maxiter": args.maxiter,
            "density_cutoff": args.density_cutoff,
        }
    )
    np.savez(args.out_npz, **results)
    print("Sum of charges: ", sum(pro_model.charges))


def parse_args(args=None):
    """Parse command-line arguments."""
    description = "Density partitioning of a given density on a grid."
    parser = argparse.ArgumentParser(prog="denspart", description=description)
    parser.add_argument("in_npz", help="The NPZ file with grid and density.")
    parser.add_argument("out_npz", help="The NPZ file in which resutls will be stored.")
    parser.add_argument(
        "--gtol",
        type=float,
        default=1e-8,
        help="gtol convergence criterion for SciPy's trust-constr minimizer. "
        "[default=%(default)s]",
    )
    parser.add_argument(
        "-m",
        "--maxiter",
        type=int,
        default=1000,
        help="Maximum number of iterations in SciPy's trust-constr minimizer. "
        "[default=%(default)s]",
    )
    parser.add_argument(
        "-c",
        "--density-cutoff",
        type=float,
        default=1e-10,
        help="Density cutoff, used to estimate local grid sizes. "
        "Set to zero for while grid integrations (molecules only). "
        "[default=%(default)s]",
    )
    return parser.parse_args(args)


if __name__ == "__main__":
    main()
