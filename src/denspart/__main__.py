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

from .cache import ComputeCache
from .mbis import MBISProModel
from .properties import compute_multipole_moments, compute_radial_moments
from .vh import optimize_reduce_pro_model

__all__ = ["main"]


def main(args=None):
    """Partitioning command-line interface."""
    args = parse_args(args)
    nshell_map = parse_nshell_arg(args.nshell)
    data = np.load(args.in_npz)
    if "cellvecs" not in data or data["cellvecs"].size == 0:
        grid = Grid(data["points"], data["weights"])
    else:
        print("Using periodic grid")
        grid = PeriodicGrid(data["points"], data["weights"], data["cellvecs"], wrap=True)
    density = data["density"]
    print("MBIS partitioning --")
    pro_model_init = MBISProModel.from_geometry(data["atnums"], data["atcoords"], nshell_map)
    cache = ComputeCache() if args.do_cache else None
    pro_model, localgrids = optimize_reduce_pro_model(
        pro_model_init,
        grid,
        density,
        args.gtol,
        args.maxiter,
        args.density_cutoff,
        cache,
    )
    print("Promodel")
    pro_model.pprint()
    print("Computing additional properties")
    results = pro_model.to_dict()
    results.update(
        {
            "charges": pro_model.charges,
            "radial_moments": compute_radial_moments(pro_model, grid, density, localgrids, cache),
            "multipole_moments": compute_multipole_moments(
                pro_model, grid, density, localgrids, cache
            ),
            "gtol": args.gtol,
            "maxiter": args.maxiter,
            "density_cutoff": args.density_cutoff,
        }
    )
    np.savez_compressed(args.out_npz, **results)
    print("Sum of charges: ", sum(pro_model.charges))


def parse_nshell_arg(nshell):
    """Convert a list of nshell command-line arguments into a more convenient dictionary."""
    nshell_map = {}
    for word in nshell:
        if word.count(":") != 1:
            raise ValueError("Each nshell specification should have at least one colon.")
        atnum, num = word.split(":")
        nshell_map[int(atnum)] = int(num)
    return nshell_map


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
        "Set to zero for whole-grid integrations (molecules only). "
        "[default=%(default)s]",
    )
    parser.add_argument(
        "--nshell",
        default=[],
        nargs="+",
        help="A whitespace-separate list of atnum:num items, e.g. 12:2 "
        "mean two shells for magnesium. "
        "The num part must be a positive integer and cannot exceed the "
        "default number of shells specified for that element. "
        "At least one argument must be given.",
    )
    parser.add_argument(
        "--nocache",
        dest="do_cache",
        default=True,
        action="store_false",
        help="Disable caching. The cache increases memory consumption, "
        "it speeds up the calculation by about a factor of 2, "
        "and it could introduce more bugs.",
    )
    return parser.parse_args(args)


if __name__ == "__main__":
    main()
