#!/usr/bin/env python3

import argparse

import numpy as np

from grid.basegrid import Grid

from denspart.mbis import partition
from denspart.properties import compute_rcubed


def main():
    """Partitioning command-line interface."""
    args = parse_args()
    data = np.load(args.fn_rho)
    if data["cellvecs"].size == 0:
        grid = Grid(data["points"], data["weights"])
    else:
        raise NotImplementedError
    rho = data["rho"]
    print("MBIS partitioning")
    pro_model, localgrids = partition(
        data["atnums"],
        data["atcoords"],
        grid,
        rho,
        args.gtol,
        args.ftol,
        args.rho_cutoff,
    )
    print("Properties")
    results = {
        "charges": pro_model.charges,
        "rcubed": compute_rcubed(pro_model, grid, rho, localgrids),
        "gtol": args.gtol,
        "ftol": args.ftol,
        "rho_cutoff": args.rho_cutoff,
    }
    results.update(pro_model.results)
    print("Charges:")
    print(results["charges"])
    np.savez(args.fn_results, **results)


def parse_args():
    """Parse command-line arguments."""
    DESCRIPTION = """\
    Density partitioning of a given density on a grid.
    """
    parser = argparse.ArgumentParser(prog="denspart", description=DESCRIPTION)
    parser.add_argument("fn_rho", help="The NPZ file with grid and density.")
    parser.add_argument(
        "fn_results", help="The NPZ file in which resutls will be stored."
    )
    parser.add_argument(
        "--gtol",
        type=float,
        default=1e-8,
        help="gtol convergence criterion for L-BFGS-B. [default=%(default)s]",
    )
    parser.add_argument(
        "--ftol",
        type=float,
        default=1e-14,
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
