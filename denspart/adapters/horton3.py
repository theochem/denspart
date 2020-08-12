"""Prepare inputs for denspart with HORTON3 modules.

This implementation makes some ad hoc coices on the molecular integration
grid, which should be revised in future. For now, this is something that is
just supposed to just work. The code is not tuned for precision nor for
efficiency.

This module is far from polished and is currently only used for prototyping:

- The integration grid is not final. It may have precision issues and it is not
  pruned. Furthermore, all atoms are given the same atomic grid, which is far
  from optimal and the Becke partitioning is used, which can be improved upon.

- Only the spin-summed density is computed, using the post-hf 1RDM if it is
  present. The spin-difference density is ignored.

"""


import argparse

import numpy as np

from iodata import load_one

from gbasis.wrappers import from_iodata
from gbasis.evals.density import evaluate_density

from grid.becke import BeckeWeights
from grid.molgrid import MolGrid
from grid.onedgrid import GaussChebyshev
from grid.rtransform import BeckeTF


__all__ = ["prepare_input"]


def prepare_input(iodata, nrad, nang):
    """Prepare input for denspart with HORTON3 modules.

    Parameters
    ----------
    iodata
        An instance with IOData containing the necessary data to compute the
        electron density on the grid.
    nrad
        Number of radial grid points.
    nang
        Number of angular grid points.

    Returns
    -------
    grid
        A molecular integration grid.
    rho
        The electron density on the grid.

    """
    grid = _setup_grid(iodata.atnums, iodata.atcoords, nrad, nang)
    one_rdm = iodata.one_rdms.get("post_scf", iodata.one_rdms.get("scf"))
    if one_rdm is None:
        if iodata.mo is None:
            raise ValueError(
                "The input file lacks wavefunction data with which "
                "the density can be computed."
            )
        coeffs, occs = iodata.mo.coeffs, iodata.mo.occs
        one_rdm = np.dot(coeffs * occs, coeffs.T)
    rho = _compute_density(iodata, one_rdm, grid.points)
    return grid, rho


def _setup_grid(atnums, atcoords, nrad, nang):
    """Set up a simple molecular integration grid for a given molecular geometry.

    Parameters
    ----------
    atnums: np.ndarray(N,)
        Atomic numbers
    atcoords: np.ndarray(N, 3)
        Atomic coordinates.

    Returns
    -------
    grid
        A molecular integration grid, instance (of a subclass of)
        grid.basegrid.Grid.

    """
    becke = BeckeWeights(order=3)
    # Fix for missing radii.
    becke._radii[2] = 0.5
    becke._radii[10] = 1.0
    becke._radii[18] = 2.0
    becke._radii[36] = 2.5
    becke._radii[54] = 3.5
    oned = GaussChebyshev(nrad)
    rgrid = BeckeTF(1e-4, 1.5).transform_1d_grid(oned)
    grid = MolGrid.horton_molgrid(atcoords, atnums, rgrid, nang, becke)
    assert np.isfinite(grid.points).all()
    assert np.isfinite(grid.weights).all()
    assert (grid.weights >= 0).all()
    return grid


def _compute_density(iodata, one_rdm, points):
    """Evaluate the density on a give set of grid points.

    Parameters
    ----------
    iodata: IOData
        An instance of IOData, containing an atomic orbital basis set.
    one_rdm: np.ndarray(nbasis, nbasis)
        The one-particle reduced density matrix in the atomic orbital basis.
    points: np.ndarray(N, 3)
        A set of grid points.

    Returns
    -------
    rho
        The electron density on the grid points.

    """
    basis, coord_types = from_iodata(iodata)
    rho = evaluate_density(one_rdm, basis, points, coord_type=coord_types)
    assert (rho >= 0).all()
    return rho


def main():
    """Command-line interface."""
    args = parse_args()
    iodata = load_one(args.fn_wfn)
    grid, rho = prepare_input(iodata, args.nrad, args.nang)
    np.savez(
        args.fn_rho,
        **{
            "atcoords": iodata.atcoords,
            "atnums": iodata.atnums,
            "atcorenums": iodata.atcorenums,
            "points": grid.points,
            "weights": grid.weights,
            "rho": rho,
            "cellvecs": np.zeros((0, 3)),
        }
    )


def parse_args():
    """Parse command-line arguments."""
    DESCRIPTION = """\
    Setup a default integration grid and compute the density with HORTON3.
    """
    parser = argparse.ArgumentParser(
        prog="denspart-rho-horton3", description=DESCRIPTION
    )
    parser.add_argument("fn_wfn", help="The wavefunction file.")
    parser.add_argument(
        "fn_rho",
        help="The NPZ file in which the grid and the " "density will be stored.",
    )
    parser.add_argument(
        "-r",
        "--nrad",
        type=int,
        default=150,
        help="Number of radial grid points. [default=%(default)s]",
    )
    parser.add_argument(
        "-a",
        "--nang",
        type=int,
        default=194,
        help="Number of angular grid points. [default=%(default)s]",
    )
    return parser.parse_args()
