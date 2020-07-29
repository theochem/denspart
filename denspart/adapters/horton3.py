"""Prepare inputs for denspart with HORTON3 modules.

This implementation makes some ad hoc coices on the molecular integration
grid, which should be revised in future. For now, this is something that is
just supposed to just work. The code is not tuned for precision nor for
efficiency.

This module is far from polished and is currently only used for prototyping.

"""


import numpy as np

from gbasis.wrappers import from_iodata
from gbasis.evals.density import evaluate_density

from grid.becke import BeckeWeights
from grid.molgrid import MolGrid
from grid.onedgrid import GaussChebyshev
from grid.rtransform import BeckeTF


__all__ = ["prepare_input"]


def prepare_input(iodata):
    """Prepare input for denspart with HORTON3 modules.

    XXX **WARNING** XXX: This function is far from final: the integration grid
    as not optimized yet!!

    XXX **WARNING** XXX: This function takes the spin-summed density, using the
    post-hf one if it is present. The spin-difference density is ignored.

    Parameters
    ----------
    iodata
        An instance with IOData containing the necessary data to compute the
        electron density on the grid.

    Returns
    -------
    grid
        A molecular integration grid.
    rho
        The electron density on the grid.

    """
    grid = _setup_grid(iodata.atnums, iodata.atcoords)
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


def _setup_grid(atnums, atcoords):
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
    oned = GaussChebyshev(150)
    rgrid = BeckeTF(1e-4, 1.5).transform_1d_grid(oned)
    grid = MolGrid.horton_molgrid(atcoords, atnums, rgrid, 194, becke)
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
    print(coord_types)
    rho = evaluate_density(one_rdm, basis, points, coord_type=coord_types)
    assert (rho >= 0).all()
    return rho
