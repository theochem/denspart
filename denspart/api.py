"""End-user API for denspart."""


import numpy as np

from .mbis import partition_mbis


METHODS = {"MBIS": partition_mbis}


def setup_grid(iodata):
    from grid.becke import BeckeWeights
    from grid.molgrid import MolGrid
    from grid.onedgrid import GaussChebyshev
    from grid.rtransform import BeckeTF

    oned = GaussChebyshev(100)
    with np.errstate(all="ignore"):
        rgrid = BeckeTF(1e-4, 1.5).transform_1d_grid(oned)
        grid = MolGrid.horton_molgrid(
            iodata.atcoords, iodata.atnums, rgrid, 110, BeckeWeights()
        )
    assert np.isfinite(grid.points).all()
    assert np.isfinite(grid.weights).all()
    return grid


def compute_density(iodata, points):
    from gbasis.wrappers import from_iodata
    from gbasis.evals.density import evaluate_density

    basis = from_iodata(iodata)
    rho = evaluate_density(iodata.one_rdms["scf"], basis, points)
    assert (rho >= 0).all()
    return rho


def prune_grid(grid, rho):
    from grid.basegrid import Grid

    mask = rho > 0
    return Grid(grid.points[mask], grid.weights[mask]), rho[mask]


def partition(iodata, method_name, **method_options):
    print("Setting up grid")
    grid = setup_grid(iodata)
    print("Computing density")
    rho = compute_density(iodata, grid.points)
    print("Pruning zero-density grid points")
    grid, rho = prune_grid(grid, rho)
    assert (rho > 0).all()
    print("Partitioning")
    results = METHODS[method_name](
        iodata.atnums, iodata.atcoords, grid, rho, **method_options
    )
    iodata.atffparams["charges"] = results["charges"]
