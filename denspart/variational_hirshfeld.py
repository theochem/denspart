"""Generic code for variational Hirshfeld methods.

This code is very preliminary, so no serious docstrings yet.
"""


from functools import partial

import numpy as np
from scipy.optimize import minimize

from grid.basegrid import SubGrid


np.seterr(invalid="raise", divide="raise", over="raise")


__all__ = ["optimize_pro_model", "BasisFunction", "ProModel", "ekld"]


RHO_CUTOFF = 1e-10


def optimize_pro_model(pro_model, grid, rho):
    # Precompute the subgrids (should be optional)
    subgrids = [
        grid.get_subgrid(fn.center, fn.get_cutoff_radius(fn.pars))
        for fn in pro_model.fns
    ]
    # Define initial guess and cost
    pars0 = np.concatenate([fn.pars for fn in pro_model.fns])
    cost_grad = partial(
        ekld, grid=grid, rho=rho, pro_model=pro_model, subgrids=subgrids
    )
    # Optimize parameters within the bounds.
    bounds = sum([fn.bounds for fn in pro_model.fns], [])
    optresult = minimize(
        cost_grad, pars0, method="trust-constr", jac=True, bounds=bounds
    )
    # Assign the optimal parameters to the pro_model.
    pars1 = optresult.x
    ipar = 0
    for fn in pro_model.fns:
        fn.pars[:] = pars1[ipar : ipar + fn.npar]
        ipar += fn.npar
    return pro_model


class BasisFunction:
    def __init__(self, iatom, center, pars, bounds):
        if len(pars) != len(bounds):
            raise ValueError(
                "The number of parameters must equal the number of bounds."
            )
        self.iatom = iatom
        self.center = center
        self.pars = pars
        self.bounds = bounds

    @property
    def npar(self):
        """Number of parameters."""
        return len(self.pars)

    def get_population(self):
        raise NotImplementedError

    def _get_cutoff_radius(self, pars):
        raise NotImplementedError

    def compute(self, points, pars):
        raise NotImplementedError

    def compute_derivatives(self, points, pars):
        raise NotImplementedError


class ProModel:
    def __init__(self, atnums, atcoords, fns):
        self.atcoords = atcoords
        self.atnums = atnums
        self.fns = fns

    @property
    def charges(self):
        charges = np.array(self.atnums, dtype=float)
        ipar = 0
        for fn in self.fns:
            charges[fn.iatom] -= fn.get_population()
            ipar += fn.npar
        return charges


def ekld(pars, grid, rho, pro_model, subgrids):
    """Compute the Extended KL divergence and its gradient."""
    # Compute pro-density
    pro = np.zeros_like(grid.weights)
    ipar = 0
    # print("  whole grid:", grid.size)
    for ifn, fn in enumerate(pro_model.fns):
        fnpars = pars[ipar : ipar + fn.npar]
        if subgrids is None:
            subgrid = grid.get_subgrid(fn.center, fn.get_cutoff_radius(fnpars))
        else:
            subgrid = subgrids[ifn]
        # print("  subgrid:", subgrid.size)
        np.add.at(pro, subgrid.indices, fn.compute(subgrid.points, fnpars))
        ipar += fn.npar
    # compute potentially tricky quantities
    sick = (rho < RHO_CUTOFF) | (pro < RHO_CUTOFF)
    with np.errstate(all="ignore"):
        lnratio = np.log(rho) - np.log(pro)
        ratio = rho / pro
    lnratio[sick] = 0.0
    ratio[sick] = 0.0
    # Function value
    kld = np.einsum("i,i,i", grid.weights, rho, lnratio)
    constraint = np.einsum("i,i", grid.weights, rho - pro)
    ekld = kld - constraint
    # Gradient
    ipar = 0
    gradient = np.zeros_like(pars)
    for ifn, fn in enumerate(pro_model.fns):
        fnpars = pars[ipar : ipar + fn.npar]
        if subgrids is None:
            subgrid = grid.get_subgrid(fn.center, fn.get_cutoff_radius(fnpars))
        else:
            subgrid = subgrids[ifn]
        fn_derivatives = fn.compute_derivatives(subgrid.points, fnpars)
        gradient[ipar : ipar + fn.npar] = -np.einsum(
            "i,i,ji", subgrid.weights, ratio[subgrid.indices], fn_derivatives
        ) + np.einsum("i,ji", subgrid.weights, fn_derivatives)
        ipar += fn.npar
    return ekld, gradient
