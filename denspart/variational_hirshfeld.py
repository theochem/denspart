"""Generic code for variational Hirshfeld methods.

This code is very preliminary, so no serious docstrings yet.
"""


from functools import partial

import numpy as np
from scipy.optimize import minimize

from grid.basegrid import SubGrid


np.seterr(invalid="raise", divide="raise", over="raise")


__all__ = ["partition", "BasisFunction", "ekld"]


RHO_CUTOFF = 1e-10


def partition(basis, grid, rho, atnums):
    pars0 = np.concatenate([fn.pars0 for fn in basis])
    bounds = sum([fn.bounds for fn in basis], [])
    cost_grad = partial(ekld, grid=grid, rho=rho, basis=basis)
    optresult = minimize(
        cost_grad, pars0, method="trust-constr", jac=True, bounds=bounds
    )
    pars1 = optresult.x
    # The following is seriously ugly. it will be removed
    charges = np.array(atnums, dtype=float)
    ipar = 0
    for fn in basis:
        charges[fn.iatom] -= pars1[ipar]
        ipar += fn.npar
    return {"charges": charges}


class BasisFunction:
    def __init__(self, iatom, center):
        self.iatom = iatom
        self.center = center

    def get_radius(self, pars):
        raise NotImplementedError

    def compute(self, pars, points):
        raise NotImplementedError

    def compute_derivatives(self, pars, points):
        raise NotImplementedError


def _compute_pro(pars, grid, basis):
    pro = np.zeros_like(grid.weights)
    ipar = 0
    print("  whole grid:", grid.size)
    for fn in basis:
        fnpars = pars[ipar : ipar + fn.npar]
        subgrid = grid.get_subgrid(fn.center, fn.get_radius(fnpars))
        print("  subgrid:", subgrid.size)
        np.add.at(pro, subgrid.indices, fn.compute(fnpars, subgrid.points))
        ipar += fn.npar
    return pro


def ekld(pars, grid, rho, basis):
    """Compute the Extended KL divergence and its gradient."""
    pro = _compute_pro(pars, grid, basis)
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
    for fn in basis:
        fnpars = pars[ipar : ipar + fn.npar]
        subgrid = grid.get_subgrid(fn.center, fn.get_radius(fnpars))
        basis_derivatives = fn.compute_derivatives(fnpars, subgrid.points)
        gradient[ipar : ipar + fn.npar] = -np.einsum(
            "i,i,ji", subgrid.weights, ratio[subgrid.indices], basis_derivatives
        ) + np.einsum("i,ji", subgrid.weights, basis_derivatives)
        ipar += fn.npar
    return ekld, gradient
