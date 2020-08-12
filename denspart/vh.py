"""Generic code for variational Hirshfeld methods.

This code is very preliminary, so no serious docstrings yet.
"""


from functools import partial

import numpy as np
from scipy.optimize import minimize


__all__ = ["optimize_pro_model", "BasisFunction", "ProModel", "ekld"]


RHO_CUTOFF = 1e-10


def optimize_pro_model(pro_model, grid, rho):
    # Precompute the local grids (should be optional)
    if True:
        print("Building local grids")
        localgrids = [
            grid.get_localgrid(fn.center, fn.get_cutoff_radius(fn.pars))
            for fn in pro_model.fns
        ]
    else:
        localgrids = None
    # Define initial guess and cost
    print("Optimization")
    print("        elkd          kld   constraint    grad.norm")
    print(" -----------  -----------  -----------  -----------")
    with np.errstate(all="raise"):
        # The errstate is changed to detect potential nasty numerical issues.
        pars0 = np.concatenate([fn.pars for fn in pro_model.fns])
        cost_grad = partial(
            ekld, grid=grid, rho=rho, pro_model=pro_model, localgrids=localgrids
        )
    # Optimize parameters within the bounds.
    bounds = sum([fn.bounds for fn in pro_model.fns], [])
    optresult = minimize(
        cost_grad,
        pars0,
        method="l-bfgs-b",
        jac=True,
        bounds=bounds,
        options={"gtol": 1e-8, "ftol": 1e-14},
    )
    print(" -----------  -----------  -----------  -----------")
    # Check for convergence.
    print('Optimizer message: "{}"'.format(optresult.message.decode("utf-8")))
    if not optresult.success:
        raise RuntimeError("Convergence failure.")
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

    def compute_population(self, pars):
        return pars[0]

    def compute_population_derivatives(self, pars):
        return np.array([1.0, 0.0])

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
    def natom(self):
        return len(self.atnums)

    @property
    def charges(self, pars=None):
        charges = np.array(self.atnums, dtype=float)
        ipar = 0
        for fn in self.fns:
            if pars is None:
                fnpars = fn.pars
            else:
                fnpars = pars[ipar : ipar + fn.npar]
            charges[fn.iatom] -= fn.compute_population(fnpars)
            ipar += fn.npar
        return charges

    def compute_population(self, pars=None):
        ipar = 0
        result = 0.0
        for ifn, fn in enumerate(self.fns):
            if pars is None:
                fnpars = fn.pars
            else:
                fnpars = pars[ipar : ipar + fn.npar]
            result += fn.compute_population(fnpars)
            ipar += fn.npar
        return result

    def compute_density(self, grid, pars=None, localgrids=None):
        # Compute pro-density
        pro = np.zeros_like(grid.weights)
        ipar = 0
        # print("  whole grid:", grid.size)
        for ifn, fn in enumerate(self.fns):
            if pars is None:
                fnpars = fn.pars
            else:
                fnpars = pars[ipar : ipar + fn.npar]
            if localgrids is None:
                localgrid = grid.get_localgrid(fn.center, fn.get_cutoff_radius(fnpars))
            else:
                localgrid = localgrids[ifn]
            # print("  localgrid:", localgrid.size)
            np.add.at(pro, localgrid.indices, fn.compute(localgrid.points, fnpars))
            ipar += fn.npar
        return pro

    def compute_proatom(self, iatom, grid, pars=None):
        # Compute pro-density
        pro = np.zeros_like(grid.weights)
        ipar = 0
        for ifn, fn in enumerate(self.fns):
            if pars is None:
                fnpars = fn.pars
            else:
                fnpars = pars[ipar : ipar + fn.npar]
            if fn.iatom == iatom:
                pro += fn.compute(grid.points, fnpars)
            ipar += fn.npar
        return pro


def ekld(pars, grid, rho, pro_model, localgrids):
    """Compute the Extended KL divergence and its gradient."""
    pro = pro_model.compute_density(grid, pars, localgrids)
    # compute potentially tricky quantities
    sick = (rho < RHO_CUTOFF) | (pro < RHO_CUTOFF)
    with np.errstate(all="ignore"):
        lnratio = np.log(rho) - np.log(pro)
        ratio = rho / pro
    lnratio[sick] = 0.0
    ratio[sick] = 0.0
    # Function value
    kld = np.einsum("i,i,i", grid.weights, rho, lnratio)
    propop = pro_model.compute_population(pars)
    # TODO: compute integral of rho only once
    constraint = np.einsum("i,i", grid.weights, rho) - propop
    ekld = kld - constraint
    # Gradient
    ipar = 0
    gradient = np.zeros_like(pars)
    for ifn, fn in enumerate(pro_model.fns):
        fnpars = pars[ipar : ipar + fn.npar]
        if localgrids is None:
            localgrid = grid.get_localgrid(fn.center, fn.get_cutoff_radius(fnpars))
        else:
            localgrid = localgrids[ifn]
        fn_derivatives = fn.compute_derivatives(localgrid.points, fnpars)
        gradient[ipar : ipar + fn.npar] = -np.einsum(
            "i,i,ji", localgrid.weights, ratio[localgrid.indices], fn_derivatives
        ) + fn.compute_population_derivatives(fnpars)
        ipar += fn.npar
    print(
        "{:12.7f} {:12.7f} {:12.7f} {:12.7f}".format(
            ekld, kld, -constraint, np.linalg.norm(gradient)
        )
    )
    return ekld, gradient
