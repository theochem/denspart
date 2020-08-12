"""Generic code for variational Hirshfeld methods.

This code is very preliminary, so no serious docstrings yet.
"""


from functools import partial

import numpy as np
from scipy.optimize import minimize


__all__ = ["optimize_pro_model", "BasisFunction", "ProModel", "ekld"]


def optimize_pro_model(pro_model, grid, rho, gtol=1e-8, ftol=1e-14, rho_cutoff=1e-10):
    """Optimize the promodel using the L-BFGS-B minimizer from SciPy.

    Parameters
    ----------
    pro_model
        The model for the pro-molecular density, an instance of ``ProModel``.
        It contains the initial parameters as an attribute.
    grid
        The integration grid, an instance of ``grid.basegrid.Grid``.
    rho
        The electron density evaluated on the grid.
    gtol
        Convergence parameter gtol of SciPy's L-BFGS-B minimizer.
    ftol
        Convergence parameter ftol of SciPy's L-BFGS-B minimizer.
    rho_cutoff
        Density cutoff used to estimated sizes of local grids. Set to zero for
        whole-grid integrations. (This will not work for periodic systems.)

    Returns
    -------
    pro_model
        The model for the pro-molecular density, an instance of ``ProModel``.
        It contains the optimized parameters as an attribute.
    localgrids
        Local integration grids used for the pro-model basis functions.

    """
    # Precompute the local grids.
    print("Building local grids")
    localgrids = [
        grid.get_localgrid(fn.center, fn.get_cutoff_radius(fn.pars, rho_cutoff))
        for fn in pro_model.fns
    ]
    # Compute the total population
    pop = np.einsum("i,i", grid.weights, rho)
    print("Integral of rho:", pop)
    # Define initial guess and cost
    print("Optimization")
    print("        elkd          kld   constraint    grad.norm")
    print(" -----------  -----------  -----------  -----------")
    with np.errstate(all="raise"):
        # The errstate is changed to detect potentially nasty numerical issues.
        pars0 = np.concatenate([fn.pars for fn in pro_model.fns])
        cost_grad = partial(
            ekld,
            grid=grid,
            rho=rho,
            pro_model=pro_model,
            localgrids=localgrids,
            pop=pop,
        )
    # Optimize parameters within the bounds.
    bounds = sum([fn.bounds for fn in pro_model.fns], [])
    optresult = minimize(
        cost_grad,
        pars0,
        method="l-bfgs-b",
        jac=True,
        bounds=bounds,
        options={"gtol": gtol, "ftol": ftol},
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
    return pro_model, localgrids


class BasisFunction:
    """Base class for atom-centered basis functions for the pro-molecular density."""

    def __init__(self, iatom, center, pars, bounds):
        """Initialize a basis function.

        Parameters
        ----------
        iatom
            Index of the atom with which this function is associated.
        center
            The center of the function in Cartesian coordinates.
        pars
            The initial values of the proparameters for this function.
        bounds
            List of tuples with (lower, upper) bounds for each parameter. Use
            -np.inf and np.inf to disable bounds.

        """
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
        """Compute the total population of this basis function."""
        raise NotImplementedError

    def compute_population_derivatives(self, pars):
        """Compute the total derivatives of the population w.r.t. proparameters."""
        raise NotImplementedError

    def get_cutoff_radius(self, pars, rho_cutoff):
        """Estimate the cutoff radius for the given density cutoff."""
        raise NotImplementedError

    def compute(self, points, pars):
        """Compute the basisfunction values on a grid."""
        raise NotImplementedError

    def compute_derivatives(self, points, pars):
        """Compute derivatives of the basisfunction values on a grid."""
        raise NotImplementedError


class ProModel:
    def __init__(self, atnums, atcoords, fns):
        """Initialize the prodensity model.

        Parameters
        ----------
        atnums
            Atomic numbers
        atcoords
            Atomic coordinates
        fns
            A list of basis functions, instances of ``BasisFunction``.
        """
        self.atnums = atnums
        self.atcoords = atcoords
        self.fns = fns

    @property
    def natom(self):
        """The number of atoms."""
        return len(self.atnums)

    @property
    def charges(self):
        """Proatomic charges."""
        charges = np.array(self.atnums, dtype=float)
        for fn in self.fns:
            charges[fn.iatom] -= fn.compute_population(fn.pars)
        return charges

    @property
    def results(self):
        """A dictionary with additional results derived from the pro-parameters."""
        return {}

    def compute_population(self, pars=None):
        """Compute proatomic populations (for the given parameters)."""
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

    def compute_density(self, grid, localgrids, pars=None):
        """Compute prodensity on a grid (for the given parameters).

        Parameters
        ----------
        grid
            The whole integration grid, on which the results is computed.
        localgrids
            A list of local grids, one for each basis function.
        pars
            Proparameters. When not given, The basis functions owns parameters
            are used.

        Returns
        -------
        pro
            The prodensity on the points of ``grid``.

        """
        pro = np.zeros_like(grid.weights)
        ipar = 0
        for ifn, fn in enumerate(self.fns):
            if pars is None:
                fnpars = fn.pars
            else:
                fnpars = pars[ipar : ipar + fn.npar]
            localgrid = localgrids[ifn]
            np.add.at(pro, localgrid.indices, fn.compute(localgrid.points, fnpars))
            ipar += fn.npar
        return pro

    def compute_proatom(self, iatom, grid, pars=None):
        """Compute proatom density on a grid (for the given parameters).

        Parameters
        ----------
        iatom
            The atomic index.
        grid
            The whole integration grid, on which the results is computed.
        pars
            Proparameters (for the whole molecule). When not given, The basis
            functions owns parameters are used.

        Returns
        -------
        pro
            The prodensity on the points of ``grid``.

        """
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


def ekld(pars, grid, rho, pro_model, localgrids, pop, rho_cutoff=1e-15):
    """Compute the Extended KL divergence and its gradient.

    Parameters
    ----------
    pars
        A NumPy array with promodel parameters.
    grid
        A numerical integration grid with, instance of ``grid.basegrid.Grid``.
    rho
        The electron density evaluated on the grid.
    pro_model
        The model for the pro-molecular density, an instance of ``ProModel``.
    local_grids
        A list of local integration grids for the pro-model basis functions.
    pop
        The integral of rho, to be precomputed before calling this function.
    rho_cutoff
        Density cutoff used to neglect grid points with low densities. Including
        them can result in numerical noise in the result and its derivatives.

    Returns
    -------
    ekld
        The extended KL-d, i.e. including the Lagrange multiplier.
    gradient
        The gradient of ekld w.r.t. the pro-model parameters.

    """
    pro = pro_model.compute_density(grid, localgrids, pars)
    # Compute potentially tricky quantities.
    sick = (rho < rho_cutoff) | (pro < rho_cutoff)
    with np.errstate(all="ignore"):
        lnratio = np.log(rho) - np.log(pro)
        ratio = rho / pro
    lnratio[sick] = 0.0
    ratio[sick] = 0.0
    # Function value
    kld = np.einsum("i,i,i", grid.weights, rho, lnratio)
    constraint = pop - pro_model.compute_population(pars)
    ekld = kld - constraint
    # Gradient
    ipar = 0
    gradient = np.zeros_like(pars)
    for ifn, fn in enumerate(pro_model.fns):
        fnpars = pars[ipar : ipar + fn.npar]
        localgrid = localgrids[ifn]
        fn_derivatives = fn.compute_derivatives(localgrid.points, fnpars)
        gradient[ipar : ipar + fn.npar] = -np.einsum(
            "i,i,ji", localgrid.weights, ratio[localgrid.indices], fn_derivatives
        ) + fn.compute_population_derivatives(fnpars)
        ipar += fn.npar
    # Screen output
    print(
        "{:12.7f} {:12.7f} {:12.7f} {:12.7f}".format(
            ekld, kld, -constraint, np.linalg.norm(gradient)
        )
    )
    return ekld, gradient
