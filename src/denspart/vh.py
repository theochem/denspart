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
"""Generic code for variational Hirshfeld methods.

This code is very preliminary, so no serious docstrings yet.
"""


import time
from functools import partial

import numpy as np
from scipy.optimize import SR1, minimize

__all__ = ["optimize_reduce_pro_model", "BasisFunction", "ProModel", "ekld"]


def optimize_reduce_pro_model(
    pro_model,
    grid,
    density,
    gtol=1e-8,
    maxiter=1000,
    density_cutoff=1e-10,
    cache=None,
):
    """Optimize the pro-model and removed redundant basis functions.

    Parameters
    ----------
    See optimize_pro_model for details.

    """
    while True:
        pro_model, localgrids = optimize_pro_model(
            pro_model, grid, density, gtol, maxiter, density_cutoff, cache
        )
        reduced_pro_model = pro_model.reduce()
        if len(pro_model.fns) == len(reduced_pro_model.fns):
            break
        print("Restarting optimization with reduced pro-model.")
        pro_model = reduced_pro_model
    return pro_model, localgrids


def optimize_pro_model(
    pro_model,
    grid,
    density,
    gtol=1e-8,
    maxiter=1000,
    density_cutoff=1e-10,
    cache=None,
):
    """Optimize the promodel using the trust-constr minimizer from SciPy.

    Parameters
    ----------
    pro_model
        The model for the pro-molecular density, an instance of ``ProModel``.
        It contains the initial parameters as an attribute.
    grid
        The integration grid, an instance of ``grid.basegrid.Grid``.
    density
        The electron density evaluated on the grid.
    gtol
        Convergence parameter gtol of SciPy's trust-constr minimizer.
    maxiter
        Maximum number of iterations in SciPy's trust-constr minimizer.
    density_cutoff
        Density cutoff used to estimated sizes of local grids. Set to zero for
        whole-grid integrations. (This will not work for periodic systems.)
    cache
        An optional ComputeCache instance for reusing intermediate results.

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
        grid.get_localgrid(fn.center, fn.get_cutoff_radius(density_cutoff)) for fn in pro_model.fns
    ]
    # Compute the total population
    pop = np.einsum("i,i", grid.weights, density)
    print("Integral of density:", pop)
    # Define initial guess and cost
    print("Optimization")
    print("#Iter  #Call         ekld          kld  -constraint     grad.rms  cputime (s)")
    print("-----  -----  -----------  -----------  -----------  -----------  -----------")
    pars0 = np.concatenate([fn.pars for fn in pro_model.fns])
    cost_grad = partial(
        ekld,
        grid=grid,
        density=density,
        pro_model=pro_model,
        localgrids=localgrids,
        pop=pop,
        cache=cache,
    )
    pro_model.ekld_info = None

    def callback(_current_pars, opt_result):
        info = pro_model.ekld_info
        # if info is None:
        #    return
        gradient = info["gradient"]
        print(
            "{:5d} {:6d} {:12.7f} {:12.7f} {:12.4e} {:12.4e} {:12.7f}".format(
                opt_result.nit,
                opt_result.njev,
                info["ekld"],
                info["kld"],
                -info["constraint"],
                # TODO: projected gradient may be better.
                np.linalg.norm(gradient) / np.sqrt(len(gradient)),
                info["time"],
            )
        )

    with np.errstate(all="raise"):
        # The errstate is changed to detect potentially nasty numerical issues.
        # Optimize parameters within the bounds.
        bounds = sum([fn.bounds for fn in pro_model.fns], [])

        optresult = minimize(
            cost_grad,
            pars0,
            method="trust-constr",
            jac=True,
            hess=SR1(),
            bounds=bounds,
            callback=callback,
            options={"gtol": gtol, "maxiter": maxiter},
        )

    print("-----  -----------  -----------  -----------  -----------  -----------")
    # Check for convergence.
    print(f'Optimizer message: "{optresult.message}"')
    if not optresult.success:
        raise RuntimeError("Convergence failure.")
    # Wrap up
    print(f"Total charge:       {pro_model.atnums.sum() - pop:20.7e}")
    print(f"Sum atomic charges: {pro_model.charges.sum():20.7e}")
    pro_model.assign_pars(optresult.x)
    return pro_model, localgrids


class BasisFunction:
    """Base class for atom-centered basis functions for the pro-molecular density.

    Each basis function instance stores also its parameters in ``self.pars``,
    which are always kept up-to-date. This simplifies the code a lot because
    the methods below can easily access the ``self.pars`` attribute when they
    need it, instead of having to rely on the caller to pass them in correctly.
    This is in fact a typical antipattern, but here it works well.
    """

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
            List of tuples with ``(lower, upper)`` bounds for each parameter.
            Use ``-np.inf`` and ``np.inf`` to disable bounds.

        """
        if len(pars) != len(bounds):
            raise ValueError("The number of parameters must equal the number of bounds.")
        self.iatom = iatom
        self.center = center
        self.pars = pars
        self.bounds = bounds

    @property
    def npar(self):
        """Number of parameters."""
        return len(self.pars)

    @property
    def population(self):
        """Population of this basis function."""
        raise NotImplementedError

    @property
    def population_derivatives(self):
        """Derivatives of the population w.r.t. proparameters."""
        raise NotImplementedError

    def get_cutoff_radius(self, density_cutoff):
        """Estimate the cutoff radius for the given density cutoff."""
        raise NotImplementedError

    def compute(self, points):
        """Compute the basisfunction values on a grid."""
        raise NotImplementedError

    def compute_derivatives(self, points):
        """Compute derivatives of the basisfunction values on a grid."""
        raise NotImplementedError


class ProModelMeta(type):
    """Meta class for ProModel classes.

    This meta class registers all subclasses, making it easy to recreate a ProModel
    instance from the data stored in an NPZ file. Note that Python pickle files are
    not used for storing result because these are not suitable for long-term data
    preservation.

    """

    registry = {}

    def __new__(mcs, name, bases, namespace, **kwargs):
        result = super().__new__(mcs, name, bases, namespace, **kwargs)
        ProModelMeta.registry[name] = result
        return result


class ProModel(metaclass=ProModelMeta):
    """Base class for the promolecular density."""

    registry = {}

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
        """Number of atoms."""
        return len(self.atnums)

    @property
    def charges(self):
        """Proatomic charges."""
        charges = np.array(self.atnums, dtype=float)
        for fn in self.fns:
            charges[fn.iatom] -= fn.population
        return charges

    @classmethod
    def from_geometry(cls, atnums, atcoords):
        """Derive a ProModel with a sensible initial guess from a molecular geometry."""
        raise NotImplementedError

    def reduce(self, eps=1e-4):
        """Return a new ProModel in which redundant functions are merged together.

        Parameters
        ----------
        eps
            When the population of a basis function is lower then eps, it is removed.

        """
        new_fns = [fn for fn in self.fns if fn.population > eps]
        return self.__class__(self.atnums, self.atcoords, new_fns)

    def to_dict(self):
        """Return a dictionary representation of the pro-model, with with additional.

        Notes
        -----
        The primary purpose is to include sufficient information in the returned result
        to reconstruct this instance from the dictionary.

        It is recommended that subclasses try to include additional information that may
        be convenient for end users.

        All values in the dictionary must be np.ndarray instances.

        """
        # Number of functions per atom
        atnfns = np.zeros(self.natom, dtype=int)
        # Number of parameters per atom
        atnpars = np.zeros(self.natom, dtype=int)
        for fn in self.fns:
            atnfns[fn.iatom] += 1
            atnpars[fn.iatom] += len(fn.pars)
        return {
            "class": np.array(self.__class__.__name__),
            "atnums": self.atnums,
            "atcoords": self.atcoords,
            "atnfns": atnfns,
            "atnpars": atnpars,
            "propars": np.concatenate([fn.pars for fn in self.fns]),
        }

    @classmethod
    def from_dict(cls, data):
        """Create an instance of a ProModel subclass from a dictionary made with to_dict."""
        subcls = ProModelMeta.registry[str(data["class"])]
        if cls == subcls:
            raise TypeError("Cannot instantiate ProModel base class.")
        return subcls.from_dict(data)

    @property
    def population(self):
        """Promolecular population."""
        return sum(fn.population for fn in self.fns)

    def assign_pars(self, pars):
        """Assign the promolecule parameters to the basis functions."""
        ipar = 0
        for fn in self.fns:
            fn.pars[:] = pars[ipar : ipar + fn.npar]
            ipar += fn.npar

    def compute_density(self, grid, localgrids=None, cache=None):
        """Compute prodensity on a grid (for the given parameters).

        Parameters
        ----------
        grid
            The whole integration grid, on which the results is computed.
        localgrids
            A list of local grids, one for each basis function.
        cache
            An optional ComputeCache instance for reusing intermediate results.

        Returns
        -------
        pro
            The pro-molecule density on the points of ``grid``.

        """
        pro = np.zeros_like(grid.weights)
        if localgrids is None:
            for fn in self.fns:
                pro += fn.compute(grid.points, cache)
        else:
            for fn, localgrid in zip(self.fns, localgrids):
                np.add.at(pro, localgrid.indices, fn.compute(localgrid.points, cache))
        return pro

    def compute_proatom(self, iatom, points, cache=None):
        """Compute proatom density on a set of points.

        Parameters
        ----------
        iatom
            The atomic index.
        points
            A set of points on which the proatom must be computed.
        cache
            An optional ComputeCache instance for reusing intermediate results.

        Returns
        -------
        pro
            The pro-atom density on the points of ``grid``.

        """
        pro = 0
        for fn in self.fns:
            if fn.iatom == iatom:
                pro += fn.compute(points, cache)
        return pro

    def pprint(self):
        """Print a table with the pro-parameters."""
        print(" ifn iatom  atn       parameters...")
        for ifn, fn in enumerate(self.fns):
            print(
                "{:4d}  {:4d}  {:3d}  {:s}".format(
                    ifn,
                    fn.iatom,
                    self.atnums[fn.iatom],
                    " ".join(format(par, "15.8f") for par in fn.pars),
                )
            )


def ekld(pars, grid, density, pro_model, localgrids, pop, cache=None, density_cutoff=1e-15):
    """Compute the Extended KL divergence and its gradient.

    Parameters
    ----------
    pars
        A NumPy array with promodel parameters.
    grid
        A numerical integration grid with, instance of ``grid.basegrid.Grid``.
    density
        The electron density evaluated on the grid.
    pro_model
        The model for the pro-molecular density, an instance of ``ProModel``.
    local_grids
        A list of local integration grids for the pro-model basis functions.
    pop
        The integral of density, to be precomputed before calling this function.
    cache
        An optional ComputeCache instance for reusing intermediate results.
    density_cutoff
        Density cutoff used to neglect grid points with low densities. Including
        them can result in numerical noise in the result and its derivatives.

    Returns
    -------
    ekld
        The extended KL-d, i.e. including the Lagrange multiplier.
    gradient
        The gradient of ekld w.r.t. the pro-model parameters.

    """
    time_start = time.process_time()

    pro_model.assign_pars(pars)
    pro = pro_model.compute_density(grid, localgrids, cache)

    # Compute potentially tricky quantities.
    sick = (density < density_cutoff) | (pro < density_cutoff)
    with np.errstate(all="ignore"):
        lnratio = np.log(density) - np.log(pro)
        ratio = density / pro
    lnratio[sick] = 0.0
    ratio[sick] = 0.0
    # Function value
    kld = np.einsum("i,i,i", grid.weights, density, lnratio)

    constraint = pop - pro_model.population
    result = kld - constraint
    # Gradient
    ipar = 0
    gradient = np.zeros_like(pars)

    for ifn, fn in enumerate(pro_model.fns):
        localgrid = localgrids[ifn]
        fn_derivatives = fn.compute_derivatives(localgrid.points, cache)
        gradient[ipar : ipar + fn.npar] = fn.population_derivatives - np.einsum(
            "i,i,ji", localgrid.weights, ratio[localgrid.indices], fn_derivatives
        )
        ipar += fn.npar

    # Save some quantities for screen output
    time_stop = time.process_time()
    pro_model.ekld_info = {
        "ekld": result,
        "kld": kld,
        "constraint": constraint,
        "gradient": gradient,
        "time": time_stop - time_start,
    }
    if cache is not None:
        cache.discard("end-ekld")
    return result, gradient
