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
# pylint: disable=too-many-lines
"""Gaussian Iterative Stockholder Analysis (GISA) partitioning scheme."""


import warnings
from itertools import combinations

import numpy as np

from .cache import compute_cached
from .mbis import connected_vertices
from .vh import BasisFunction, ProModel

__all__ = ["GISAProModel"]


class GaussianFunction(BasisFunction):
    """Gaussian basis function for the GISA pro density.

    See BasisFunction base class for API documentation.
    """

    def __init__(self, iatom, center, pars, exponent):
        self.exponent = exponent
        if len(pars) != 1 and not (np.asarray(pars) >= 0).all():
            raise TypeError("Expecting one positive parameter.")
        super().__init__(iatom, center, pars, [(5e-5, 1e2)])

    @property
    def population(self):
        return self.pars[0]

    @property
    def population_derivatives(self):
        return np.array([1.0])

    def get_cutoff_radius(self, density_cutoff):
        if density_cutoff <= 0.0:
            return np.inf
        population, exponent = self.pars[0], self.exponent
        prefactor = population * (exponent / np.pi) ** 1.5
        if prefactor < 0 or prefactor < density_cutoff:
            return np.inf
        else:
            return np.sqrt((np.log(prefactor) - np.log(density_cutoff)) / exponent)

    def _compute_dists(self, points, cache=None):
        return compute_cached(
            cache,
            until="forever",
            key=("dists", *self.center, len(points)),
            func=(lambda: np.linalg.norm(points - self.center, axis=1)),
        )

    def _compute_exp(self, exponent, dists, cache=None):
        # print(exponent, np.max(dists**2), np.min(dists**2))
        return compute_cached(
            cache,
            until="end-ekld",
            key=("exp", *self.center, exponent, len(dists)),
            func=(lambda: np.exp(-exponent * dists**2)),
        )

    def compute(self, points, cache=None):
        population, exponent = self.pars[0], self.exponent
        if exponent < 0 or population < 0:
            return np.full(len(points), np.inf)
        dists = self._compute_dists(points, cache)
        exp = self._compute_exp(exponent, dists, cache)
        prefactor = population * (exponent / np.pi) ** 1.5
        return prefactor * exp

    def compute_derivatives(self, points, cache=None):
        population, exponent = self.pars[0], self.exponent
        if exponent < 0 or population < 0:
            warnings.warn("exponent or population is negative!", stacklevel=1)
            exponent = -exponent if exponent < 0 else exponent
            population = -population if population < 0 else population
        dists = self._compute_dists(points, cache)
        exp = self._compute_exp(exponent, dists, cache)
        factor = (exponent / np.pi) ** 1.5
        # vector = (population * exponent**2 / 8 / np.pi) * (3 - dists * exponent)
        return np.array([factor * exp])


class GISAProModel(ProModel):
    """ProModel for MBIS partitioning."""

    @classmethod
    def from_geometry(cls, atnums, atcoords):
        """Derive a ProModel with a sensible initial guess from a molecular geometry.

        Parameters
        ----------
        atnums
            An array with atomic numbers, shape ``(natom, )``.
        atcoords
            An array with atomic coordinates, shape ``(natom, 3)``
        """
        fns = []
        for iatom, (atnum, atcoord) in enumerate(zip(atnums, atcoords, strict=True)):
            exponents = get_alpha(atnum)
            populations = get_initial_population(atnum, exponents)
            for population, exponent in zip(populations, exponents, strict=True):
                fns.append(GaussianFunction(iatom, atcoord, [population], exponent))
        return cls(atnums, atcoords, fns)

    def reduce(self, eps=1e-4):
        """Return a new ProModel in which redundant functions are merged together.

        Parameters
        ----------
        eps
            When abs(e1 - e2) < eps * (e1 + e2) / 2, were e1 and e2 are exponents,
            two functions will be merged. Also when the population of a basis function
            is lower then eps, it is removed.

        """
        pro_model = super().reduce(eps)
        # Group functions by atoms
        grouped_fns = {}
        for fn in pro_model.fns:
            grouped_fns.setdefault(fn.iatom, []).append(fn)
        # Loop over all atoms and merge where possible
        new_fns = []
        for iatom, fns in grouped_fns.items():
            pairs = [
                (fn1, fn2)
                for fn1, fn2 in combinations(fns, 2)
                if abs(fn1.exponent - fn2.exponent) < eps * (fn1.exponent + fn2.exponent) / 2
            ]
            clusters = connected_vertices(pairs, fns)
            for cluster in clusters:
                population = sum(fn.population for fn in cluster)
                exponent = sum(fn.exponent for fn in cluster) / len(cluster)
                new_fns.append(
                    GaussianFunction(iatom, pro_model.atcoords[iatom], [population], exponent)
                )
        return pro_model.__class__(pro_model.atnums, pro_model.atcoords, new_fns)

    def to_dict(self):
        """Return dictionary with additional results derived from the pro-parameters."""
        return super().to_dict()

    @classmethod
    def from_dict(cls, data):
        """Recreate the pro-model from a dictionary."""
        if data["class"] != "GISAProModel":
            raise TypeError("The dictionary class field should be GISAProModel.")
        fns = []
        ipar = 0
        atnums = data["atnums"]
        atcoords = data["atcoords"]
        pars = data["propars"]
        atnfns = data["atnfns"]
        for iatom, atcoord in enumerate(atcoords):
            exponents = get_alpha(atnums[iatom])
            for iprim in range(atnfns[iatom]):
                fn_pars = pars[ipar]
                fns.append(GaussianFunction(iatom, atcoord, fn_pars, exponents[iprim]))
                ipar += 1
        return cls(atnums, atcoords, fns)


def get_alpha(atnum):
    """The exponents used for primitive Gaussian functions of each element."""
    param_dict = {
        1: np.array([5.672, 1.505, 0.5308, 0.2204]),
        6: np.array([148.3, 42.19, 15.33, 6.146, 0.7846, 0.2511]),
        7: np.array([178.0, 52.42, 19.87, 1.276, 0.6291, 0.2857]),
        8: np.array([220.1, 65.66, 25.98, 1.685, 0.6860, 0.2311]),
    }
    if atnum in param_dict:
        return param_dict[atnum]
    else:
        raise NotImplementedError


def get_initial_population(atnum, exponents):
    """Get initial population based on atomic number `atnum`."""
    return np.ones_like(exponents) * atnum / len(exponents)
