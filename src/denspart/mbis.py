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
"""Minimal Basis Iterative Stockholder."""

from itertools import combinations

import numpy as np

from .cache import compute_cached
from .vh import BasisFunction, ProModel

__all__ = ["MBISProModel"]


class ExponentialFunction(BasisFunction):
    """Exponential basis function for the MBIS pro density.

    See BasisFunction base class for API documentation.
    """

    def __init__(self, iatom, center, pars):
        if len(pars) != 2 and not (pars >= 0).all():
            raise TypeError("Expecting two positive parameters.")
        super().__init__(iatom, center, pars, [(5e-5, 1e2), (0.1, 1e3)])

    @property
    def population(self):
        return self.pars[0]

    @property
    def exponent(self):
        """Exponent of the exponential functions."""
        return self.pars[1]

    @property
    def population_derivatives(self):
        return np.array([1.0, 0.0])

    def get_cutoff_radius(self, density_cutoff):
        """Cutoff radius at which the exponential function becomes smaller than the given cutoff.

        Parameter
        ---------
        density_cutoff
            The threshold value for the density

        Returns
        -------
        radius
            The distance from the center where the exponential function
            becomes smaller than the density_cutoff.
        """
        if density_cutoff <= 0.0:
            return np.inf
        population, exponent = self.pars
        return (
            np.log(population) + 3 * np.log(exponent) - np.log(8 * np.pi) - np.log(density_cutoff)
        ) / exponent

    def _compute_dists(self, points, cache=None):
        return compute_cached(
            cache,
            until="forever",
            key=("dists", *self.center, len(points)),
            func=(lambda: np.linalg.norm(points - self.center, axis=1)),
        )

    def _compute_exp(self, exponent, dists, cache=None):
        return compute_cached(
            cache,
            until="end-ekld",
            key=("exp", *self.center, exponent, len(dists)),
            func=(lambda: np.exp(-exponent * dists)),
        )

    def compute(self, points, cache=None):
        population, exponent = self.pars
        if exponent < 0 or population < 0:
            return np.full(len(points), np.inf)
        dists = self._compute_dists(points, cache)
        exp = self._compute_exp(exponent, dists, cache)
        prefactor = population * (exponent**3 / 8 / np.pi)
        return prefactor * exp

    def compute_derivatives(self, points, cache=None):
        population, exponent = self.pars
        if exponent < 0 or population < 0:
            return np.full((2, len(points)), np.inf)
        dists = self._compute_dists(points, cache)
        exp = self._compute_exp(exponent, dists, cache)
        factor = exponent**3 / 8 / np.pi
        vector = (population * exponent**2 / 8 / np.pi) * (3 - dists * exponent)
        return np.array([factor * exp, vector * exp])


def connected_vertices(pairs, vertices):
    """Derive the connected vertices from a list of pairs.

    Note: this could be done more efficiently with NetworkX or similar libraries, but
    adding a heavy dependency for a simple feature is not worth it.
    """
    lookup = dict((vertex, [vertex]) for vertex in vertices)
    for pair in pairs:
        item0, item1 = pair
        members0 = lookup.get(item0)
        members1 = lookup.get(item1)
        if members0 is None:
            if members1 is None:
                cluster = [item0, item1]
                lookup[item0] = cluster
                lookup[item1] = cluster
            else:
                members1.append(item0)
                lookup[item0] = members1
        elif members1 is None:
            members0.append(item1)
            lookup[item1] = members0
        else:
            members0.extend(members1)
            for item in members1:
                lookup[item] = members0
    return set(frozenset(cluster) for cluster in lookup.values())


class MBISProModel(ProModel):
    """ProModel for MBIS partitioning."""

    @classmethod
    def from_geometry(cls, atnums, atcoords, nshell_map=None):
        """Derive a ProModel with a sensible initial guess from a molecular geometry.

        Parameters
        ----------
        atnums
            An array with atomic numbers, shape ``(natom, )``.
        atcoords
            An array with atomic coordinates, shape ``(natom, 3)``
        nshell_map
            A dictionary with the number of shells needed for a specific element.
            When absent or when some elements are not included, all shells
            in INITIAL_MBIS_PARAMETERS are be used. This argument can be
            used to reduce that number of shells, outer shells being removed first.
        """
        fns = []
        nshell_map = {} if nshell_map is None else nshell_map
        for iatom, (atnum, atcoord) in enumerate(zip(atnums, atcoords, strict=True)):
            nshell = nshell_map.get(atnum)
            shells = INITIAL_MBIS_PARAMETERS[atnum]
            if nshell is not None:
                if nshell <= 0 or nshell > len(shells):
                    raise ValueError(
                        f"The number of shells for atomic number {atnum} "
                        f"must be in [1, {len(shells)}]. Got {nshell}."
                    )
                shells = shells[:nshell]
            for population, exponent in shells:
                fns.append(ExponentialFunction(iatom, atcoord, [population, exponent]))
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
                    ExponentialFunction(iatom, pro_model.atcoords[iatom], [population, exponent])
                )
        return pro_model.__class__(pro_model.atnums, pro_model.atcoords, new_fns)

    def to_dict(self):
        """Return dictionary with additional results derived from the pro-parameters."""
        results = super().to_dict()
        valence_charges = np.zeros(self.natom, dtype=float)
        valence_widths = np.zeros(self.natom, dtype=float)
        for fn in self.fns:
            width = 1 / fn.pars[1]
            if width > valence_widths[fn.iatom]:
                valence_widths[fn.iatom] = width
                valence_charges[fn.iatom] = -fn.pars[0]

        core_charges = self.charges - valence_charges
        results.update(
            {
                "core_charges": core_charges,
                "valence_charges": valence_charges,
                "valence_widths": valence_widths,
            }
        )
        return results

    @classmethod
    def from_dict(cls, data):
        """Recreate the pro-model from a dictionary."""
        if data["class"] != "MBISProModel":
            raise TypeError("The dictionary class field should be MBISProModel.")
        fns = []
        ipar = 0
        atnums = data["atnums"]
        atcoords = data["atcoords"]
        pars = data["propars"]
        atnfns = data["atnfns"]
        for iatom, atcoord in enumerate(atcoords):
            for _ in range(atnfns[iatom]):
                fn_pars = pars[ipar : ipar + 2]
                fns.append(ExponentialFunction(iatom, atcoord, fn_pars))
                ipar += 2
        return cls(atnums, atcoords, fns)


INITIAL_MBIS_PARAMETERS = {
    # niter = 2
    # kld/n = 0.00236
    # n     = 1.00000
    # npro  = 1.00000
    1: [(1.00000, 1.76216)],
    # niter = 2
    # kld/n = 0.00707
    # n     = 2.00000
    # npro  = 2.00000
    2: [(2.00000, 3.11975)],
    # niter = 25
    # kld/n = 0.00543
    # n     = 3.00000
    # npro  = 3.00000
    3: [(1.86359, 5.56763), (1.13641, 0.80520)],
    # niter = 27
    # kld/n = 0.00931
    # n     = 4.00000
    # npro  = 4.00000
    4: [(1.75663, 8.15111), (2.24337, 1.22219)],
    # niter = 28
    # kld/n = 0.00500
    # n     = 5.00000
    # npro  = 5.00000
    5: [(1.73486, 10.46135), (3.26514, 1.51797)],
    # niter = 32
    # kld/n = 0.00372
    # n     = 6.00000
    # npro  = 6.00000
    6: [(1.70730, 12.79758), (4.29270, 1.85580)],
    # niter = 38
    # kld/n = 0.00306
    # n     = 7.00000
    # npro  = 7.00000
    7: [(1.68283, 15.13096), (5.31717, 2.19942)],
    # niter = 43
    # kld/n = 0.00279
    # n     = 8.00000
    # npro  = 8.00000
    8: [(1.66122, 17.46129), (6.33878, 2.54326)],
    # niter = 47
    # kld/n = 0.00275
    # n     = 9.00000
    # npro  = 9.00000
    9: [(1.64171, 19.78991), (7.35829, 2.88601)],
    # niter = 52
    # kld/n = 0.00287
    # n     = 10.00000
    # npro  = 10.00000
    10: [(1.62380, 22.11938), (8.37620, 3.22746)],
    # niter = 61
    # kld/n = 0.00349
    # n     = 11.00000
    # npro  = 11.00000
    11: [(1.48140, 25.82522), (8.28761, 4.02120), (1.23098, 0.80897)],
    # niter = 66
    # kld/n = 0.00650
    # n     = 12.00000
    # npro  = 12.00000
    12: [(1.39674, 29.19802), (8.10904, 4.76791), (2.49422, 1.08302)],
    # niter = 66
    # kld/n = 0.00659
    # n     = 13.00000
    # npro  = 13.00000
    13: [(1.34503, 32.33363), (8.12124, 5.42812), (3.53372, 1.15994)],
    # niter = 63
    # kld/n = 0.00802
    # n     = 14.00000
    # npro  = 14.00000
    14: [(1.28865, 35.65432), (7.98931, 6.17545), (4.72204, 1.33797)],
    # niter = 72
    # kld/n = 0.00939
    # n     = 15.00000
    # npro  = 15.00000
    15: [(1.23890, 39.00531), (7.83125, 6.95265), (5.92985, 1.52690)],
    # niter = 79
    # kld/n = 0.01052
    # n     = 16.00000
    # npro  = 16.00000
    16: [(1.19478, 42.38177), (7.66565, 7.75584), (7.13957, 1.71687)],
    # niter = 85
    # kld/n = 0.01145
    # n     = 17.00000
    # npro  = 17.00000
    17: [(1.15482, 45.79189), (7.50031, 8.58542), (8.34487, 1.90546)],
    # niter = 91
    # kld/n = 0.01224
    # n     = 18.00000
    # npro  = 18.00000
    18: [(1.11803, 49.24317), (7.33917, 9.44200), (9.54280, 2.09210)],
    # niter = 113
    # kld/n = 0.01424
    # n     = 19.00000
    # npro  = 19.00000
    19: [
        (1.09120, 52.59376),
        (7.15086, 10.29851),
        (9.57061, 2.42121),
        (1.18733, 0.67314),
    ],
    # niter = 136
    # kld/n = 0.01500
    # n     = 20.00000
    # npro  = 20.00000
    20: [
        (1.07196, 55.86008),
        (7.01185, 11.11887),
        (9.29555, 2.76621),
        (2.62063, 0.88692),
    ],
    # niter = 153
    # kld/n = 0.01344
    # n     = 21.00000
    # npro  = 21.00000
    21: [
        (1.05870, 59.04659),
        (6.96404, 11.86718),
        (9.97866, 2.93024),
        (2.99860, 0.98040),
    ],
    # niter = 165
    # kld/n = 0.01218
    # n     = 22.00000
    # npro  = 22.00000
    22: [
        (1.04755, 62.22091),
        (6.90438, 12.62229),
        (10.84355, 3.08264),
        (3.20452, 1.05403),
    ],
    # niter = 174
    # kld/n = 0.01115
    # n     = 23.00000
    # npro  = 23.00000
    23: [
        (1.03828, 65.38117),
        (6.83516, 13.38417),
        (11.79532, 3.23508),
        (3.33124, 1.11609),
    ],
    # niter = 180
    # kld/n = 0.01026
    # n     = 24.00000
    # npro  = 24.00000
    24: [
        (1.03069, 68.52633),
        (6.75998, 14.15132),
        (12.79256, 3.38991),
        (3.41677, 1.17116),
    ],
    # niter = 184
    # kld/n = 0.00949
    # n     = 25.00000
    # npro  = 25.00000
    25: [
        (1.02450, 71.65908),
        (6.68141, 14.92337),
        (13.81149, 3.54730),
        (3.48260, 1.22220),
    ],
    # niter = 187
    # kld/n = 0.00879
    # n     = 26.00000
    # npro  = 26.00000
    26: [
        (1.01960, 74.77846),
        (6.60101, 15.69935),
        (14.84330, 3.70685),
        (3.53609, 1.27026),
    ],
    # niter = 189
    # kld/n = 0.00817
    # n     = 27.00000
    # npro  = 27.00000
    27: [
        (1.01575, 77.88779),
        (6.51976, 16.47941),
        (15.88061, 3.86829),
        (3.58388, 1.31647),
    ],
    # niter = 191
    # kld/n = 0.00761
    # n     = 28.00000
    # npro  = 28.00000
    28: [
        (1.01282, 80.98814),
        (6.43837, 17.26336),
        (16.92012, 4.03115),
        (3.62869, 1.36133),
    ],
    # niter = 211
    # kld/n = 0.00639
    # n     = 29.00000
    # npro  = 29.00000
    29: [
        (1.01839, 83.81831),
        (6.47823, 17.85149),
        (18.65720, 4.05312),
        (2.84618, 1.37570),
    ],
    # niter = 194
    # kld/n = 0.00663
    # n     = 30.00000
    # npro  = 30.00000
    30: [
        (1.00931, 87.16777),
        (6.27682, 18.84319),
        (18.99747, 4.35989),
        (3.71640, 1.44857),
    ],
    # niter = 164
    # kld/n = 0.00684
    # n     = 31.00000
    # npro  = 31.00000
    31: [
        (1.00600, 90.34057),
        (6.16315, 19.71091),
        (19.81836, 4.57852),
        (4.01249, 1.29122),
    ],
    # niter = 152
    # kld/n = 0.00707
    # n     = 32.00000
    # npro  = 32.00000
    32: [
        (0.99467, 93.80965),
        (5.91408, 20.85993),
        (19.89501, 4.95158),
        (5.19624, 1.39361),
    ],
    # niter = 165
    # kld/n = 0.00766
    # n     = 33.00000
    # npro  = 33.00000
    33: [
        (0.98548, 97.22822),
        (5.68319, 22.01684),
        (19.83497, 5.33969),
        (6.49637, 1.51963),
    ],
    # niter = 180
    # kld/n = 0.00831
    # n     = 34.00000
    # npro  = 34.00000
    34: [
        (0.97822, 100.60094),
        (5.47209, 23.17528),
        (19.68845, 5.73803),
        (7.86124, 1.65366),
    ],
    # niter = 193
    # kld/n = 0.00895
    # n     = 35.00000
    # npro  = 35.00000
    35: [
        (0.97231, 103.94730),
        (5.27765, 24.33975),
        (19.48822, 6.14586),
        (9.26182, 1.78869),
    ],
    # niter = 205
    # kld/n = 0.00955
    # n     = 36.00000
    # npro  = 36.00000
    36: [
        (0.96735, 107.28121),
        (5.09646, 25.51581),
        (19.25332, 6.56380),
        (10.68288, 1.92256),
    ],
    # niter = 272
    # kld/n = 0.01017
    # n     = 37.00000
    # npro  = 37.00000
    37: [
        (0.96706, 110.48309),
        (4.99899, 26.50212),
        (18.99122, 6.92726),
        (10.76759, 2.18101),
        (1.27514, 0.66954),
    ],
    # niter = 340
    # kld/n = 0.01027
    # n     = 38.00000
    # npro  = 38.00000
    38: [
        (0.96801, 113.67680),
        (4.93897, 27.41353),
        (18.80330, 7.25498),
        (10.35813, 2.46293),
        (2.93159, 0.86625),
    ],
    # niter = 387
    # kld/n = 0.00975
    # n     = 39.00000
    # npro  = 39.00000
    39: [
        (0.96684, 116.97488),
        (4.85128, 28.43242),
        (18.94517, 7.56989),
        (10.53663, 2.55617),
        (3.70008, 0.97332),
    ],
    # niter = 435
    # kld/n = 0.00939
    # n     = 40.00000
    # npro  = 40.00000
    40: [
        (0.96535, 120.30371),
        (4.74742, 29.51436),
        (19.08019, 7.90688),
        (11.05560, 2.60522),
        (4.15144, 1.06342),
    ],
    # niter = 490
    # kld/n = 0.00908
    # n     = 41.00000
    # npro  = 41.00000
    41: [
        (0.96228, 123.69423),
        (4.59559, 30.76696),
        (19.29222, 8.28677),
        (13.04646, 2.49390),
        (3.10346, 1.06692),
    ],
    # niter = 550
    # kld/n = 0.00903
    # n     = 42.00000
    # npro  = 42.00000
    42: [
        (0.96141, 127.02899),
        (4.47623, 31.93860),
        (19.27950, 8.67584),
        (14.22293, 2.54514),
        (3.05992, 1.12354),
    ],
    # niter = 581
    # kld/n = 0.00903
    # n     = 43.00000
    # npro  = 43.00000
    43: [
        (0.96101, 130.36147),
        (4.35465, 33.14068),
        (19.21943, 9.08439),
        (15.54548, 2.60482),
        (2.91943, 1.16411),
    ],
    # niter = 586
    # kld/n = 0.00907
    # n     = 44.00000
    # npro  = 44.00000
    44: [
        (0.96111, 133.68940),
        (4.23256, 34.36928),
        (19.12341, 9.51019),
        (16.94562, 2.67378),
        (2.73729, 1.19329),
    ],
    # niter = 572
    # kld/n = 0.00914
    # n     = 45.00000
    # npro  = 45.00000
    45: [
        (0.96172, 137.01241),
        (4.11107, 35.62210),
        (19.00142, 9.95142),
        (18.37578, 2.75113),
        (2.55000, 1.21476),
    ],
    # niter = 940
    # kld/n = 0.00920
    # n     = 46.00000
    # npro  = 46.00000
    46: [
        (0.96222, 140.34382),
        (3.97975, 36.95092),
        (18.97452, 10.39657),
        (20.49458, 2.75967),
        (1.58893, 1.30000),
    ],
    # niter = 518
    # kld/n = 0.00933
    # n     = 47.00000
    # npro  = 47.00000
    47: [
        (0.96441, 143.64600),
        (3.87219, 38.19648),
        (18.71042, 10.87495),
        (21.22900, 2.92470),
        (2.22399, 1.24459),
    ],
    # niter = 464
    # kld/n = 0.00947
    # n     = 48.00000
    # npro  = 48.00000
    48: [
        (0.96721, 146.96006),
        (3.78203, 39.39447),
        (18.45110, 11.34436),
        (21.44461, 3.11456),
        (3.35504, 1.32681),
    ],
    # niter = 323
    # kld/n = 0.00953
    # n     = 49.00000
    # npro  = 49.00000
    49: [
        (0.96990, 150.29845),
        (3.69397, 40.61549),
        (18.22732, 11.81657),
        (22.45558, 3.25901),
        (3.65322, 1.16914),
    ],
    # niter = 345
    # kld/n = 0.00971
    # n     = 50.00000
    # npro  = 50.00000
    50: [
        (0.97329, 153.61921),
        (3.60552, 41.84383),
        (17.87583, 12.33351),
        (22.43949, 3.49629),
        (5.10586, 1.27096),
    ],
    # niter = 372
    # kld/n = 0.00992
    # n     = 51.00000
    # npro  = 51.00000
    51: [
        (0.97679, 156.98207),
        (3.53402, 43.02243),
        (17.55869, 12.83309),
        (22.26210, 3.73890),
        (6.66839, 1.37644),
    ],
    # niter = 393
    # kld/n = 0.01007
    # n     = 52.00000
    # npro  = 52.00000
    52: [
        (0.98027, 160.39249),
        (3.47671, 44.15853),
        (17.26668, 13.31791),
        (21.90419, 3.99243),
        (8.37215, 1.49131),
    ],
    # niter = 415
    # kld/n = 0.01019
    # n     = 53.00000
    # npro  = 53.00000
    53: [
        (0.98368, 163.85359),
        (3.43054, 45.26231),
        (16.99206, 13.79195),
        (21.42699, 4.25770),
        (10.16674, 1.60670),
    ],
    # niter = 437
    # kld/n = 0.01029
    # n     = 54.00000
    # npro  = 54.00000
    54: [
        (0.98698, 167.36704),
        (3.39333, 46.34048),
        (16.72687, 14.25880),
        (20.87117, 4.53670),
        (12.02165, 1.72006),
    ],
    # niter = 824
    # kld/n = 0.01009
    # n     = 55.00000
    # npro  = 55.00000
    55: [
        (0.98855, 171.08380),
        (3.39137, 47.32865),
        (16.81286, 14.56630),
        (20.37387, 4.67981),
        (12.00664, 1.93716),
        (1.42671, 0.64967),
    ],
    # Merging exponents:  13.325427 13.320737
    # Populations before: 6.540189 12.930462
    # Population after:   19.470651
    # niter = 1236
    # kld/n = 0.01242
    # n     = 56.00000
    # npro  = 56.00000
    56: [
        (0.99731, 175.01327),
        (3.94901, 45.62379),
        (19.47065, 13.32231),
        (25.53046, 3.58081),
        (6.05257, 1.01688),
    ],
    # Merging exponents:  13.970795 13.968286
    # Populations before: 6.651932 12.385003
    # Population after:   19.036935
    # niter = 931
    # kld/n = 0.01135
    # n     = 57.00000
    # npro  = 57.00000
    57: [
        (0.99490, 178.92585),
        (3.78923, 47.36151),
        (19.03694, 13.96916),
        (25.48253, 3.84663),
        (7.69641, 1.12920),
    ],
    # Merging exponents:  14.264029 14.260789
    # Populations before: 7.127339 11.926493
    # Population after:   19.053831
    # niter = 1036
    # kld/n = 0.01116
    # n     = 58.00000
    # npro  = 58.00000
    58: [
        (0.99773, 182.69729),
        (3.79809, 48.23836),
        (19.05383, 14.26200),
        (26.50828, 3.89345),
        (7.64207, 1.14501),
    ],
    # Merging exponents:  14.568959 14.565152
    # Populations before: 7.849370 11.189641
    # Population after:   19.039011
    # niter = 1170
    # kld/n = 0.01097
    # n     = 59.00000
    # npro  = 59.00000
    59: [
        (1.00049, 186.51613),
        (3.80487, 49.13072),
        (19.03901, 14.56672),
        (27.58038, 3.94881),
        (7.57525, 1.15879),
    ],
    # Merging exponents:  14.883327 14.879035
    # Populations before: 9.619595 9.379373
    # Population after:   18.998968
    # niter = 1424
    # kld/n = 0.01079
    # n     = 60.00000
    # npro  = 60.00000
    60: [
        (1.00319, 190.38490),
        (3.81025, 50.03637),
        (18.99897, 14.88121),
        (28.67528, 4.01194),
        (7.51231, 1.17175),
    ],
    # niter = 1774
    # kld/n = 0.00828
    # n     = 61.00000
    # npro  = 61.00000
    61: [
        (0.99995, 194.01705),
        (3.24882, 54.04025),
        (16.47119, 16.98218),
        (23.00328, 5.32410),
        (12.91959, 2.51531),
        (4.35716, 0.98405),
    ],
    # niter = 1740
    # kld/n = 0.00804
    # n     = 62.00000
    # npro  = 62.00000
    62: [
        (1.00209, 197.99687),
        (3.22761, 55.17619),
        (16.34706, 17.42152),
        (23.80125, 5.42023),
        (13.27747, 2.57444),
        (4.34451, 0.99393),
    ],
    # niter = 1737
    # kld/n = 0.00782
    # n     = 63.00000
    # npro  = 63.00000
    63: [
        (1.00429, 202.02953),
        (3.20862, 56.30909),
        (16.21652, 17.86527),
        (24.63902, 5.51636),
        (13.59806, 2.63146),
        (4.33349, 1.00348),
    ],
    # niter = 1742
    # kld/n = 0.00761
    # n     = 64.00000
    # npro  = 64.00000
    64: [
        (1.00654, 206.11731),
        (3.19138, 57.44219),
        (16.08281, 18.31332),
        (25.49620, 5.61277),
        (13.89990, 2.68701),
        (4.32318, 1.01283),
    ],
    # niter = 1749
    # kld/n = 0.00741
    # n     = 65.00000
    # npro  = 65.00000
    65: [
        (1.00883, 210.26214),
        (3.17612, 58.57320),
        (15.94486, 18.76554),
        (26.38056, 5.70951),
        (14.17647, 2.74080),
        (4.31316, 1.02181),
    ],
    # niter = 1755
    # kld/n = 0.00721
    # n     = 66.00000
    # npro  = 66.00000
    66: [
        (1.01117, 214.46688),
        (3.16274, 59.70253),
        (15.80365, 19.22175),
        (27.28732, 5.80665),
        (14.43201, 2.79301),
        (4.30312, 1.03043),
    ],
    # niter = 1759
    # kld/n = 0.00702
    # n     = 67.00000
    # npro  = 67.00000
    67: [
        (1.01354, 218.73341),
        (3.15115, 60.83054),
        (15.65984, 19.68187),
        (28.21315, 5.90421),
        (14.67005, 2.84364),
        (4.29227, 1.03860),
    ],
    # niter = 1761
    # kld/n = 0.00683
    # n     = 68.00000
    # npro  = 68.00000
    68: [
        (1.01595, 223.06415),
        (3.14128, 61.95766),
        (15.51409, 20.14581),
        (29.15557, 6.00216),
        (14.89317, 2.89270),
        (4.27995, 1.04625),
    ],
    # niter = 1759
    # kld/n = 0.00665
    # n     = 69.00000
    # npro  = 69.00000
    69: [
        (1.01840, 227.46136),
        (3.13302, 63.08447),
        (15.36688, 20.61357),
        (30.11252, 6.10047),
        (15.10393, 2.94016),
        (4.26525, 1.05328),
    ],
    # niter = 1754
    # kld/n = 0.00648
    # n     = 70.00000
    # npro  = 70.00000
    70: [
        (1.02088, 231.92848),
        (3.12631, 64.21152),
        (15.21913, 21.08480),
        (31.08374, 6.19886),
        (15.30077, 2.98599),
        (4.24917, 1.05982),
    ],
    # niter = 1746
    # kld/n = 0.00631
    # n     = 71.00000
    # npro  = 71.00000
    71: [
        (1.02340, 236.46708),
        (3.12101, 65.33964),
        (15.07067, 21.55992),
        (32.06704, 6.29753),
        (15.48957, 3.02994),
        (4.22831, 1.06533),
    ],
    # niter = 1948
    # kld/n = 0.00614
    # n     = 72.00000
    # npro  = 72.00000
    72: [
        (1.02614, 241.05912),
        (3.11380, 66.48612),
        (14.91939, 22.05124),
        (33.32537, 6.39717),
        (14.51375, 3.06086),
        (5.10155, 1.16902),
    ],
    # niter = 1973
    # kld/n = 0.00600
    # n     = 73.00000
    # npro  = 73.00000
    73: [
        (1.02929, 245.67915),
        (3.10090, 67.66413),
        (14.72483, 22.58997),
        (34.56189, 6.52150),
        (13.79943, 3.03942),
        (5.78367, 1.25985),
    ],
    # niter = 1885
    # kld/n = 0.00590
    # n     = 74.00000
    # npro  = 74.00000
    74: [
        (1.03282, 250.33518),
        (3.08458, 68.85920),
        (14.49437, 23.16956),
        (35.56831, 6.67658),
        (13.59733, 2.99108),
        (6.22259, 1.33543),
    ],
    # niter = 1833
    # kld/n = 0.00583
    # n     = 75.00000
    # npro  = 75.00000
    75: [
        (1.03659, 255.04631),
        (3.06638, 70.07019),
        (14.24505, 23.77881),
        (36.35311, 6.85375),
        (13.82288, 2.93796),
        (6.47600, 1.39947),
    ],
    # niter = 1878
    # kld/n = 0.00581
    # n     = 76.00000
    # npro  = 76.00000
    76: [
        (1.04065, 259.80929),
        (3.04708, 71.28687),
        (13.97524, 24.41922),
        (36.87935, 7.05641),
        (14.58843, 2.89227),
        (6.46926, 1.45041),
    ],
    # niter = 2008
    # kld/n = 0.00581
    # n     = 77.00000
    # npro  = 77.00000
    77: [
        (1.04497, 264.63166),
        (3.02753, 72.50494),
        (13.69135, 25.08677),
        (37.19139, 7.28053),
        (15.80637, 2.86375),
        (6.23840, 1.48910),
    ],
    # niter = 2133
    # kld/n = 0.00583
    # n     = 78.00000
    # npro  = 78.00000
    78: [
        (1.04951, 269.52094),
        (3.00836, 73.72166),
        (13.39860, 25.77813),
        (37.33902, 7.52233),
        (17.34748, 2.85595),
        (5.85704, 1.51755),
    ],
    # niter = 2207
    # kld/n = 0.00601
    # n     = 79.00000
    # npro  = 79.00000
    79: [
        (1.05572, 274.27910),
        (2.96900, 75.00010),
        (12.96851, 26.66074),
        (37.27090, 7.84449),
        (20.99429, 2.78498),
        (3.74158, 1.47920),
    ],
    # niter = 2181
    # kld/n = 0.00594
    # n     = 80.00000
    # npro  = 80.00000
    80: [
        (1.05910, 279.52863),
        (2.97260, 76.14674),
        (12.80192, 27.22274),
        (37.30541, 8.04711),
        (20.91497, 2.89600),
        (4.94600, 1.55349),
    ],
    # niter = 979
    # kld/n = 0.00591
    # n     = 81.00000
    # npro  = 81.00000
    81: [
        (1.06254, 284.86830),
        (2.97535, 77.31377),
        (12.63238, 27.79957),
        (37.40661, 8.25124),
        (23.16204, 2.90463),
        (3.76108, 1.25293),
    ],
    # niter = 978
    # kld/n = 0.00585
    # n     = 82.00000
    # npro  = 82.00000
    82: [
        (1.06627, 290.28925),
        (2.98032, 78.45692),
        (12.42797, 28.40635),
        (36.95686, 8.51126),
        (23.07427, 3.10731),
        (5.49431, 1.35142),
    ],
    # niter = 971
    # kld/n = 0.00582
    # n     = 83.00000
    # npro  = 83.00000
    83: [
        (1.06952, 295.89089),
        (2.99110, 79.60411),
        (12.26809, 28.96419),
        (36.60258, 8.74903),
        (23.18839, 3.28577),
        (6.88031, 1.38803),
    ],
    # niter = 1066
    # kld/n = 0.00578
    # n     = 84.00000
    # npro  = 84.00000
    84: [
        (1.07247, 301.65782),
        (3.00695, 80.74997),
        (12.13336, 29.48765),
        (36.12686, 8.98696),
        (22.79251, 3.51606),
        (8.86784, 1.48179),
    ],
    # niter = 1240
    # kld/n = 0.00574
    # n     = 85.00000
    # npro  = 85.00000
    85: [
        (1.07504, 307.60992),
        (3.02761, 81.90658),
        (12.03635, 29.96333),
        (35.62798, 9.21085),
        (22.13626, 3.77443),
        (11.09675, 1.58575),
    ],
    # niter = 1511
    # kld/n = 0.00569
    # n     = 86.00000
    # npro  = 86.00000
    86: [
        (1.07727, 313.74785),
        (3.05219, 83.07870),
        (11.97392, 30.39458),
        (35.09317, 9.42149),
        (21.34497, 4.06298),
        (13.45848, 1.68982),
    ],
    # niter = 3115
    # kld/n = 0.00557
    # n     = 87.00000
    # npro  = 87.00000
    87: [
        (1.07866, 320.13828),
        (3.07367, 84.37488),
        (12.01829, 30.74418),
        (35.74839, 9.48671),
        (19.04804, 4.19650),
        (14.27307, 1.97507),
        (1.75989, 0.71651),
    ],
    # niter = 30668
    # kld/n = 0.00540
    # n     = 88.00000
    # npro  = 88.00000
    88: [
        (1.08004, 326.68617),
        (3.09096, 85.73578),
        (12.07540, 31.10148),
        (37.02057, 9.50368),
        (15.73782, 4.21159),
        (14.71155, 2.36783),
        (4.28366, 0.89370),
    ],
    # Merging exponents:  9.173651 9.165177
    # Populations before: 14.536884 25.936997
    # Population after:   40.473881
    # niter = 3957
    # kld/n = 0.00534
    # n     = 89.00000
    # npro  = 89.00000
    89: [
        (1.07846, 333.89969),
        (3.17050, 86.83801),
        (12.57061, 30.80584),
        (40.47386, 9.16822),
        (24.82719, 3.11081),
        (6.87937, 1.04676),
    ],
    # Merging exponents:  9.431775 9.424734
    # Populations before: 14.756506 25.330770
    # Population after:   40.087275
    # niter = 2827
    # kld/n = 0.00518
    # n     = 90.00000
    # npro  = 90.00000
    90: [
        (1.08208, 340.45553),
        (3.16504, 88.26458),
        (12.37624, 31.47596),
        (40.08726, 9.42733),
        (24.56010, 3.31023),
        (8.72929, 1.15770),
    ],
    # Merging exponents:  9.627249 9.620639
    # Populations before: 15.204702 25.112659
    # Population after:   40.317361
    # niter = 2840
    # kld/n = 0.00507
    # n     = 91.00000
    # npro  = 91.00000
    91: [
        (1.08553, 347.22535),
        (3.17119, 89.63206),
        (12.22718, 32.06968),
        (40.31735, 9.62313),
        (25.55323, 3.30511),
        (8.64552, 1.17701),
    ],
    # Merging exponents:  9.828700 9.822889
    # Populations before: 15.769845 24.981753
    # Population after:   40.751598
    # niter = 2936
    # kld/n = 0.00501
    # n     = 92.00000
    # npro  = 92.00000
    92: [
        (1.08939, 354.11371),
        (3.17670, 90.99571),
        (12.03566, 32.71878),
        (40.75159, 9.82514),
        (28.06165, 3.18256),
        (6.88501, 1.11150),
    ],
    # Merging exponents:  10.087683 10.082800
    # Populations before: 16.123297 24.569645
    # Population after:   40.692942
    # niter = 2551
    # kld/n = 0.00492
    # n     = 93.00000
    # npro  = 93.00000
    93: [
        (1.09355, 361.13749),
        (3.17538, 92.42691),
        (11.81000, 33.44525),
        (40.69294, 10.08473),
        (29.47989, 3.22145),
        (6.74825, 1.12213),
    ],
    # Merging exponents:  10.362682 10.358581
    # Populations before: 16.456136 24.121356
    # Population after:   40.577492
    # niter = 2249
    # kld/n = 0.00484
    # n     = 94.00000
    # npro  = 94.00000
    94: [
        (1.09794, 368.32500),
        (3.17404, 93.87067),
        (11.57027, 34.20518),
        (40.57749, 10.36024),
        (30.96551, 3.26929),
        (6.61475, 1.13111),
    ],
    # Merging exponents:  10.650419 10.646941
    # Populations before: 16.773266 23.648280
    # Population after:   40.421546
    # niter = 2021
    # kld/n = 0.00477
    # n     = 95.00000
    # npro  = 95.00000
    95: [
        (1.10254, 375.69056),
        (3.17360, 95.32199),
        (11.32206, 34.99094),
        (40.42154, 10.64838),
        (32.48783, 3.32401),
        (6.49243, 1.13917),
    ],
    # Merging exponents:  10.945019 10.942109
    # Populations before: 17.084304 23.175550
    # Population after:   40.259854
    # niter = 1852
    # kld/n = 0.00471
    # n     = 96.00000
    # npro  = 96.00000
    96: [
        (1.10729, 383.24958),
        (3.17377, 96.78830),
        (11.07137, 35.79948),
        (40.25985, 10.94334),
        (33.99604, 3.38149),
        (6.39168, 1.14787),
    ],
    # Merging exponents:  11.248063 11.246067
    # Populations before: 17.387440 22.691812
    # Population after:   40.079252
    # niter = 1739
    # kld/n = 0.00465
    # n     = 97.00000
    # npro  = 97.00000
    97: [
        (1.11222, 391.01298),
        (3.17544, 98.26135),
        (10.81808, 36.62729),
        (40.07925, 11.24693),
        (35.51639, 3.44250),
        (6.29863, 1.15589),
    ],
    # Merging exponents:  11.558296 11.557429
    # Populations before: 17.685969 22.200537
    # Population after:   39.886507
    # niter = 1678
    # kld/n = 0.00461
    # n     = 98.00000
    # npro  = 98.00000
    98: [
        (1.11730, 398.99467),
        (3.17879, 99.74179),
        (10.56415, 37.47189),
        (39.88651, 11.55781),
        (37.04014, 3.50607),
        (6.21310, 1.16337),
    ],
    # Merging exponents:  11.875171 11.874890
    # Populations before: 17.983276 21.703229
    # Population after:   39.686505
    # niter = 1631
    # kld/n = 0.00457
    # n     = 99.00000
    # npro  = 99.00000
    99: [
        (1.12255, 407.20909),
        (3.18387, 101.23136),
        (10.31099, 38.33175),
        (39.68650, 11.87502),
        (38.56155, 3.57146),
        (6.13453, 1.17038),
    ],
    # Merging exponents:  12.197598 12.197517
    # Populations before: 18.283126 21.200499
    # Population after:   39.483625
    # niter = 1572
    # kld/n = 0.00454
    # n     = 100.00000
    # npro  = 100.00000
    100: [
        (1.12794, 415.67191),
        (3.19069, 102.73254),
        (10.05989, 39.20531),
        (39.48362, 12.19755),
        (40.07701, 3.63798),
        (6.06085, 1.17681),
    ],
    # Merging exponents:  12.524735 12.524712
    # Populations before: 18.589514 20.691265
    # Population after:   39.280779
    # niter = 1483
    # kld/n = 0.00452
    # n     = 101.00000
    # npro  = 101.00000
    101: [
        (1.13348, 424.39937),
        (3.19917, 104.24845),
        (9.81173, 40.09171),
        (39.28078, 12.52472),
        (41.58459, 3.70516),
        (5.99026, 1.18259),
    ],
    # Merging exponents:  12.855892 12.855886
    # Populations before: 18.907030 20.173593
    # Population after:   39.080623
    # niter = 1495
    # kld/n = 0.00451
    # n     = 102.00000
    # npro  = 102.00000
    102: [
        (1.13915, 433.40929),
        (3.20918, 105.78307),
        (9.56720, 40.99055),
        (39.08062, 12.85589),
        (43.08210, 3.77261),
        (5.92175, 1.18763),
    ],
    # Merging exponents:  13.190432 13.190430
    # Populations before: 19.240857 19.644295
    # Population after:   38.885152
    # niter = 1501
    # kld/n = 0.00450
    # n     = 103.00000
    # npro  = 103.00000
    103: [
        (1.14496, 442.72080),
        (3.22058, 107.34061),
        (9.32700, 41.90122),
        (38.88515, 13.19043),
        (44.56942, 3.83997),
        (5.85289, 1.19177),
    ],
    # Merging exponents:  13.603086 13.603086
    # Populations before: 19.447586 18.899529
    # Population after:   38.347115
    # niter = 1459
    # kld/n = 0.00443
    # n     = 104.00000
    # npro  = 104.00000
    104: [
        (1.15135, 452.26605),
        (3.22875, 108.92084),
        (9.04304, 42.93588),
        (38.34711, 13.60309),
        (45.02618, 4.00670),
        (7.20356, 1.29377),
    ],
    # Merging exponents:  13.973048 13.973048
    # Populations before: 19.670158 18.227460
    # Population after:   37.897618
    # niter = 1501
    # kld/n = 0.00439
    # n     = 105.00000
    # npro  = 105.00000
    105: [
        (1.15720, 462.29654),
        (3.24386, 110.55376),
        (8.82683, 43.83101),
        (37.89762, 13.97305),
        (45.25449, 4.16409),
        (8.62000, 1.39805),
    ],
    # Merging exponents:  14.311341 14.311341
    # Populations before: 19.911944 17.604853
    # Population after:   37.516797
    # niter = 1533
    # kld/n = 0.00435
    # n     = 106.00000
    # npro  = 106.00000
    106: [
        (1.16267, 472.80654),
        (3.26226, 112.26116),
        (8.65940, 44.62894),
        (37.51680, 14.31134),
        (45.31035, 4.31472),
        (10.08853, 1.50038),
    ],
    # Merging exponents:  14.646367 14.646367
    # Populations before: 20.181227 16.957447
    # Population after:   37.138674
    # niter = 1563
    # kld/n = 0.00433
    # n     = 107.00000
    # npro  = 107.00000
    107: [
        (1.16812, 483.75845),
        (3.28113, 114.03669),
        (8.50793, 45.40411),
        (37.13867, 14.64637),
        (45.28879, 4.46883),
        (11.61536, 1.59154),
    ],
    # Merging exponents:  14.972374 14.972374
    # Populations before: 20.479082 16.294621
    # Population after:   36.773703
    # niter = 1589
    # kld/n = 0.00431
    # n     = 108.00000
    # npro  = 108.00000
    108: [
        (1.17350, 495.19373),
        (3.30025, 115.89209),
        (8.37792, 46.14357),
        (36.77370, 14.97237),
        (45.11349, 4.62719),
        (13.26114, 1.68611),
    ],
    # Merging exponents:  15.290072 15.290072
    # Populations before: 20.819701 15.601403
    # Population after:   36.421104
    # niter = 1610
    # kld/n = 0.00429
    # n     = 109.00000
    # npro  = 109.00000
    109: [
        (1.17885, 507.14435),
        (3.31904, 117.83528),
        (8.26742, 46.85201),
        (36.42110, 15.29007),
        (44.80854, 4.78963),
        (15.00504, 1.78141),
    ],
    # Merging exponents:  15.601828 15.601828
    # Populations before: 21.224393 14.850710
    # Population after:   36.075102
    # niter = 1630
    # kld/n = 0.00428
    # n     = 110.00000
    # npro  = 110.00000
    110: [
        (1.18421, 519.64313),
        (3.33703, 119.87184),
        (8.17332, 47.53691),
        (36.07510, 15.60183),
        (44.38981, 4.95728),
        (16.84052, 1.87653),
    ],
    # Merging exponents:  15.910167 15.910167
    # Populations before: 21.731832 13.997132
    # Population after:   35.728964
    # niter = 1650
    # kld/n = 0.00426
    # n     = 111.00000
    # npro  = 111.00000
    111: [
        (1.18963, 532.72803),
        (3.35389, 122.00711),
        (8.09284, 48.20504),
        (35.72896, 15.91017),
        (43.86996, 5.13167),
        (18.76472, 1.97119),
    ],
    # Merging exponents:  16.217528 16.217527
    # Populations before: 22.423358 12.952556
    # Population after:   35.375915
    # niter = 1676
    # kld/n = 0.00425
    # n     = 112.00000
    # npro  = 112.00000
    112: [
        (1.19513, 546.44374),
        (3.36937, 124.24696),
        (8.02369, 48.86232),
        (35.37591, 16.21753),
        (43.26106, 5.31444),
        (20.77485, 2.06523),
    ],
    # Merging exponents:  16.196079 16.196076
    # Populations before: 24.186636 11.825912
    # Population after:   36.012548
    # niter = 1815
    # kld/n = 0.00451
    # n     = 113.00000
    # npro  = 113.00000
    113: [
        (1.19831, 561.46167),
        (3.38080, 126.93676),
        (8.19964, 49.01826),
        (36.01255, 16.19608),
        (44.61015, 5.20890),
        (19.59855, 1.98048),
    ],
    # Merging exponents:  16.414870 16.414861
    # Populations before: 27.656404 8.336923
    # Population after:   35.993327
    # niter = 2046
    # kld/n = 0.00442
    # n     = 114.00000
    # npro  = 114.00000
    114: [
        (1.20363, 576.73694),
        (3.38927, 129.51002),
        (8.18834, 49.61420),
        (35.99333, 16.41487),
        (45.19984, 5.27324),
        (20.02560, 1.97092),
    ],
    # niter = 2112
    # kld/n = 0.00402
    # n     = 115.00000
    # npro  = 115.00000
    115: [
        (1.21246, 591.92054),
        (3.39659, 131.76192),
        (7.83915, 50.99178),
        (34.63427, 17.11127),
        (42.94186, 5.74241),
        (22.77382, 2.32051),
        (2.20185, 1.02716),
    ],
    # niter = 2672
    # kld/n = 0.00394
    # n     = 116.00000
    # npro  = 116.00000
    116: [
        (1.21804, 608.90539),
        (3.40088, 134.61027),
        (7.83224, 51.62782),
        (34.54736, 17.35046),
        (42.66989, 5.85605),
        (22.24232, 2.46122),
        (4.08928, 1.16490),
    ],
    # niter = 3665
    # kld/n = 0.00385
    # n     = 117.00000
    # npro  = 117.00000
    117: [
        (1.22355, 626.95975),
        (3.40140, 137.66654),
        (7.84820, 52.25136),
        (34.58793, 17.55428),
        (42.79420, 5.92328),
        (20.78333, 2.58696),
        (6.36139, 1.30054),
    ],
    # niter = 5324
    # kld/n = 0.00375
    # n     = 118.00000
    # npro  = 118.00000
    118: [
        (1.22909, 646.17926),
        (3.39867, 140.93076),
        (7.87683, 52.88265),
        (34.70741, 17.73814),
        (43.32835, 5.95447),
        (18.42228, 2.70154),
        (9.03736, 1.42977),
    ],
}
