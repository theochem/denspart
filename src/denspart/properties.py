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
"""Compute atom-in-molecules properties."""

import numpy as np

from .cache import compute_cached

__all__ = ["compute_radial_moments", "compute_multipole_moments", "spherical_harmonics"]


def safe_ratio(density, pro, density_cutoff=1e-15):
    """Compute density / pro, with safeguards for small values.

    Parameters
    ----------
    density
        The "real" density.
    pro
        The "model" density.
    density_cutoff
        Densities below this cutoff are ignored.

    Returns
    -------
    ratio
        density / pro, or zero when density or pro are below the cutoff.

    """
    # Compute potentially tricky quantities.
    sick = (density < density_cutoff) | (pro < density_cutoff)
    with np.errstate(all="ignore"):
        ratio = density / pro
    ratio[sick] = 0.0
    return ratio


def _compute_dists(points, center, cache=None):
    """Helper function for cached distance computation."""
    return compute_cached(
        cache,
        until="forever",
        key=("dists", *center, len(points)),
        func=(lambda: np.linalg.norm(points - center, axis=1)),
    )


def compute_radial_moments(
    pro_model, grid, density, localgrids, density_cutoff=1e-10, cache=None, nmax=4
):
    """Compute expectation values of r^n for each atom.

    Parameters
    ----------
    pro_model
        The model for the pro-molecular density, an instance of ``ProModel``.
    grid
        The whole integration grid, instance of ``grid.basegrid.Grid.``
    density
        The electron density.
    localgrids
        A list of local grids, one for each basis function.
    density_cutoff
        Density cutoff used to estimated sizes of local grids. Set to zero for
        whole-grid integrations. (This will not work for periodic systems.)
    cache
        An optional ComputeCache instance for reusing intermediate results.
    nmax
        Maximum degree of the radial moment to be computed.

    Returns
    -------
    rexp
        An array with expectation values of r^n for n from 0 to nmax.

    """
    pro = pro_model.compute_density(grid, localgrids)
    result = np.zeros((pro_model.natom, nmax + 1))
    radii = pro_model.get_cutoff_radii(density_cutoff)
    for iatom, (atcoord, radius) in enumerate(zip(pro_model.atcoords, radii, strict=True)):
        localgrid = grid.get_localgrid(atcoord, radius)
        pro_atom = pro_model.compute_proatom(iatom, localgrid.points, cache)
        ratio = safe_ratio(density[localgrid.indices], pro[localgrid.indices])
        dists = _compute_dists(localgrid.points, atcoord, cache)
        for degree in np.arange(nmax + 1):
            result[iatom, degree] = localgrid.integrate(pro_atom, ratio, dists**degree)
    return result


def compute_multipole_moments(
    pro_model, grid, density, localgrids, density_cutoff=1e-10, cache=None, ellmax=4
):
    """Compute expectation values of r^n for each atom.

    Parameters
    ----------
    pro_model
        The model for the pro-molecular density, an instance of ``ProModel``.
    grid
        The whole integration grid, instance of ``grid.basegrid.Grid.``
    density
        The electron density.
    localgrids
        A list of local grids, one for each basis function.
    density_cutoff
        Density cutoff used to estimated sizes of local grids. Set to zero for
        whole-grid integrations. (This will not work for periodic systems.)
    cache
        An optional ComputeCache instance for reusing intermediate results.
    ellmax
        Maximum angular momentum to be computed.

    Returns
    -------
    moments
        An array with multipole moments. The ordering is fixed by the function
        spherical_harmonics in this module.

    """
    pro = pro_model.compute_density(grid, localgrids)
    result = np.zeros((pro_model.natom, (ellmax + 1) ** 2 - 1))
    radii = pro_model.get_cutoff_radii(density_cutoff)
    for iatom, (atcoord, radius) in enumerate(zip(pro_model.atcoords, radii, strict=True)):
        localgrid = grid.get_localgrid(atcoord, radius)
        operators = np.zeros(((ellmax + 1) ** 2 - 1, localgrid.size))
        operators[:3] = (localgrid.points - atcoord)[:, [2, 0, 1]].T
        spherical_harmonics(operators, ellmax, solid=True)
        pro_atom = pro_model.compute_proatom(iatom, localgrid.points, cache)
        ratio = safe_ratio(density[localgrid.indices], pro[localgrid.indices])
        for iop, operator in enumerate(operators):
            result[iatom, iop] = localgrid.integrate(pro_atom, ratio, operator)
    return result


def spherical_harmonics(work, ellmax, solid=False, racah=None):
    """Recursive calculation of spherical harmonics.

    Parameters
    ----------
    work
        The input and output array. First three elements should contain x, y and z.
        After calling this function, the spherical harmonics are stored in Horton 2
        order: c10 c11 s11 c20 c21 s21 c22 s22 c30 c31 s31 c32 s32 c33 s33 ...
        (c stands for cosine-like, s for sine like, first digit is ell, second digit is m.)
    ellmax
        Maximum angular momentum. The work array should have at least (ellmax + 1)**2 - 1
        elements along the first dimension.
    solid
        When True, the real regular solid harmonics are computed instead of the normal
        spherical harmonics.
    racah
        Use Racah's normalization. The default is False for conventional spherical harmonics
        and True for solid harmonics. Setting this to False for solid harmonics will
        raise an error. When ``racah==True``, the L2 norm of the spherical harmonics is
        4 pi / (2 ell + 1).

    """
    if racah is None:
        racah = solid
    if solid and not racah:
        raise ValueError("Regular solid spherical harmonics always use racah normalization.")
    if work.shape[0] < (ellmax + 1) ** 2 - 1:
        raise ValueError("Work array is too small for given ellmax.")

    shape = work[0].shape
    z = work[0]
    x = work[1]
    y = work[2]

    r2 = x * x + y * y + z * z
    if not solid:
        r = np.sqrt(r2)
        mask = r > 0
        rmask = r[mask]
        z[mask] /= rmask
        x[mask] /= rmask
        y[mask] /= rmask
        r2[mask] = 1

    # temporary arrays to store PI(z,r) polynomials
    tmp_shape = (ellmax + 1, *shape)
    pi_old = np.zeros(tmp_shape)
    pi_new = np.zeros(tmp_shape)
    a = np.zeros(tmp_shape)
    b = np.zeros(tmp_shape)

    # Initialize the temporary arrays
    pi_old[0] = 1
    pi_new[0] = z
    pi_new[1] = 1
    a[1] = x
    b[1] = y

    old_offset = 0  # first array index of the moments of the previous shell
    old_npure = 3  # number of moments in previous shell
    for ell in range(2, ellmax + 1):
        new_npure = old_npure + 2
        new_offset = old_offset + old_npure

        # Polynomials PI(z,r) for current ell
        factor = 2 * ell - 1
        for m in range(ell - 1):
            tmp = pi_old[m].copy()
            pi_old[m] = pi_new[m]
            pi_new[m] = (z * factor * pi_old[m] - r2 * (ell + m - 1) * tmp) / (ell - m)

        pi_old[ell - 1] = pi_new[ell - 1]
        pi_new[ell] = factor * pi_old[ell - 1]
        pi_new[ell - 1] = z * pi_new[ell]

        # construct new polynomials A(x,y) and B(x,y)
        a[ell] = x * a[ell - 1] - y * b[ell - 1]
        b[ell] = x * b[ell - 1] + y * a[ell - 1]

        # construct solid harmonics
        work[new_offset] = pi_new[0]
        factor = np.sqrt(2)
        for m in range(1, ell + 1):
            factor /= np.sqrt((ell + m) * (ell - m + 1))
            work[new_offset + 2 * m - 1] = factor * a[m] * pi_new[m]
            work[new_offset + 2 * m] = factor * b[m] * pi_new[m]
        old_npure = new_npure
        old_offset = new_offset

    if not (solid or racah):
        work /= 2 * np.sqrt(np.pi)
        begin = 0
        end = 3
        for ell in range(1, ellmax + 1):
            print(begin, end, 2 * ell + 1)
            work[begin:end] *= np.sqrt(2 * ell + 1)
            begin = end
            end += 2 * ell + 3
