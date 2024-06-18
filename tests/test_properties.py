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
"""Unit tests for properties module."""

import numpy as np
import pytest
from denspart.properties import spherical_harmonics
from numpy.testing import assert_allclose
from scipy.special import sph_harm


def sph_harm_real(m, n, theta, phi):
    """Construct real spherical harmonics, using SciPy's sph_harm function.

    Parameters
    ----------
    m
        order
    n
        degree
    theta
        azimuthal angle
    phi
        polar angle

    Returns
    -------
    harmonics
        One array is returned when m == 0. Two arrays are returned when m != 0,
        the first being the cosine-like function (real part) and the second being the
        sine-like function (imaginary part). Results are always L2-normalized and
        SciPy's Condon-Shortley phase is removed.

    """
    assert m >= 0
    if m == 0:
        return sph_harm(m, n, theta, phi).real
    else:
        tmp = sph_harm(m, n, theta, phi) * np.sqrt(2) * (-1) ** m
        return tmp.real, tmp.imag


@pytest.mark.parametrize("ellmax", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("solid", [True, False])
@pytest.mark.parametrize("racah", [True, False])
def test_spherical_harmonics(ellmax, solid, racah):
    npt = 20
    points = np.random.normal(0, 1, (npt, 3))
    r = np.linalg.norm(points, axis=1)
    x, y, z = points.T
    phi = np.arccos(z / r)
    theta = np.arctan2(y, x)

    # Prepare results array
    result = np.zeros(((ellmax + 1) ** 2 - 1, npt), float)
    result[0] = z
    result[1] = x
    result[2] = y

    if solid and not racah:
        with pytest.raises(ValueError):
            spherical_harmonics(result, ellmax, solid, racah)
    else:
        # Comparison
        spherical_harmonics(result, ellmax, solid, racah)
        i = 0
        for ell in range(1, ellmax + 1):
            factor = np.sqrt(4 * np.pi / (2 * ell + 1)) if racah else 1
            if solid:
                factor *= r**ell
            for m in range(ell + 1):
                if m == 0:
                    assert_allclose(result[i], factor * sph_harm_real(0, ell, theta, phi))
                    i += 1
                else:
                    yc, ys = sph_harm_real(m, ell, theta, phi)
                    assert_allclose(result[i], factor * yc)
                    i += 1
                    assert_allclose(result[i], factor * ys)
                    i += 1


def test_regular_solid_spherical_harmonics():
    npt = 20
    points = np.random.normal(0, 1, (npt, 3))
    r = np.linalg.norm(points, axis=1)
    x, y, z = points.T

    # Prepare results array
    ellmax = 3
    result = np.zeros(((ellmax + 1) ** 2 - 1, npt), float)
    result[0] = z
    result[1] = x
    result[2] = y

    # Comparison
    spherical_harmonics(result, ellmax, solid=True)
    # ell = 1, m = 0
    assert_allclose(result[0], z)
    # ell = 1, m = 1
    assert_allclose(result[1], x)
    assert_allclose(result[2], y)
    # ell = 2, m = 0
    assert_allclose(result[3], (3 * z**2 - r**2) / 2)
    # ell = 2, m = 1
    assert_allclose(result[4], np.sqrt(3) * z * x)
    assert_allclose(result[5], np.sqrt(3) * z * y)
    # ell = 2, m = 2
    assert_allclose(result[6], np.sqrt(3) / 2 * (x**2 - y**2))
    assert_allclose(result[7], np.sqrt(3) * x * y)
    # ell = 3, m = 0
    assert_allclose(result[8], z * (5 * z**2 - 3 * r**2) / 2)
