#!/usr/bin/env python3
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
"""Prepare inputs for denspart with HORTON3 modules.

This implementation makes some ad hoc coices on the molecular integration
grid, which should be revised in future. For now, this is something that is
just supposed to just work. The code is not tuned for precision nor for
efficiency.

This module is far from polished and is currently only used for prototyping:

- The integration grid is not final. It may have precision issues and it is not
  pruned. Furthermore, all atoms are given the same atomic grid, which is far
  from optimal and the Becke partitioning is used, which can be improved upon.

- Only the spin-summed density is computed, using the post-hf 1RDM if it is
  present. The spin-difference density is ignored.

- It is slow.

"""


import argparse

import numpy as np

from iodata import load_one

from gbasis.wrappers import from_iodata
from gbasis.evals.density import evaluate_density

from grid.becke import BeckeWeights
from grid.atomgrid import AtomGrid
from grid.molgrid import MolGrid
from grid.onedgrid import GaussChebyshev, HortonLinear, OneDGrid
from grid.rtransform import BeckeTF, HyperbolicRTransform, PowerRTransform
from grid.interpolate import spline_with_atomic_grid

from ase.units import Bohr

from gpaw.utilities import unpack2


__all__ = ["prepare_input"]


def prepare_input(iodata, nrad, nang, chunk_size):
    """Prepare input for denspart with HORTON3 modules.

    Parameters
    ----------
    iodata
        An instance with IOData containing the necessary data to compute the
        electron density on the grid.
    nrad
        Number of radial grid points.
    nang
        Number of angular grid points.
    chunk_size
        Number of points on which the density is evaluated in one pass.

    Returns
    -------
    grid
        A molecular integration grid.
    rho
        The electron density on the grid.

    """
    grid = _setup_grid(iodata.atnums, iodata.atcoords, nrad, nang)
    one_rdm = iodata.one_rdms.get("post_scf", iodata.one_rdms.get("scf"))
    if one_rdm is None:
        if iodata.mo is None:
            raise ValueError(
                "The input file lacks wavefunction data with which "
                "the density can be computed."
            )
        coeffs, occs = iodata.mo.coeffs, iodata.mo.occs
        one_rdm = np.dot(coeffs * occs, coeffs.T)
    rho = _compute_density(iodata, one_rdm, grid.points, chunk_size)
    return grid, rho


def prepare_input_gpw(atoms, calc, grid_size):
    """Prepare input for denspart with HORTON3 modules.

    Returns
    -------
    input_data
        A dictionary with all input data for a partitioning.
    """

    atnums = atoms.get_atomic_numbers()
    atcorenums = atoms.get_atomic_numbers()
    atcoords = atoms.get_positions(wrap=True)/Bohr
    cellvecs = calc.density.gd.cell_cv

    print('Loading grid data, setups, atoms')
    pseudo_grid_data = _get_pseudo_grid_data(calc, cellvecs, atnums)
    setups, atoms = _get_atomic_grid_data(calc, atnums, atcoords)

    print('Computing augmentation spheres')
    compute_augmentation_spheres(pseudo_grid_data, setups, atoms, atnums, atcoords, grid_size, cellvecs)
    compute_uniform_points(pseudo_grid_data)
    density = denspart_conventions(pseudo_grid_data, atoms, atnums)

    input_data = {
        "atcoords": atcoords,
        "atnums": atnums,
        "atcorenums": atcorenums,
        "points": density['points'],
        "weights": density['weights'],
        "rho": density['rho'],
        "cellvecs": cellvecs,
    }

    return input_data
 

def compute_augmentation_spheres(grid_data, setups, atoms, numbers, coordinates, grid_size, cellvecs):

    w = abs(np.linalg.det(grid_data['grid_vecs']))
    nelec_pseudo = grid_data['pseudo_density'].sum()*w
    if grid_data['nspins'] == 2:
        spin_pseudo = grid_data['pseudo_spindensity'].sum()*w
    natom = len(numbers)
    qcors = grid_data['charge_corrections']
    myqcors = np.zeros(natom)
    if grid_data['nspins'] == 2:
        sqcors = grid_data['spincharge_corrections']
        mysqcors = np.zeros(natom)
    else:
        sqcors = None
        mysqcors = None

    print('   Atom  DensPart QCor      GPAW QCor          Error')
    print('~~~~~~~  ~~~~~~~~~~~~~  ~~~~~~~~~~~~~  ~~~~~~~~~~~~~')

    for iatom in range(natom):

        atomg = atoms['%03i' % iatom]
        id_setup = atomg['id_setup']
        setupg = setups[id_setup]

        rad = setupg['nc_radgrid']
        atgrid_short = AtomGrid.from_predefined(
            numbers[iatom] if numbers[iatom] <= 36 else 27,
            rad,
            grid_type=grid_size,
            center=coordinates[iatom]
        )

        rhoc, rhoct, rhov, rhovt, spinrhov, spinrhovt = \
                eval_correction(atgrid_short, atomg, setupg, coordinates[iatom])

        rhoc_cor = rhoc - rhoct
        rhov_cor = rhov - rhovt
        spinrhov_cor = spinrhov - spinrhovt

        # store some extra stuff in the dictionary
        atomg['grid_points'] = atgrid_short.points
        atomg['grid_weights'] = atgrid_short.weights
        atomg['rhoc'] = rhoc
        atomg['rhoct'] = rhoct
        atomg['rhoc_cor'] = rhoc_cor
        atomg['rhov'] = rhov
        atomg['rhovt'] = rhovt
        atomg['rhov_cor'] = rhov_cor
        atomg['spinrhov'] = spinrhov
        atomg['spinrhovt'] = spinrhovt
        atomg['spinrhov_cor'] = spinrhov_cor

        # add things up and compare
        # - core part
        myqcors[iatom] = atgrid_short.integrate(rhoc_cor) - numbers[iatom]
        # - valence part
        vcor = atgrid_short.integrate(rhov_cor)
        myqcors[iatom] += vcor
        print('{:2d} {:4d}   {:12.7f}   {:12.7f}   {:12.5e}'.format(
            numbers[iatom], iatom,
            myqcors[iatom], qcors[iatom], myqcors[iatom] - qcors[iatom]))

        if sqcors is not None:
            mysqcors[iatom] = atgrid_short.integrate(spinrhov_cor)
            print('spin      {:12.7f}   {:12.7f}   {:12.5e}'.format(
                mysqcors[iatom], sqcors[iatom], mysqcors[iatom] - sqcors[iatom]))

    # Final checks
    print('GPAW total charge:     %10.3e' % (nelec_pseudo + qcors.sum()))
    print('DensPart total charge: %10.3e' % (nelec_pseudo + myqcors.sum()))
    assert np.allclose(qcors, myqcors)
    if sqcors is not None:
        print('GPAW total spin:       %10.3e' % (spin_pseudo + sqcors.sum()))
        print('DensPart total spin:   %10.3e' % (spin_pseudo + mysqcors.sum()))
        assert np.allclose(sqcors, mysqcors)


def eval_correction(grid, atomg, setupg, center):

    d = np.linalg.norm(grid._points, axis=1)
    N = len(d)

    # Compute the core density correction
    cs_nc = setupg['nc']
    rhoc = cs_nc(d)
    cs_nct = setupg['nct']
    rhoct = cs_nct(d)

    # Compute real spherical harmonics on grid
    lmax = 4
    polys = np.zeros((grid.size, (lmax+1)**2-1), float)
    polys[:,0] = grid.points[:,2] - center[2]
    polys[:,1] = grid.points[:,0] - center[0]
    polys[:,2] = grid.points[:,1] - center[1]

    if True:
        # divide by r before computing harmonics -> real spherical (not solid)
        r = np.sqrt(polys[:,0]**2 + polys[:,1]**2 + polys[:,2]**2)
        nskip = (r == 0.0).sum()
        polys[nskip:,:3] /= r[nskip:].reshape(-1,1)
        fill_pure_polynomials(polys[nskip:], 4)
        polys[:nskip] = 1.0
    else:
        fill_pure_polynomials(polys, 4)

    # Evaluate each pseudo and ae basis function in the atomic grid
    ls = setupg['ls']
    basis_fns = []
    basist_fns = []
    begin = 0
    for iradial, l in enumerate(ls):
        nfn = 2*l+1
        offset = l**2

        # evaluate radial functions
        phi = setupg['phi_%03i' % iradial]
        basis = phi(d)
        phit = setupg['phit_%03i' % iradial]
        basist = phit(d)

        # multiply with the corresponding spherical harmonics
        if l == 0:
            basis_fns.append(basis)
            basist_fns.append(basist)
        else:
            for ifn in range(nfn):
                basis_fns.append(basis*polys[:,offset+ifn-1])
                basist_fns.append(basist*polys[:,offset+ifn-1])

    # Debugging check
    if True:
        # Construct the local overlap matrix
        olp = np.zeros((len(basis_fns), len(basis_fns)))
        olpt = np.zeros((len(basis_fns), len(basis_fns)))
        for ibasis0 in range(len(basis_fns)):
            for ibasis1 in range(ibasis0+1):
                olp[ibasis0, ibasis1] = grid.integrate(basis_fns[ibasis0]*basis_fns[ibasis1])
                olp[ibasis1, ibasis0] = olp[ibasis0, ibasis1]
                olpt[ibasis0, ibasis1] = grid.integrate(basist_fns[ibasis0]*basist_fns[ibasis1])
                olpt[ibasis1, ibasis0] = olpt[ibasis0, ibasis1]
        assert np.allclose(olp - olpt, setupg['overlap'])

    # Load the atomic density matrix
    dm = atomg['dm']
    if 'sdm' in atomg:
        sdm = atomg['sdm']
    else:
        sdm = None

    rhov, rhovt, spinrhov, spinrhovt = np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N)
    # Loop over all pairs of basis functions and add product times density matrix coeff
    for ibasis0 in range(len(basis_fns)):
        for ibasis1 in range(ibasis0+1):
            factor = (ibasis0!=ibasis1) + 1
            rhov += factor*dm[ibasis0, ibasis1]*basis_fns[ibasis0]*basis_fns[ibasis1]
            rhovt += factor*dm[ibasis0, ibasis1]*basist_fns[ibasis0]*basist_fns[ibasis1]
            if sdm is not None:
                spinrhov += factor*sdm[ibasis0, ibasis1]*basis_fns[ibasis0]*basis_fns[ibasis1]
                spinrhovt += factor*sdm[ibasis0, ibasis1]*basist_fns[ibasis0]*basist_fns[ibasis1]

    # Sanity check
    if True:
        rhov_cor = rhov - rhovt
        assert np.allclose(grid.integrate(rhov_cor), np.dot((olp-olpt).ravel(), dm.ravel()))

    return rhoc, rhoct, rhov, rhovt, spinrhov, spinrhovt


def _get_pseudo_grid_data(calc, cellvecs, atnums):

    grid_data = {}
    grid_data['shape'] = calc.density.gd.N_c
    grid_data['grid_vecs'] = calc.density.gd.h_cv
    grid_data['nspins'] = calc.wfs.nspins
    for i in range(3):
        assert np.allclose(calc.density.gd.h_cv[i],
                           cellvecs[i]/grid_data['shape'][i])

    if calc.wfs.nspins == 1:
        grid_data['charge_corrections'] = calc.get_pseudo_density_corrections()
        grid_data['pseudo_density'] = calc.get_pseudo_density()*(Bohr**3)
        grid_data['ae_density'] = calc.get_all_electron_density(gridrefinement=1)*(Bohr**3)
    else:
        corrections = calc.get_pseudo_density_corrections()
        grid_data['charge_corrections'] = corrections[0] + corrections[1]
        grid_data['spincharge_corrections'] = corrections[0] - corrections[1]

        rho_pseudo_alpha = calc.get_pseudo_density(0)*(Bohr**3)
        rho_pseudo_beta = calc.get_pseudo_density(1)*(Bohr**3)
        grid_data['pseudo_density'] = rho_pseudo_alpha + rho_pseudo_beta
        grid_data['pseudo_spindensity'] = rho_pseudo_alpha - rho_pseudo_beta

        rho_ae_alpha = calc.get_all_electron_density(spin=0, gridrefinement=1)*(Bohr**3)
        rho_ae_beta = calc.get_all_electron_density(spin=1, gridrefinement=1)*(Bohr**3)
        grid_data['ae_density'] = rho_ae_alpha + rho_ae_beta
        grid_data['ae_spindensity'] = rho_ae_alpha - rho_ae_beta

    # Sanity checks
    assert (grid_data['pseudo_density'].shape == grid_data['shape']).all()
    assert (grid_data['ae_density'].shape == grid_data['shape']).all()
    w = abs(np.linalg.det(grid_data['grid_vecs']))
    q_pseudo = grid_data['pseudo_density'].sum()*w
    q_corr = grid_data['charge_corrections'].sum()
    assert np.allclose(q_pseudo, -q_corr)

    if calc.wfs.nspins == 2:
        # some checks specific for spin-polarized results
        assert (grid_data['pseudo_spindensity'].shape == grid_data['shape']).all()
        assert (grid_data['ae_spindensity'].shape == grid_data['shape']).all()

    q_ae = grid_data['ae_density'].sum()*w
    assert np.allclose(q_ae, atnums.sum())

    return grid_data


def _get_atomic_grid_data(calc, numbers, coordinates):

    def get_horton_order(ls):
        d = {
            0: np.array([0]),
            1: np.array([1, 2, 0]),
            2: np.array([2, 3, 1, 4, 0]),
            3: np.array([3, 4, 2, 5, 1, 6, 0]),
            4: np.array([4, 5, 3, 6, 2, 7, 1, 8, 0]),
        }
        result = []
        for l in ls:
            local_order = d[l]
            result.extend(local_order + len(result))
        return np.array(result)

    def dump_spline(g, name, y, setup, l):
        import numpy as np
        from scipy.interpolate import CubicSpline

        # Radial grid parameters
        a = setup.rgd.a
        b = setup.rgd.b
        N = setup.rgd.N
        rcut = max(setup.rcut_j)
        shortN = int(np.ceil(rcut/(a+b*rcut)))
        # Correct normalization
        ycorrected = y*np.sqrt((2*l+1)/np.pi)/2

        rtf = HyperbolicRTransform(a, b)
        odg = OneDGrid(np.arange(shortN), np.ones(shortN), (0, shortN))
        rad_short = rtf.transform_1d_grid(odg)
        cs_short = CubicSpline(rad_short.points, ycorrected[:shortN], bc_type='natural')
        setupg[name + '_radgrid'] = rad_short
        setupg[name] = cs_short
        setupg[name + '_rcut'] = rcut

    setups = {}
    atoms = {}

    for iatom, id_setup in enumerate(calc.density.setups.id_a):
        setup = calc.density.setups[iatom]
        setupg_name = '%03i_%s_%s' % id_setup

        if setupg_name not in setups.keys():
            setupg = {}
            setupg['ls'] = setup.l_j
            order = get_horton_order(setupg['ls'])
            setupg['order'] = order
            setupg['overlap'] = setup.dO_ii[order][:,order]

            dump_spline(setupg, 'nc', setup.data.nc_g, setup, 0)
            dump_spline(setupg, 'nct', setup.data.nct_g, setup, 0)
            for iradial, phi_g in enumerate(setup.data.phi_jg):
                l = setupg['ls'][iradial]
                dump_spline(setupg, 'phi_%03i' % iradial, phi_g, setup, l)
            for iradial, phit_g in enumerate(setup.data.phit_jg):
                l = setupg['ls'][iradial]
                dump_spline(setupg, 'phit_%03i' % iradial, phit_g, setup, l)
            setups[setupg_name] = setupg
        else:
            setupg = setups[setupg_name]
            order = setupg['order']

        atomg = {}
        if calc.wfs.nspins == 1:
            atomg['dm'] = unpack2(calc.density.D_asp.get(iatom)[0])[order][:,order]
        else:
            # spin-summed atomic density matrices
            dma = unpack2(calc.density.D_asp.get(iatom)[0])[order][:,order]
            dmb = unpack2(calc.density.D_asp.get(iatom)[1])[order][:,order]
            atomg['dm'] = dma + dmb
            atomg['sdm'] = dma - dmb
        assert atomg['dm'].shape == (setup.ni, setup.ni)
        atomg['id_setup'] = setupg_name

        atoms['%03i' % iatom] = atomg

    return setups, atoms


def compute_uniform_points(grid_data):
    """Computes the trivial positions and weights of the uniform grid points."""
    import numpy as np

    # construct array with point coordinates
    shape = grid_data['shape']
    grid_rvecs = grid_data['grid_vecs']
    npoint = np.product(shape)
    points = np.zeros(tuple(shape) + (3,))
    points += np.outer(np.arange(shape[0]), grid_rvecs[0]).reshape(shape[0],1,1,3)
    points += np.outer(np.arange(shape[1]), grid_rvecs[1]).reshape(1,shape[1],1,3)
    points += np.outer(np.arange(shape[2]), grid_rvecs[2]).reshape(1,1,shape[2],3)

    # for debugging
    if True:
        for i0 in range(shape[0]):
            for i1 in range(shape[1]):
                for i2 in range(shape[2]):
                    assert np.allclose(points[i0,i1,i2], i0*grid_rvecs[0] + i1*grid_rvecs[1] + i2*grid_rvecs[2])
    points.shape = (-1, 3)
    weights = np.empty(len(points))
    weights.fill(abs(np.linalg.det(grid_rvecs)))

    grid_data['grid_points'] = points
    grid_data['grid_weights'] = weights


def denspart_conventions(grid_data, atoms, atnums):

    density = {}
    natom = len(atnums)

    grid_parts = []
    grid_parts.append(GridPart(grid_data, 'pseudo_density'))

    for iatom in range(natom):
        grid_parts.append(GridPart(atoms['%03i' % iatom], ['rhoc_cor', 'rhov_cor']))

    # Count the number of points
    npoint_total = sum(len(grid_part.points) for grid_part in grid_parts)
    # Make numpy arrays
    points = np.zeros((npoint_total, 3))    
    weights = np.zeros((npoint_total,))
    rho = np.zeros((npoint_total,))

    # Fill arrays
    begin = 0
    for grid_part in grid_parts:
        begin = grid_part.store_data(begin, points, weights, rho)
    assert begin == npoint_total

    if grid_data['nspins'] == 2:

        spin_grid_parts = []
        spin_grid_parts.append(GridPart(grid_data, 'pseudo_spindensity'))

        for iatom in range(natom):
            spin_grid_parts.append(GridPart(atoms['%03i' % iatom], 'spinrhov_cor'))

        spinrho = np.zeros((npoint_total,))
        begin = 0
        for grid_part in spin_grid_parts:
            begin = grid_part.store_data(begin, points, weights, spinrho)
        assert begin == npoint_total
        density['spinrho'] = spinrho

    density['points'] = points
    density['weights'] = weights
    density['rho'] = rho

    return density


class GridPart(object):
    def __init__(self, gridpart, densname):
        self.points = gridpart['grid_points']
        self.weights = gridpart['grid_weights']
        if isinstance(densname, list):
            self.rho = np.zeros(gridpart[densname[0]].shape)
            for name in densname:
                self.rho += gridpart[name]
        else:
            self.rho = gridpart[densname]

    def store_data(self, begin, points, weights, rho):
        size = len(self.points)
        end = begin + size
        points[begin:end] = self.points
        weights[begin:end] = self.weights
        rho[begin:end] += self.rho.ravel()
        return end


def fill_pure_polynomials(output, lmax):
    if output.ndim == 1:
        return fill_polynomial_row(output, lmax, 0)
    elif output.ndim == 2:
        shape = output.shape
        output = output.reshape((shape[0] * shape[1]))

        result = 0
        for irep in range(shape[0]):
            result = fill_polynomial_row(output, lmax, shape[1]*irep)

        output = output.reshape((shape[0], shape[1]))
        return result


def fill_polynomial_row(output, lmax, stride):
    if lmax <= 0: return -1
    if lmax <= 1: return 0

    z = output[stride]
    x = output[stride+1]
    y = output[stride+2]

    r2 = x*x + y*y + z*z

    # work arrays to store PI(z,r) polynomials
    pi_old, pi_new, a, b = np.zeros(lmax+1), np.zeros(lmax+1), \
                           np.zeros(lmax+1), np.zeros(lmax+1)

    pi_old[0] = 1
    pi_new[0] = z
    pi_new[1] = 1
    a[1] = x
    b[1] = y

    old_offset = 0     # first array index of the moments of the previous shell
    old_npure = 3      # number of moments in previous shell
    for l in range(2, lmax+1):
        new_npure = old_npure + 2
        new_offset = old_offset + old_npure

        # Polynomials PI(z,r) for current l
        factor = 2*l - 1
        for m in range(l-1):
            tmp = pi_old[m]
            pi_old[m] = pi_new[m]
            pi_new[m] = (z*factor*pi_old[m] - r2*(l+m-1)*tmp)/(l-m)

        pi_old[l-1] = pi_new[l-1]
        pi_new[l] = factor*pi_old[l-1]
        pi_new[l-1] = z*pi_new[l]

        # construct new polynomials A(x,y) and B(x,y)
        a[l] = x*a[l-1] - y*b[l-1]
        b[l] = x*b[l-1] + y*a[l-1]      

        # construct solid harmonics
        output[stride+new_offset] = pi_new[0]
        factor = np.sqrt(2)
        for m in range(1, l+1):
            factor /= np.sqrt((l+m)*(l-m+1))
            output[stride+new_offset+2*m-1] = factor*a[m]*pi_new[m]
            output[stride+new_offset+2*m] = factor*b[m]*pi_new[m]
        old_npure = new_npure
        old_offset = new_offset
    return old_offset


# pylint: disable=protected-access
def _setup_grid(atnums, atcoords, nrad, nang):
    """Set up a simple molecular integration grid for a given molecular geometry.

    Parameters
    ----------
    atnums: np.ndarray(N,)
        Atomic numbers
    atcoords: np.ndarray(N, 3)
        Atomic coordinates.

    Returns
    -------
    grid
        A molecular integration grid, instance (of a subclass of)
        grid.basegrid.Grid.

    """
    print("Setting up grid")
    becke = BeckeWeights(order=3)
    # Fix for missing radii.
    becke._radii[2] = 0.5
    becke._radii[10] = 1.0
    becke._radii[18] = 2.0
    becke._radii[36] = 2.5
    becke._radii[54] = 3.5
    oned = GaussChebyshev(nrad)
    rgrid = BeckeTF(1e-4, 1.5).transform_1d_grid(oned)
    grid = MolGrid.horton_molgrid(atcoords, atnums, rgrid, nang, becke)
    assert np.isfinite(grid.points).all()
    assert np.isfinite(grid.weights).all()
    assert (grid.weights >= 0).all()
    return grid


def _compute_density(iodata, one_rdm, points, chunk_size):
    """Evaluate the density on a give set of grid points.

    Parameters
    ----------
    iodata: IOData
        An instance of IOData, containing an atomic orbital basis set.
    one_rdm: np.ndarray(nbasis, nbasis)
        The one-particle reduced density matrix in the atomic orbital basis.
    points: np.ndarray(N, 3)
        A set of grid points.
    chunk_size
        Number of points on which the density is evaluated in one pass.

    Returns
    -------
    rho
        The electron density on the grid points.

    """
    basis, coord_types = from_iodata(iodata)
    istart = 0
    rho = np.zeros(len(points))
    while istart < len(points):
        print("Computing density: {} / {}".format(istart, len(rho)))
        iend = istart + chunk_size
        rho[istart:iend] = evaluate_density(
            one_rdm, basis, points[istart:iend], coord_type=coord_types
        )
        istart = iend
    assert (rho >= 0).all()
    return rho


def main():
    """Command-line interface."""
    args = parse_args()
    print("Loading file.")

    if '.gpw' in args.fn_wfn:

        from gpaw import restart
        atoms, calc = restart(args.fn_wfn, txt='/dev/null')
        # compute energy
        atoms.get_potential_energy()
        input_data = prepare_input_gpw(atoms, calc, args.grid)
        np.savez(args.fn_rho, **input_data)

    else:

        iodata = load_one(args.fn_wfn)
        grid, rho = prepare_input(iodata, args.nrad, args.nang, args.chunk_size)
        np.savez(
            args.fn_rho,
            **{
                "atcoords": iodata.atcoords,
                "atnums": iodata.atnums,
                "atcorenums": iodata.atcorenums,
                "points": grid.points,
                "weights": grid.weights,
                "rho": rho,
                "cellvecs": np.zeros((0, 3)),
            },
        )


def parse_args():
    """Parse command-line arguments."""
    description = (
        "Setup a default integration grid and compute the density with HORTON3."
    )
    parser = argparse.ArgumentParser(
        prog="denspart-rho-horton3", description=description
    )
    parser.add_argument("fn_wfn", help="The wavefunction file.")
    parser.add_argument(
        "fn_rho",
        help="The NPZ file in which the grid and the " "density will be stored.",
    )
    parser.add_argument(
        "-r",
        "--nrad",
        type=int,
        default=150,
        help="Number of radial grid points. [default=%(default)s]",
    )
    parser.add_argument(
        "-a",
        "--nang",
        type=int,
        default=194,
        help="Number of angular grid points. [default=%(default)s]",
    )
    parser.add_argument(
        "-c",
        "--chunk-size",
        type=int,
        default=10000,
        help="Number points on which the density is computed in one pass. "
        "[default=%(default)s]",
    )
    parser.add_argument(
        "-g",
        "--grid",
        type=str,
        default="ultrafine",
        help="Size of the atom grids on which the density is calculated. "
        "[default=%(default)s]",
    )
    return parser.parse_args()



