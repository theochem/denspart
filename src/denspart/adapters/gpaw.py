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
"""Prepare inputs for denspart from a GPAW calculation."""

import argparse

import numpy as np
from ase.units import Bohr
from gpaw import restart
from gpaw.utilities import unpack_density
from grid.atomgrid import AtomGrid
from grid.onedgrid import OneDGrid
from grid.rtransform import HyperbolicRTransform
from numpy.testing import assert_allclose
from scipy.interpolate import CubicSpline

from ..properties import spherical_harmonics

__all__ = ["prepare_input"]


def prepare_input(atoms, calc):
    """Prepare input for denspart from a GPAW run.

    Parameters
    ----------
    atoms
        A list of ASE atoms from a GPAW calculation.
    calc
        The GPAW calculator instance

    Returns
    -------
    input_data
        A dictionary with all input data for a partitioning.

    """
    # Get some basic system information.
    atnums = atoms.get_atomic_numbers()
    atcorenums = atoms.get_atomic_numbers()
    atcoords = atoms.get_positions(wrap=True) / Bohr
    cellvecs = calc.density.gd.cell_cv

    print("Loading uniform grid data")
    uniform_data = get_uniform_grid_data(calc, cellvecs, atnums)
    print("Loading setups & atomic density matrices")
    setups, atoms = get_atomic_grid_data(calc)
    print("Computing corrections in augmentation spheres")
    compute_augmentation_spheres(uniform_data, setups, atoms, atnums, atcoords)
    print("Computing uniform grid info")
    compute_uniform_points(uniform_data)
    print("Convert to denspart arrays")
    density = denspart_conventions(uniform_data, atoms)
    density.update(
        {
            "atcoords": atcoords,
            "atnums": atnums,
            "atcorenums": atcorenums,
            "cellvecs": cellvecs,
        }
    )

    print("Final check")
    print(
        "  total charge = {:10.3e}".format(
            density["atnums"].sum() - np.dot(density["weights"], density["density"])
        )
    )

    return density


def get_uniform_grid_data(calc, cellvecs, atnums):
    """Take the (pseudo) density on the the uniform grid. This is the easy part.

    Parameters
    ----------
    calc
        GPAW calculator instance.
    cellvecs
        3 x 3 array whose rows are cell vectors.
    atnums
        Array with atomic numbers.

    Returns
    -------
    uniform_data
        Dictionary with several items extracted from the GPAW calculator.

    """
    # Parameters that determine the sizes of most grids
    data = {}
    data["shape"] = calc.density.gd.N_c
    data["grid_vecs"] = calc.density.gd.h_cv
    data["nspins"] = calc.wfs.nspins
    for i in range(3):
        assert np.allclose(calc.density.gd.h_cv[i], cellvecs[i] / data["shape"][i])

    # Load (pseudo) density data on the uniform grid.
    if calc.wfs.nspins == 1:
        # Spin-paired case
        data["charge_corrections"] = calc.get_pseudo_density_corrections()
        data["pseudo_density"] = calc.get_pseudo_density() * (Bohr**3)
        # Conversion to atomic units is needed. (?)
        data["ae_density"] = calc.get_all_electron_density(gridrefinement=1) * (Bohr**3)
    else:
        # Spin-polarized case, convert to spin-sum and spin-difference densitities.
        corrections = calc.get_pseudo_density_corrections()
        data["charge_corrections"] = corrections[0] + corrections[1]
        data["spincharge_corrections"] = corrections[0] - corrections[1]

        density_pseudo_alpha = calc.get_pseudo_density(0) * (Bohr**3)
        density_pseudo_beta = calc.get_pseudo_density(1) * (Bohr**3)
        data["pseudo_density"] = density_pseudo_alpha + density_pseudo_beta
        data["pseudo_spindensity"] = density_pseudo_alpha - density_pseudo_beta

        # Conversion to atomic units is needed. (?)
        density_ae_alpha = calc.get_all_electron_density(spin=0, gridrefinement=1) * (Bohr**3)
        density_ae_beta = calc.get_all_electron_density(spin=1, gridrefinement=1) * (Bohr**3)
        data["ae_density"] = density_ae_alpha + density_ae_beta
        data["ae_spindensity"] = density_ae_alpha - density_ae_beta

    # Sanity checks
    assert (data["pseudo_density"].shape == data["shape"]).all()
    assert (data["ae_density"].shape == data["shape"]).all()
    # w = is the quadrature weight for the uniform grid.
    w = abs(np.linalg.det(data["grid_vecs"]))
    q_pseudo = data["pseudo_density"].sum() * w
    q_corr = data["charge_corrections"].sum()
    assert np.allclose(q_pseudo, -q_corr)

    if calc.wfs.nspins == 2:
        # some checks specific for spin-polarized results
        assert (data["pseudo_spindensity"].shape == data["shape"]).all()
        assert (data["ae_spindensity"].shape == data["shape"]).all()
        qspin_pseudo = data["pseudo_spindensity"].sum() * w
        qspin_corr = data["spincharge_corrections"].sum()
        assert np.allclose(qspin_pseudo, -qspin_corr)

    # We're assuming all systems in GPAW are neutral. In fact, this is not strictly True
    # in all cases. We may have to relax this a little.
    q_ae = data["ae_density"].sum() * w
    assert_allclose(q_ae, atnums.sum(), atol=1e-10)

    return data


def get_atomic_grid_data(calc):
    """Load atomic setups and atomic wavefunctions from GPAW calculation.

    Parameters
    ----------
    calc
        GPAW calculator instance.

    Returns
    -------
    setups
        A dictionary with atomic setups used. Keys are atomic numbers and values are
        dictionaries with relevant data for later evaluation of the density corrections
        within the augmentation spheres.
    atoms
        A list with atomic wavefunction data. Contains dm and optional spindm.

    """
    setups = {}
    atoms = []

    for iatom, id_setup in enumerate(calc.density.setups.id_a):
        setup = calc.density.setups[iatom]

        if id_setup not in setups.keys():
            print("  Converting setup", id_setup)
            # We have not encountered it before, time to parse the new setup.
            setup_data = {}
            # Angular momenta of the shells of basis functions.
            setup_data["ls"] = setup.l_j
            order = get_horton2_order(setup_data["ls"])
            setup_data["order"] = order
            # Get the overlap matrix, mostly for debugging.
            setup_data["overlap"] = setup.dO_ii[order][:, order]

            # Dump spline basis functions for nc and nct.
            # These are the core density functions (projected and all-electron).
            dump_spline(setup_data, ("nc",), setup.data.nc_g, setup, 0)
            dump_spline(setup_data, ("nct",), setup.data.nct_g, setup, 0)
            # Dump splines basis for phi and phit.
            # These are the local atomic orbital basis functions (projected and all-electron).
            for iradial, phi_g in enumerate(setup.data.phi_jg):
                ell = setup_data["ls"][iradial]
                dump_spline(setup_data, ("phi", iradial), phi_g, setup, ell)
            for iradial, phit_g in enumerate(setup.data.phit_jg):
                ell = setup_data["ls"][iradial]
                dump_spline(setup_data, ("phit", iradial), phit_g, setup, ell)
            setups[id_setup] = setup_data
        else:
            # Reuse setup that was previously loaded and take the reordering of the
            # basis functions.
            order = setups[id_setup]["order"]

        atom_data = {}
        if calc.wfs.nspins == 1:
            atom_data["dm"] = unpack_density(calc.density.D_asp.get(iatom)[0])[order][:, order]
        else:
            # spin-summed and spin-difference atomic density matrices.
            dma = unpack_density(calc.density.D_asp.get(iatom)[0])[order][:, order]
            dmb = unpack_density(calc.density.D_asp.get(iatom)[1])[order][:, order]
            atom_data["dm"] = dma + dmb
            atom_data["spindm"] = dma - dmb
        assert atom_data["dm"].shape == (setup.ni, setup.ni)
        atom_data["id_setup"] = id_setup

        atoms.append(atom_data)

    return setups, atoms


def get_horton2_order(ells):
    """Return a permutation of the basis functions to obtain HORTON 2 conventions.

    Parameters
    ----------
    ells
        Array with angular momenta of the basis functions.

    Returns
    -------
    permutation
        Reordering of the basis functions.

    """
    local_orders = {
        # Dictionary with reordering of the pure functions to match HORTON 2
        # conventions.
        0: np.array([0]),
        1: np.array([1, 2, 0]),
        2: np.array([2, 3, 1, 4, 0]),
        3: np.array([3, 4, 2, 5, 1, 6, 0]),
        4: np.array([4, 5, 3, 6, 2, 7, 1, 8, 0]),
    }
    result = []
    for ell in ells:
        result.extend(local_orders[ell] + len(result))
    return np.array(result)


def dump_spline(data, key, y, setup, ell):
    """Convert a spline from a GPAW atom setup.

    Parameters
    ----------
    data
        Dictionary in which the spline is stored.
    key
        Used for making dictionary keys.
    y
        Function values at the spline grid points.
    setup
        The GPAW setup to which this spline belongs.
    ell
        Angular momentum.

    """
    # Radial grid parameters
    a = setup.rgd.a
    b = setup.rgd.b
    rcut = max(setup.rcut_j)
    # The following is the size of the grid within the muffin tin sphere.
    size_short = int(np.ceil(rcut / (a + b * rcut)))

    # Create radial grid.
    rtf = HyperbolicRTransform(a, b)
    odg = OneDGrid(np.arange(size_short), np.ones(size_short), (0, size_short))
    rad_short = rtf.transform_1d_grid(odg)
    # Sanity checks
    assert_allclose(rad_short.points, setup.rgd.r_g[:size_short], atol=1e-10)
    assert_allclose(rad_short.weights, setup.rgd.dr_g[:size_short], atol=1e-10)

    # Correct normalization and create spline.
    ycorrected = y * np.sqrt((2 * ell + 1) / np.pi) / 2
    cs_short = CubicSpline(rad_short.points, ycorrected[:size_short], bc_type="natural")

    # Radial grid within the muffin tin sphere
    data[(*key, "radgrid")] = rad_short
    # Cubic spline
    data[(*key, "spline")] = cs_short
    # Radius of the sphere.
    data[(*key, "rcut")] = rcut


def compute_augmentation_spheres(uniform_data, setups, atoms, atnums, atcoords):
    """Compute the density density corrections within the muffin tin spheres on grids.

    Parametes
    ---------
    uniform_data, setups, atoms
        Data generated by get_uniform_grid_data and get_atomic_grid_data.
    atnums
        Atomic numbers
    atcoords
        Atomic (nuclear) coordinates.

    All results are stored in the atoms argument.

    """
    w = abs(np.linalg.det(uniform_data["grid_vecs"]))
    nelec_pseudo = uniform_data["pseudo_density"].sum() * w
    if uniform_data["nspins"] == 2:
        spin_pseudo = uniform_data["pseudo_spindensity"].sum() * w
    natom = len(atnums)

    # Charge corrections are also computed here as a double check.
    qcors = uniform_data["charge_corrections"]
    myqcors = np.zeros(natom)
    if uniform_data["nspins"] == 2:
        sqcors = uniform_data["spincharge_corrections"]
        mysqcors = np.zeros(natom)
    else:
        sqcors = None
        mysqcors = None

    print("  ~~~~~~~  ~~~~~~~~~~~~~  ~~~~~~~~~~~~~  ~~~~~~~~~~~~~")
    print("     Atom  DensPart QCor      GPAW QCor          Error")
    print("  ~~~~~~~  ~~~~~~~~~~~~~  ~~~~~~~~~~~~~  ~~~~~~~~~~~~~")

    for iatom, atom_data in enumerate(atoms):
        setup_data = setups[atom_data["id_setup"]]

        # Do the actual nasty work...
        atgrid_short = eval_correction(atom_data, setup_data)
        atom_data["grid_points"] = atgrid_short.points + atcoords[iatom]
        atom_data["grid_weights"] = atgrid_short.weights

        # Add things up and compare.
        # - core part
        myqcors[iatom] = atgrid_short.integrate(atom_data["density_c_cor"]) - atnums[iatom]
        # - valence part
        vcor = atgrid_short.integrate(atom_data["density_v_cor"])
        myqcors[iatom] += vcor
        print(
            f"  {atnums[iatom]:2d} {iatom:4d}   {myqcors[iatom]:12.7f}"
            f"   {qcors[iatom]:12.7f}   {myqcors[iatom] - qcors[iatom]:12.5e}"
        )

        if sqcors is not None:
            mysqcors[iatom] = atgrid_short.integrate(atom_data["spindensity_v_cor"])
            print(
                f"spin      {mysqcors[iatom]:12.7f}   {sqcors[iatom]:12.7f}"
                f"   {mysqcors[iatom] - sqcors[iatom]:12.5e}"
            )

    print("  ~~~~~~~  ~~~~~~~~~~~~~  ~~~~~~~~~~~~~  ~~~~~~~~~~~~~")

    # Checks on the total charge
    print(f"  GPAW total charge:     {nelec_pseudo + qcors.sum():10.3e}")
    print(f"  DensPart total charge: {nelec_pseudo + myqcors.sum():10.3e}")
    assert_allclose(qcors, myqcors, atol=1e-10)
    if sqcors is not None:
        print(f"  GPAW total spin:       {spin_pseudo + sqcors.sum():10.3e}")
        print(f"  DensPart total spin:   {spin_pseudo + mysqcors.sum():10.3e}")
        assert_allclose(sqcors, mysqcors, atol=1e-10)


def eval_correction(atom_data, setup_data):
    """Compute the pseudo to all-electron corrections for one muffin-tin sphere.

    Parameters
    ----------
    atom_data
        Dictionary with the density matrices.
    setup_data
        Atomic (basis) functions stored on radial grids.

    Returns
    -------
    grid
        The atomic grid for integrations in the muffin tin sphere.

    Notes
    -----
    Conventions used in variable names, following GPAW conventions:
    - with t = pseudo
    - without t = all-electron
    - c = core
    - v = valence

    """
    # Setup atomic grid within the muffin tin sphere.
    radgrid = setup_data[("nc", "radgrid")]
    ells = setup_data["ls"]
    ellmax = max(ells)
    # Twice ellmax is used for the degree of the angular grid, because we include products
    # of two orbitals up to angular momentum ellmax. Those products have up to angular
    # momentum 2 * ellmax.
    grid = AtomGrid(
        radgrid,
        degrees=[2 * ellmax] * radgrid.size,
    )

    d = np.linalg.norm(grid.points, axis=1)

    # Compute the core density correction.
    cs_nc = setup_data[("nc", "spline")]
    cs_nct = setup_data[("nct", "spline")]
    atom_data["density_c"] = cs_nc(d)
    atom_data["density_ct"] = cs_nct(d)
    atom_data["density_c_cor"] = atom_data["density_c"] - atom_data["density_ct"]

    # Compute real spherical harmonics (with Racah normalization) on the grid.
    polys = np.zeros(((ellmax + 1) ** 2 - 1, grid.size), float)
    polys[0] = grid.points[:, 2]
    polys[1] = grid.points[:, 0]
    polys[2] = grid.points[:, 1]
    spherical_harmonics(polys, ellmax, racah=True)

    # Evaluate each pseudo and ae basis function in the atomic grid.
    basis_fns = []
    basist_fns = []
    for iradial, ell in enumerate(ells):
        # Evaluate radial functions.
        phi = setup_data[("phi", iradial, "spline")]
        basis = phi(d)
        phit = setup_data[("phit", iradial, "spline")]
        basist = phit(d)

        # Multiply with the corresponding spherical harmonics
        if ell == 0:
            basis_fns.append(basis)
            basist_fns.append(basist)
        else:
            # Number of spherical harmonics and offset in the polys array.
            nfn = 2 * ell + 1
            offset = ell**2 - 1
            for ifn in range(nfn):
                poly = polys[offset + ifn]
                basis_fns.append(basis * poly)
                basist_fns.append(basist * poly)

    # Sanity check:
    # Construct the local overlap matrix and compare to the one taken from GPAW.
    olp = np.zeros((len(basis_fns), len(basis_fns)))
    olpt = np.zeros((len(basis_fns), len(basis_fns)))
    for ibasis0, (phi0, phit0) in enumerate(zip(basis_fns, basist_fns, strict=True)):
        for ibasis1 in range(ibasis0 + 1):
            phi1 = basis_fns[ibasis1]
            phit1 = basist_fns[ibasis1]
            olp[ibasis0, ibasis1] = grid.integrate(phi0 * phi1)
            olp[ibasis1, ibasis0] = olp[ibasis0, ibasis1]
            olpt[ibasis0, ibasis1] = grid.integrate(phit0 * phit1)
            olpt[ibasis1, ibasis0] = olpt[ibasis0, ibasis1]
    assert_allclose(olp - olpt, setup_data["overlap"], atol=1e-10)

    # Load the atomic density matrix
    dm = atom_data["dm"]
    if "spindm" in atom_data:
        spindm = atom_data["spindm"]
    else:
        spindm = None

    # Loop over all pairs of basis functions and add product times density matrix coeff
    density_v = np.zeros(grid.size)
    density_vt = np.zeros(grid.size)
    if spindm is not None:
        spindensity_v = np.zeros(grid.size)
        spindensity_vt = np.zeros(grid.size)
    for ibasis0, (phi0, phit0) in enumerate(zip(basis_fns, basist_fns, strict=True)):
        for ibasis1 in range(ibasis0 + 1):
            phi1 = basis_fns[ibasis1]
            phit1 = basist_fns[ibasis1]
            factor = (ibasis0 != ibasis1) + 1
            density_v += factor * dm[ibasis0, ibasis1] * phi0 * phi1
            density_vt += factor * dm[ibasis0, ibasis1] * phit0 * phit1
            if spindm is not None:
                spindensity_v += factor * spindm[ibasis0, ibasis1] * phi0 * phi1
                spindensity_vt += factor * spindm[ibasis0, ibasis1] * phit0 * phit1

    # Store electronic valence densities
    density_v_cor = density_v - density_vt
    # Sanity check
    assert np.allclose(grid.integrate(density_v_cor), np.dot((olp - olpt).ravel(), dm.ravel()))
    atom_data["density_v"] = density_v
    atom_data["density_vt"] = density_vt
    atom_data["density_v_cor"] = density_v_cor
    if spindm is not None:
        spindensity_v_cor = spindensity_v - spindensity_vt
        # Sanity check
        assert np.allclose(
            grid.integrate(spindensity_v_cor),
            np.dot((olp - olpt).ravel(), spindm.ravel()),
        )
        atom_data["spindensity_v"] = spindensity_v
        atom_data["spindensity_vt"] = spindensity_vt
        atom_data["spindensity_v_cor"] = spindensity_v_cor

    return grid


def compute_uniform_points(uniform_data):
    """Compute the trivial positions and weights of the uniform grid points."""
    # construct array with point coordinates
    shape = uniform_data["shape"]
    grid_rvecs = uniform_data["grid_vecs"]
    points = np.zeros((*tuple(shape), 3))
    # pylint: disable=too-many-function-args
    points += np.outer(np.arange(shape[0]), grid_rvecs[0]).reshape(shape[0], 1, 1, 3)
    points += np.outer(np.arange(shape[1]), grid_rvecs[1]).reshape(1, shape[1], 1, 3)
    points += np.outer(np.arange(shape[2]), grid_rvecs[2]).reshape(1, 1, shape[2], 3)

    # Check some points.
    npoint = points.size // 3
    for ipoint in range(0, npoint, npoint // 100):
        # pylint: disable=unbalanced-tuple-unpacking
        i0, i1, i2 = np.unravel_index(ipoint, shape)
        assert np.allclose(
            points[i0, i1, i2],
            i0 * grid_rvecs[0] + i1 * grid_rvecs[1] + i2 * grid_rvecs[2],
        )
    points.shape = (-1, 3)
    weights = np.empty(len(points))
    weights.fill(abs(np.linalg.det(grid_rvecs)))

    uniform_data["grid_points"] = points
    uniform_data["grid_weights"] = weights


def denspart_conventions(uniform_data, atoms):
    """Convert all result from all the above functions into a format suitable for denspart.

    Parameters
    ----------
    uniform_data
        Dictionary with detailed data from the uniform grid of a GPAW calculation.
    atoms
        List with dicationaries with atomic grid data.

    Returns
    -------
    density
        Dictionary with just the data needed for running denspart.

    """
    grid_parts = [GridPart(uniform_data, "pseudo_density")]
    print("  Uniform grid size:", grid_parts[0].density.size)
    for atom in atoms:
        grid_parts.append(GridPart(atom, "density_c_cor", "density_v_cor"))
        print("  Atom grid size:", grid_parts[-1].density.size)
    result = {
        "points": np.concatenate([gp.points for gp in grid_parts]),
        "weights": np.concatenate([gp.weights for gp in grid_parts]),
        "density": np.concatenate([gp.density for gp in grid_parts]),
    }
    print("  Total grid size:", result["density"].size)

    if uniform_data["nspins"] == 2:
        spin_grid_parts = [GridPart(uniform_data, "pseudo_spindensity")]
        for atom in atoms:
            spin_grid_parts.append(GridPart(atom, "spindensity_v_cor"))
        result["spindensity"] = np.concatenate([gp.density for gp in grid_parts])

    return result


class GridPart:
    """Helper class for collecting density grid data."""

    def __init__(self, data, *densnames):
        self.points = data["grid_points"].reshape(-1, 3)
        self.weights = data["grid_weights"].ravel()
        self.density = sum(data[name] for name in densnames).ravel()


def main(args=None):
    """Command-line interface."""
    args = parse_args(args)
    print("Loading file", args.fn_gpw)
    atoms, calc = restart(args.fn_gpw, txt="/dev/null")
    print("Recomputing the energy to restore internal GPAW data structures.")
    atoms.get_potential_energy()
    density = prepare_input(atoms, calc)
    np.savez_compressed(args.fn_density, **density)


def parse_args(args):
    """Parse command-line arguments."""
    description = "Convert a gpw file from GPAW into denspart input."
    parser = argparse.ArgumentParser(prog="denspart-from-gpaw", description=description)
    parser.add_argument("fn_gpw", help="The wavefunction file.")
    parser.add_argument(
        "fn_density",
        help="The NPZ file in which the grid and the density will be stored.",
    )
    return parser.parse_args(args)


if __name__ == "__main__":
    main()
