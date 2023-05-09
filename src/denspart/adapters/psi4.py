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
"""Write an density.npz file in a Psi4 job."""


import numpy as np
import psi4


def write_density_npz(wfn, fn_npz="density.npz"):
    """Compute the density on a quadrature grid and write to a DensPart NPZ file.

    Parameters
    ----------
    wfn
        A Psi4 wavefunction object.
    fn_npz
        Filename to write to.

    """
    mol = wfn.molecule()
    if mol.point_group().full_name().lower() != "c1":
        raise ValueError("Symmetry is not supported.")

    restricted = not isinstance(wfn, psi4.core.UHF)

    # In case of a DFT calculation, the required objects are already present.
    try:
        vpot = wfn.V_potential()
    except AttributeError:
        vpot = None

    # If there is no quadrature machiner, make it.
    if vpot is None:
        # We only need the density, so faking in LDA calculation.
        functional, _ = psi4.procrouting.dft.build_superfunctional("SVWN", restricted)
        vpot = psi4.core.VBase.build(wfn.basisset(), functional, "RV" if restricted else "UV")
        if restricted:
            vpot.set_D([wfn.Da()])
        else:
            vpot.set_D([wfn.Da(), wfn.Db()])
        vpot.initialize()

    # Store the current density matrix in the property calculator.
    func = vpot.properties()[0]
    if restricted:
        func.set_pointers(wfn.Da())
    else:
        func.set_pointers(wfn.Da(), wfn.Db())

    # Loop over all blocks and compute the density.
    data = []
    for b in range(vpot.nblocks()):
        block = vpot.get_block(b)
        func.compute_points(block)
        dblock = func.point_values()["RHO_A"].to_array()[: block.npoints()]
        if not restricted:
            dblock += func.point_values()["RHO_B"].to_array()[: block.npoints()]
        tmp = [
            block.x().to_array(),
            block.y().to_array(),
            block.z().to_array(),
            block.w().to_array(),
            dblock,
        ]
        data.append(np.array(tmp).T)
    data = np.concatenate([np.array(part) for part in data])

    # Write the NPZ file.
    molarrays = mol.to_arrays()
    denspart = {
        "points": data[:, :3],
        "weights": data[:, 3],
        "density": data[:, 4],
        "atnums": molarrays[3].astype(int),
        "atcoords": molarrays[0],
    }
    print("Number of electrons", np.dot(data[:, 3], data[:, 4]))
    np.savez_compressed(fn_npz, **denspart)
    print("Written", fn_npz)
