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
"""Test the input preparation with HORTON3 modules."""

import os
from importlib import resources

import numpy as np
import pytest
from iodata.utils import FileFormatWarning
from numpy.testing import assert_allclose

from ..horton3 import main

FILENAMES = [
    "2h-azirine-cc.fchk",
    "2h-azirine-ci.fchk",
    "2h-azirine-mp2.fchk",
    "2h-azirine-mp3.fchk",
    "atom_om2.cp2k.out",
    "atom_si.cp2k.out",
    "carbon_gs_ae_contracted.cp2k.out",
    "carbon_gs_ae_uncontracted.cp2k.out",
    "carbon_gs_pp_contracted.cp2k.out",
    "carbon_gs_pp_uncontracted.cp2k.out",
    "carbon_sc_ae_contracted.cp2k.out",
    "carbon_sc_ae_uncontracted.cp2k.out",
    "carbon_sc_pp_contracted.cp2k.out",
    "carbon_sc_pp_uncontracted.cp2k.out",
    "ch3_hf_sto3g.fchk",
    "ch3_rohf_sto3g_g03.fchk",
    "ethanol.mkl",
    "F.molden",
    "h2_ccpvqz.wfn",
    "h2o.molden.input",
    "h2o_sto3g_decontracted.wfn",
    "h2o_sto3g.fchk",
    "h2o_sto3g.wfn",
    "h2_sto3g.mkl",
    "h2_ub3lyp_ccpvtz.wfx",
    "he2_ghost_psi4_1.0.molden",
    "he_d_orbital.wfn",
    "he_p_orbital.wfn",
    "he_s_orbital.fchk",
    "he_s_orbital.wfn",
    "he_spdfgh_orbital.fchk",
    "he_spdfgh_orbital.wfn",
    "he_spdfgh_virtual.fchk",
    "he_spdfgh_virtual.wfn",
    "he_spdf_orbital.fchk",
    "he_spdf_orbital.wfn",
    "he_spd_orbital.fchk",
    "he_spd_orbital.wfn",
    "he_sp_orbital.fchk",
    "he_sp_orbital.wfn",
    "he_s_virtual.fchk",
    "he_s_virtual.wfn",
    "hf_sto3g.fchk",
    "h_sto3g.fchk",
    "li2_g09_nbasis_indep.fchk",
    "li2.mkl",
    "li2.molden.input",
    "lif_fci.wfn",
    "li_h_3-21G_hf_g09.fchk",
    "lih_cation_cisd.wfn",
    "lih_cation_cisd.wfx",
    "lih_cation_fci.wfn",
    "lih_cation_rohf.wfn",
    "lih_cation_rohf.wfx",
    "lih_cation_uhf.wfn",
    "lih_cation_uhf.wfx",
    "li_sp_orbital.wfn",
    "li_sp_virtual.wfn",
    "monosilicic_acid_hf_lan.fchk",
    "neon_turbomole_def2-qzvp.molden",
    "nh3_molden_cart.molden",
    "nh3_molden_pure.molden",
    "nh3_molpro2012.molden",
    "nh3_orca.molden",
    "nh3_psi4_1.0.molden",
    "nh3_psi4.molden",
    "nh3_turbomole.molden",
    "nitrogen-cc.fchk",
    "nitrogen-ci.fchk",
    "nitrogen-mp2.fchk",
    "nitrogen-mp3.fchk",
    "o2_cc_pvtz_cart.fchk",
    "o2_cc_pvtz_pure.fchk",
    "o2_uhf_virtual.wfn",
    "o2_uhf.wfn",
    "peroxide_irc.fchk",
    "peroxide_opt.fchk",
    "peroxide_relaxed_scan.fchk",
    "peroxide_tsopt.fchk",
    "water_ccpvdz_pure_hf_g03.fchk",
    "water_dimer_ghost.fchk",
    "water_hfs_321g.fchk",
    "water_sto3g_hf_g03.fchk",
    "water_sto3g_hf.wfx",
]


@pytest.mark.parametrize("fn_wfn", FILENAMES)
def test_from_horton3_density(fn_wfn, tmpdir):
    with resources.path("iodata.test.data", fn_wfn) as fn_full:
        fn_density = os.path.join(tmpdir, "density.npz")
        with pytest.warns(None) as record:
            main([str(fn_full), fn_density])
        if len(record) == 1:
            assert issubclass(record[0].category, FileFormatWarning)
        assert os.path.isfile(fn_density)
        data = dict(np.load(fn_density))

    nelec = np.dot(data["density"], data["weights"])
    assert_allclose(nelec, data["nelec"], atol=1e-2)


@pytest.mark.parametrize("fn_wfn", ["hf_sto3g.fchk", "water_sto3g_hf_g03.fchk"])
def test_from_horton3_all(fn_wfn, tmpdir):
    with resources.path("iodata.test.data", fn_wfn) as fn_full:
        fn_density = os.path.join(tmpdir, "density.npz")
        with pytest.warns(None) as record:
            main([str(fn_full), fn_density, "-s", "-g", "-o"])
        if len(record) == 1:
            assert issubclass(record[0].category, FileFormatWarning)
        assert os.path.isfile(fn_density)
        data = dict(np.load(fn_density))

    assert "atom0/points" in data
    assert "density_gradient" in data
    assert "orbitals" in data
    assert "orbitals_gradient" in data
