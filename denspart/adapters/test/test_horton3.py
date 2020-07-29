"""Test the input preparation with HORTON3 modules."""

import pytest
from importlib.resources import path

from numpy.testing import assert_allclose

from iodata import load_one
from iodata.utils import FileFormatWarning

from ..horton3 import prepare_input


filenames = [
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


@pytest.mark.parametrize("fn_wfn", filenames)
def test_integrate_rho(fn_wfn):
    with path("iodata.test.data", fn_wfn) as fn_full:
        with pytest.warns(None) as record:
            iodata = load_one(str(fn_full))
        if len(record) == 1:
            assert issubclass(record[0].category, FileFormatWarning)
    grid, rho = prepare_input(iodata)
    assert_allclose(grid.integrate(rho), iodata.mo.nelec, atol=1e-2)
