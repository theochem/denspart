"""Compute atom-in-molecules properties."""


import numpy as np


def compute_rcubed(pro_model, grid, rho, localgrids):
    """Compute expectation values of r^3 for each atom.

    Parameters
    ----------
    pro_model
        The model for the pro-molecular density, an instance of ``ProModel``.
    grid
        The whole integration grid, instance of ``grid.basegrid.Grid.``
    rho
        The electron density.
    localgrids
        A list of local grids, one for each basis function.

    Returns
    -------
    rcubed
        An array with expectation values of r^3.

    """
    pro = pro_model.compute_density(grid, localgrids)
    result = np.zeros(pro_model.natom)
    for iatom, atcoord in enumerate(pro_model.atcoords):
        # TODO: improve cutoff
        localgrid = grid.get_localgrid(atcoord, 8.0)
        dists = np.linalg.norm(localgrid.points - atcoord, axis=1)
        pro_atom = pro_model.compute_proatom(iatom, localgrid)
        ratio = pro_atom / pro[localgrid.indices]
        result[iatom] = localgrid.integrate(rho[localgrid.indices], dists ** 3, ratio)
    return result
