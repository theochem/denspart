DensPart
########


DensPart is an atoms-in-molecules density partitioning program. At the moment, it only
features one method to partition the density, namely the Minimal Basis Iterative
Stockholder (MBIS) scheme. See http://dx.doi.org/10.1021/acs.jctc.6b00456

**Disclaimer:** This implementation is a prototype and is not extensively tested yet.
Future revisions may break backward compatibility of the API and file formats.


Minimal setup
=============

Required dependencies:

- NumPy: https://numpy.org
- QC-Grid: https://github.com/theochem/grid

Install (with dependencies):

.. code-block:: bash

    pip install git+https://github.com/theochem/grid.git
    pip install git+https://github.com/theochem/denspart.git

(There are no releases yet.)


Usage
=====

One needs to construct a ``density.npz`` file, which is used as input for the ``denspart``
script. (The optional dependencies below provide convenient tools to make such files.)

The file ``density.npz`` uses the NumPy ZIP (NPZ) format, which is a simple container file
format for arrays. More details on NPZ can be found here in `the NumPy documentation
<https://numpy.org/doc/stable/reference/routines.io.html>`_. The file ``density.npz``
should contain at least the following arrays:

- ``points``: Quadrature grid points, shape ``(npoint, 3)``.
- ``weights``: Quadrature grid weights, shape ``(npoint, )``.
- ``density``: Electron density at the grid points, shape ``(npoint, )``.
- ``atnums``: Atomic numbers, shape ``(natom, )``.
- ``atcoords``: Nuclear coordinates, shape ``(natom, 3)``.
- ``cellvecs``: (Optional) One, two or three cell vectors (rows) defining periodic boundary
  conditions, shape ``(nvec, 3)``.

All data are assumed to be in atomic units.

With a ``density.npz`` file, one can perform the partitioning as follows:

.. code-block:: bash

    denspart density.npz results.npz

The output is stored in ``results.npz``, and contains the following arrays. (These may
be subject to change in future code revisions.)

- Copied from the input file ``density.npz``:

  - ``atnums``: Atomic numbers, shape ``(natom, )``.
  - ``atcoords``: Nuclear coordinates, shape ``(natom, 3)``.

- General outputs:

  - ``atnfn``: The number of pro-density basis functions on each atom, shape ``(natom, )``.
  - ``atnpar``: The number of pro-density parameters for each atom, shape ``(natom, )``.
  - ``charges``: atomic partial charges, shape ``(natom,)``.
  - ``multipole_moments``: Multipole moments (using spherical harmonics), for ``l`` going
    from 1 to 4, shape ``(natom, (lmax + 1)**2 - 1)``. The moments are in HORTON 2 order.

    .. code-block::

        c10 c11 s11
        c20 c21 s21 c22 s22
        c30 c31 s31 c32 s32 c33 s33
        c40 c41 s41 c42 s42 c43 s43 c44 s44

    In this list, the prefix ``c`` denotes cosine-like real spherical harmonics and
    ``s`` denotes the sine-like functions. The first digit refers to the degree ``l`` and
    the second to the order ``m``.
  - ``propars``: The pro-density parameters, shape ``(sum(atnpar), )``.
  - ``radial_moments``: Expectation values of ``r**n``, for ``n`` going from 0 to 4,
    shape ``(natom, 5)``.

- MBIS-specific outputs:

  - ``core_charges``: MBIS core charges, shape ``(natom,)``.
  - ``valence_charges``: MBIS valence charges, shape ``(natom,)``.
  - ``valence_widths``: MBIS valence widths, shape ``(natom,)``.

- Algorithm settings:

  - ``gtol``: A stopping condition that was used for the optimization of the pro-density
    parameters. This is a threshold on the gradient of the extended KL divergence.
  - ``maxiter``: A stopping condition that was used for the optimization of the pro-density
    parameters. This is the maximum number of iterations allowed in SciPy's trust-constr
    minimizer.
  - ``density_cutoff``: A density cutoff parameter that was used to determine the cutoff radii
    for the local integration grids.

The arrays in the ``results.npz`` file can be accessed in Python as follows:

.. code-block:: python

    import numpy as np
    results = np.load("results.npz")
    print("charges", results["charges"])

    # From here, one can convert data to other formats:
    # - CSV
    np.savetxt("charges.csv", results["charges"], delimiter=",")
    # - JSON
    import json
    json.dump(results["charges"].tolist(), open("charges.json", "w"))

    # One can also easily post-process the results with some scripting:
    # - Molecular dipole moment predicted by the atomic charges.
    print(np.dot(results["atcoords"].T, results["charges"]))
    # - Contribution to the molecular dipole moment due to the atomic dipoles.
    #   (This includes a reordering the spherical harmonics.)
    print(results["multipole_moments"][:, [1, 2, 0]].sum(axis=0))




Optional dependencies and interfaces to quantum chemistry codes
===============================================================


IOData
------

See https://github.com/theochem/iodata

Install as follows:

.. code-block:: bash

    pip install git+https://github.com/theochem/iodata.git

When IOData is installed, the npz output of the partitioning can be converted into an
extended XYZ file as follows:

.. code-block:: bash

    denspart-write-extxyz results.npz results.xyz


IOData and GBasis
-----------------

In order to derive a ``density.npz`` from several wavefunction file formats
(wfn, wfx, molden, fchk, ...), one needs install a two dependencies:

- https://github.com/theochem/iodata
- https://github.com/theochem/gbasis

Install as follows:

.. code-block:: bash

    pip install git+https://github.com/theochem/iodata.git
    pip install git+https://github.com/theochem/gbasis.git

Once these are installed, one can compute densities on a grid from a wavefunction file.
For example:

.. code-block:: bash

    denspart-from-horton3 some-file.fchk density.npz

A minimal working example showing how to partition a density from a Gaussian FCHK
can be found in `examples/horton3 <examples/horton3>`_.


GPAW
----

One may also derive a ``density.npz`` file from a
`GPAW <https://wiki.fysik.dtu.dk/gpaw/>`_ calculation.
When GPAW is installed, one can run:

.. code-block:: bash

    denspart-from-gpaw some-file.gpw density.npz

A minimal working example can be found in `examples/gpaw <examples/gpaw>`_.
Note that you may have to add `mpirun` in front of the command.
However, the conversion does not yet support parallel execution and thus only works for the case of a single process, even when using `mpirun`.


ADF (AMS 2021.202)
------------------

One may also derive a ``density.npz`` from an ADF AMSJob.
When `AMS <https://www.scm.com/amsterdam-modeling-suite/>`_ is installed, you can install
denspart in the AMS Python environment as follows:


.. code-block:: bash

    # If needed:
    source ${ADFHOME}/amsbashrc.sh
    # Avoid setting ADF and AMS environment variables manually, because these may change
    # with different versions of AMS.

    amspython -m pip install git+https://github.com/theochem/grid.git
    amspython -m pip install git+https://github.com/theochem/denspart.git
    # For writing the extended XYZ file:
    amspython -m pip install git+https://github.com/theochem/iodata.git


Then, the conversion and partitioning are done as follows:

.. code-block:: bash

    amspython -m denspart.adapters.adf ams.results density.npz
    amspython -m denspart density.npz results.npz
    amspython -m denspart.utils.write_extxyz results.npz results.xyz

where ``ams.results`` is the directory with output files. You need to disable symmetry
and write out the TAPE10 file. More details can be found the the ``denspart.adapters.adf``
module. A minimal working example can be found in `examples/adf <examples/adf>`_.


Psi4
----

By adding a few lines to the `Psi4 <https://psicode.org/>`_ input script, it will write
an NPZ file with Psi4's built-in molecular quadrature grids:

.. code-block:: python

    energy, wfn = psi4.energy(return_wfn=True)
    from denspart.adapters.psi4 import write_density_npz
    write_density_npz(wfn)

Symmetry is not supported, so you need to set the point group to ``c1`` when specifying
the geometry. A minimal working example can be found in `examples/psi4 <examples/psi4>`_.


Development setup
=================

The development environment is configured as follows:

.. code-block:: bash

    # Install the CI driver
    pip install roberto
    # Clone git repo, assuming you have ssh access to github
    git clone git@github.com:theochem/denspart.git
    cd denspart
    # Run first part of the CI, includes making a new test env with all dependencies.
    rob lint-static
    # Activates the development env
    source activate-venv-denspart-dev-python-3.?.sh
    # Install dependencies
    # - Mandatory, but not yet included in setup.py
    pip install git+https://github.com/theochem/grid.git
    # - Optional, for testing and interfaces, not included in setup.py
    pip install --upgrade scipy
    pip install --upgrade git+https://github.com/theochem/iodata.git
    pip install --upgrade git+https://github.com/theochem/gbasis.git
    pip install --upgrade git+https://github.com/tovrstra/pytest-regressions@npz
    pip install --upgrade ase
    # (Make sure BLAS is installed, so GPAW can link with -lblas)
    pip install --upgrade gpaw
