Molecular Orbital Expansion
===========================

Final Representation
--------------------

The molecular orbitals written to ``stowfn.data`` are expressed as a linear
combination of STOs:

.. math::

    \psi_i(\mathbf{r}) = \sum_{\mu} C_{\mu i}\, \chi_\mu(\mathbf{r})

where:

* :math:`\mu` — runs over all STO basis functions on all atoms
* :math:`C_{\mu i}` — MO coefficients after cusp and spherical corrections
* :math:`\chi_\mu` — normalised STOs (see :doc:`slater`)

Spin Orbitals
-------------

For spin-polarised calculations the expansion is done separately for
:math:`\alpha` and :math:`\beta` spin channels:

.. math::

    \psi_i^\sigma(\mathbf{r})
    = \sum_{\mu} C_{\mu i}^\sigma\, \chi_\mu(\mathbf{r}),
    \qquad \sigma \in \{\alpha, \beta\}

Occupied vs Virtual Orbitals
-----------------------------

By default only occupied orbitals are written. Virtual orbitals can be
included with ``--all-orbitals``.

The occupation threshold follows the ADF output: orbitals with occupation
number greater than zero are considered occupied.

Output Format
-------------

The ``stowfn.data`` file contains for each orbital :math:`i`:

1. The orbital index and spin channel
2. The number of basis functions :math:`N_\mu`
3. For each basis function: atom index, :math:`n`, :math:`l`, :math:`m`,
   exponent :math:`\zeta`, and coefficient :math:`C_{\mu i}`

This format is read directly by CASINO when ``atom_basis_type : slater-type``
is set in the input file.

Pipeline Summary
----------------

The full transformation pipeline from TAPE21 to ``stowfn.data`` is:

.. math::

    \mathbf{C}^{\text{cart}}
    \xrightarrow{\text{cart2harm}}
    \mathbf{C}^{\text{sph}}
    \xrightarrow{\text{cusp}}
    \tilde{\mathbf{C}}^{\text{sph}}
    \xrightarrow{\text{normalise}}
    \mathbf{C}^{\text{stowfn}}
