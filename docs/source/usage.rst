Usage
=====

Workflow
--------

**Step 1.** Run the ADF program::

    adf < adf.in > adf.out

This leaves a binary file ``TAPE21`` in the working directory.

**Step 2.** Convert the binary file to ASCII format::

    dmpkf TAPE21 > TAPE21.asc

The ``dmpkf`` utility is included with the ADF distribution.

**Step 3.** Run the converter in the same directory::

    adf2stowf

This reads ``TAPE21.asc`` and writes ``stowfn.data``.

**Step 4.** Use the output in CASINO by setting::

    atom_basis_type   : slater-type

in the CASINO input file.

Command-Line Options
--------------------

.. option:: adf2stowf

    Run with defaults (equivalent to ``--cusp-method=enforce``).

.. option:: --cusp-method=enforce

    Apply cusp correction to active orbitals. **Default.**
    See :ref:`cusp-conditions` in the :doc:`adf2stowf` module documentation.

.. option:: --cusp-method=project

    Project out cusp-violating components from each orbital.
    See :ref:`cusp-conditions` in the :doc:`adf2stowf` module documentation.

.. option:: --cusp-method=none

    Disable cusp correction entirely.

.. option:: --plot-cusps

    Enable plotting of cusp constraints (requires Matplotlib).

.. option:: --all-orbitals

    Include virtual orbitals in addition to occupied ones.
    Default: occupied orbitals only.

.. option:: --cart2harm-projection

    Enforce pure spherical harmonics via orthogonal projection.
    See :ref:`cart-to-harm` in the :doc:`adf2stowf` module documentation.

.. option:: --dump

    Generate a text dump of ``TAPE21.asc``.

Warnings
--------

You may see a warning like::

    WARNING: cartesian to spherical conversion for spin 0, orb 0 violated by 0.00063567

This means the molecular orbitals (computed in a Cartesian Gaussian basis) contain
non-spherical components — for example, s-type contamination (:math:`x^2+y^2+z^2`)
in d- or f-shells. These components violate angular momentum purity and are unphysical
in a spherical harmonic representation.

To eliminate this warning, use ``--cart2harm-projection``. This applies an orthogonal
projection that removes all constraint-violating components.

.. note::

    Even after projection the total energy may not match the original Cartesian-basis
    energy, because unphysical (but energetically stabilizing) components are removed
    without fully reconstructing the wavefunction in the pure spherical basis.
