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

    Run with defaults (equivalent to ``--cusp-method=project``).

.. option:: --cusp-method=project

    Project out cusp-violating components from each orbital. **Default.**
    When an orbital's relative cusp deviation ``|psi'(0)/psi(0) + Z| / Z`` is
    too large to be repaired (typically a delocalized molecular orbital with a
    wrong-slope tail on a neighbouring nucleus), the converter warns that a
    different basis set should be chosen for that atom.
    See :ref:`cusp-conditions` in the :doc:`adf2stowf` module documentation.

.. option:: --cusp-method=enforce

    Apply cusp correction to active orbitals.
    See :ref:`cusp-conditions` in the :doc:`adf2stowf` module documentation.

.. option:: --cusp-method=none

    Disable cusp correction entirely.
    See :ref:`cusp-conditions` in the :doc:`adf2stowf` module documentation.

.. option:: --plot-cusps

    Enable plotting of cusp constraints (requires Matplotlib).

.. option:: --all-orbitals

    Include virtual orbitals in addition to occupied ones.
    Default: occupied orbitals only.

.. option:: --dump

    Generate a text dump of ``TAPE21.asc``.

The Cartesian-to-spherical conversion itself is exact (no components are
discarded; see :ref:`cart-to-harm`), so it emits no warnings.

Warnings
--------

You may see a warning like::

    WARNING: nuclear cusp at centre 1 (Z=6) deviates by 3.014; the basis cannot represent the cusp at this atom — choose a different basis set for it

This is a nuclear-cusp diagnostic, not a conversion error.  An orbital with a
non-negligible amplitude at the nucleus has a relative cusp deviation
``|psi'(0)/psi(0) + Z| / Z`` too large to repair — typically a delocalized
molecular orbital leaving a wrong-slope tail on a neighbouring nucleus.  It
signals that a different (better cusp-representing) basis set should be chosen
for that atom; it does not indicate a bug in the conversion.
